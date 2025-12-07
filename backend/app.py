import os
from typing import List, Dict, Any
import json
import re
import numpy as np

from flask import Flask, request, jsonify
from pymongo import MongoClient
from flask_cors import CORS

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from geopy.geocoders import Nominatim

# ----------- CONFIG -----------

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "dAignosis_db"
COLLECTION_NAME = "embeddings"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # base model used for your LoRA fine-tune
LORA_ADAPTER = "C:\\Users\\Yuyang Hu\\Documents\\bac\\F25\\mistral_finetuned_v3"  # default local LoRA adapter path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Optional: path or HF repo id of a LoRA adapter. Set via env var `LORA_ADAPTER`.
# Examples:
#   - local folder: C:\\path\\to\\lora-adapter
#   - HF Hub repo: username/mistral-7b-instruct-lora
# Preserve hardcoded adapter unless env var provided
_env_lora = os.getenv("LORA_ADAPTER", "").strip()
if _env_lora:
    LORA_ADAPTER = _env_lora
# Optional: merge LoRA weights into the base model for inference (uses more RAM/VRAM)
MERGE_LORA = os.getenv("MERGE_LORA", "0").strip() in {"1", "true", "True"}
# Toggle to enable/disable LoRA application
USE_LORA = os.getenv("USE_LORA", "0").strip() in {"1", "true", "True"}

# Performance/quality toggles
USE_4BIT = os.getenv("USE_4BIT", "0").strip() in {"1", "true", "True"}
USE_8BIT = os.getenv("USE_8BIT", "0").strip() in {"1", "true", "True"}
USE_SDPA = os.getenv("USE_SDPA", "1").strip() in {"1", "true", "True"}
# Default max new tokens for generation
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))

# Embedding with Mistral: mean-pool last hidden states
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "1024"))
print(f"Embedding will use mean-pooled hidden states (max {EMBED_MAX_TOKENS} tokens)")

# System prompt for the FIRST pass (concise differentials only)
SYSTEM_PROMPT_1 = """Task: Produce a compact list of plausible differentials.
Output only concise hypotheses; no explanations or advice.
Format:
- Differential Diagnoses: <comma-separated list of brief condition names>
Constraints:
- Be token-efficient (short names only).
- Avoid dialogue labels or extra text.
- No final recommendations.
"""

# System prompt for the SECOND pass (use RAG locations + guidance)
SYSTEM_PROMPT_2 = """Task: Provide a brief, very human sounding, patient-facing summary and local guidance.
Inputs include: patient's message, internal differentials, and RAG results with clinic/location data.
Goals:
- Summarize likely possibilities (not definitive diagnoses) in plain language.
- Reference nearby clinics/locations from RAG with name + city (or coordinates if no city).
- Suggest practical next steps and red-flag actions.
Format:
- Summary: <2–3 sentences>
- Likely Possibilities: <up to 5 short items>
- Where to Seek Care: <up to 3 locations from RAG; name — city/state or (lat,lon)>
- Next Steps: <2–4 short actions>
Constraints:
- Be concise; no dialogue labels.
- Do not fabricate locations; only use provided RAG entries.
- Avoid definitive diagnoses; use cautious wording.
"""


# ----------- MODEL LOADING (once at startup) -----------

print("Loading tokenizer and model... this may take a bit the first time.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # Ensure generation config aligns with tokenizer
    try:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception:
        pass

# Optional quantization config (4-bit/8-bit). Falls back if unavailable.
quant_config = None
if USE_4BIT or USE_8BIT:
    try:
        from transformers import BitsAndBytesConfig
        if USE_4BIT:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float16,
            )
            print("Quantization: Using 4-bit (NF4) with bfloat16 compute.")
        elif USE_8BIT:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            print("Quantization: Using 8-bit.")
    except Exception as e:
        print(f"Quantization requested but not available, continuing without it: {e}")
        quant_config = None

if quant_config is not None:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )
else:
    # Load base model fully onto target device (avoid meta device placeholders from auto mapping)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=None,            # load all weights, then move explicitly
        low_cpu_mem_usage=False,    # ensure real tensors allocated (avoid meta)
    )
    # Move base model to device
    base_model.to(DEVICE)

# If LoRA is enabled and an adapter is provided, apply it
if USE_LORA and LORA_ADAPTER:
    print(f"Loading LoRA adapter from '{LORA_ADAPTER}' ...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    if MERGE_LORA:
        print("Merging LoRA weights into the base model for inference...")
        model = model.merge_and_unload()
else:
    model = base_model

# Final device move (after potential merge)
if quant_config is None:
    model.to(DEVICE)

# Safety check: ensure no parameter remains on 'meta'
meta_params = [n for n, p in model.named_parameters() if p.device.type == 'meta']
if meta_params:
    raise RuntimeError(f"Model has parameters still on meta device: {meta_params[:5]} ...")

model.eval()
try:
    if USE_SDPA and quant_config is None:
        # Prefer SDPA where available for faster attention; skip when quantized
        model.config.attn_implementation = "sdpa"
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        print("Attention: SDPA enabled.")
    elif USE_SDPA and quant_config is not None:
        print("Attention: Skipping SDPA because 4/8-bit quantization is enabled.")
except Exception as _attn_err:
    print(f"Could not configure attention backend: {_attn_err}")

if torch.cuda.is_available():
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA device: {torch.cuda.get_device_name(0)} | VRAM: {props.total_memory/1e9:.1f} GB")
    except Exception:
        pass

print("Model loaded (no meta tensors).")


# ----------- HELPER FUNCTIONS -----------

def build_mistral_prompt(system_prompt: str, user_message: str) -> str:
    """Build an instruction prompt using the tokenizer's chat template when available.

    Falls back to a minimal plain-text instruction without dialogue labels
    to avoid the model fabricating additional 'User:' or 'Assistant:' turns.
    """
    # Prefer tokenizer chat template (available for Mistral Instruct models)
    try:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass

    # Fallback for non-instruct models: single-instruction format without labels
    return (
        "You are the assistant. Follow the system instructions, then answer the user.\n\n"
        f"System instructions:\n{system_prompt}\n\n"
        f"User message:\n{user_message}\n\n"
        "Write your response below as one assistant message only (no labels).\n\n"
        "Response:"
    )


def _sanitize_model_output(text: str) -> str:
    """Trim any hallucinated dialogue labels from the generated text."""
    if not text:
        return text
    # First, strip any leading dialogue label if it appears at the very start
    # e.g., "Assistant: ..." or "User: ...". This avoids returning an empty
    # string when the label is generated as the first tokens.
    text = re.sub(r"^\s*(User|Assistant|System|Patient):\s*", "", text, flags=re.IGNORECASE)

    stop_markers = [
        "\nUser:", "\nUSER:", "\nPatient:", "\nPATIENT:",
        "\nAssistant:", "\nASSISTANT:", "\nSystem:", "\nSYSTEM:",
    ]
    # Only cut at markers that appear AFTER some content (pos > 0)
    positions = []
    for m in stop_markers:
        pos = text.find(m)
        if pos > 0:
            positions.append(pos)
    if positions:
        text = text[:min(positions)]
    return text.strip()


def _embed_with_mistral(text: str) -> np.ndarray:
    """Compute a single vector embedding using the loaded Mistral model.

    Strategy: tokenize -> forward with output_hidden_states -> mean-pool last hidden
    layer over non-padding tokens via attention mask -> L2-normalize.
    """
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=EMBED_MAX_TOKENS,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]  # [1, seq, hidden]
        attn = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))  # [1, seq]
        attn = attn.unsqueeze(-1)  # [1, seq, 1]
        summed = (last_hidden * attn).sum(dim=1)  # [1, hidden]
        counts = attn.sum(dim=1).clamp(min=1)  # [1, 1]
        mean_pooled = summed / counts
        vec = mean_pooled.squeeze(0).float().detach().cpu().numpy()
        # L2 normalize
        norm = np.linalg.norm(vec) + 1e-12
        vec = (vec / norm).astype(np.float32)
        return vec


def generate_text(prompt: str,
                  max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                  temperature: float = 0.7,
                  top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("INPUT TEXT: ", prompt)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min(16, max_new_tokens),
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt part so we only return new tokens
    generated_ids = outputs[0]
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[prompt_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    text = text.strip()

    # Fallback retry if empty (can happen with some quantized attention backends)
    if not text:
        try:
            outputs2 = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 0.9),
                top_p=max(top_p, 0.95),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_ids2 = outputs2[0]
            new_tokens2 = generated_ids2[prompt_length:]
            text2 = tokenizer.decode(new_tokens2, skip_special_tokens=True).strip()
            if text2:
                return text2
        except Exception:
            pass
    return text


def query_non_relational_db(intermediate_reasoning: str) -> Dict[str, Any]:
    """Query local MongoDB by cosine similarity over stored embeddings.

    Expectations:
      - Each document in `collection` contains an `embedding` field: list[float]
      - Optional metadata fields: `index`, `condition`, `summary`, `red_flags`, `guidelines`
    Returns the 5 most similar matches by cosine similarity.
    """
    try:
        client.admin.command("ping")
    except Exception as e:
        return {
            "db_notes": f"MongoDB not reachable: {e}",
            "matched_conditions": [],
            "extra_recommendations": [],
        }

    reasoning = (intermediate_reasoning or "").strip()
    if not reasoning:
        return {
            "db_notes": "Empty reasoning provided.",
            "matched_conditions": [],
            "extra_recommendations": [],
        }

    # Compute query embedding (normalized) with Mistral
    try:
        q_vec = _embed_with_mistral(reasoning)
    except Exception as e:
        return {
            "db_notes": f"Failed to compute mistral embedding: {e}",
            "matched_conditions": [],
            "extra_recommendations": [],
        }

    # Fetch all document embeddings and metadata
    try:
        cursor = collection.find({}, {
            "embedding": 1,
            "index": 1,
            "condition": 1,
            "summary": 1,
            "red_flags": 1,
            "guidelines": 1,
            "full_text": 1,
            "metadata": 1,
        })
        docs = list(cursor)
    except Exception as e:
        return {
            "db_notes": f"Query error: {e}",
            "matched_conditions": [],
            "extra_recommendations": [],
        }

    emb_list = []
    metas = []
    for d in docs:
        emb = d.get("embedding")
        if emb is None:
            continue
        try:
            vec = np.asarray(emb, dtype=np.float32)
            # Normalize to unit length to use dot as cosine
            norm = np.linalg.norm(vec) + 1e-12
            vec = vec / norm
            emb_list.append(vec)
            metas.append({
                "index": d.get("index", None),
                "condition": d.get("condition"),
                "summary": d.get("summary"),
                "red_flags": d.get("red_flags"),
                "guidelines": d.get("guidelines"),
                "full_text": d.get("full_text"),
                "metadata": d.get("metadata"),
            })
        except Exception:
            continue

    if not emb_list:
        return {
            "db_notes": "No embeddings found in collection.",
            "matched_conditions": [],
            "extra_recommendations": [],
        }

    matrix = np.vstack(emb_list)  # [N, D]
    if matrix.shape[1] != q_vec.shape[0]:
        return {
            "db_notes": f"Embedding dimension mismatch: DB {matrix.shape[1]} vs query {q_vec.shape[0]}",
            "matched_conditions": [],
            "extra_recommendations": [],
        }

    # Cosine similarities via dot product (both normalized)
    sims = matrix @ q_vec.astype(np.float32)
    topk = int(min(5, sims.shape[0]))
    idxs = np.argpartition(-sims, topk - 1)[:topk]
    # Order exactly by similarity descending
    idxs = idxs[np.argsort(-sims[idxs])]

    top_matches = []
    for i in idxs.tolist():
        m = metas[i].copy()
        m["similarity"] = float(sims[i])
        top_matches.append(m)

    # Build extra recommendations from red flags that appear verbatim
    lower_reasoning = reasoning.lower()
    extra_recs: List[str] = []
    for m in top_matches:
        for flag in (m.get("red_flags") or []):
            if isinstance(flag, str) and flag.lower() in lower_reasoning and flag not in extra_recs:
                extra_recs.append(flag)

    return {
        "db_notes": f"Cosine similarity over {matrix.shape[0]} docs using Mistral mean-pooled embeddings.",
        "matched_conditions": top_matches,
        "extra_recommendations": extra_recs,
        "matched_indices": [m.get("index") for m in top_matches],
    }


# ----------- FLASK APP -----------

app = Flask(__name__)
CORS(app)  # allow requests from your React dev server (http://localhost:5173)
geolocator = Nominatim(user_agent="dAignosis-app", timeout=3)  # Initialize with small timeout

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "")
        history = []
        chat_id = data.get("chat_id", None)
        personal_details = data.get("personal_details", {})  # Extract personal details
        address = personal_details.get("address", "")  # Extract the address
        if address:  # Check if the address is provided
            try:
                location = geolocator.geocode(address)  # Get the location details (may timeout)
                if location:  # If geolocation is successful
                    personal_details["coordinates"] = {
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                    }
                else:
                    personal_details["coordinates"] = "Address not found"
            except Exception as geo_err:
                # Do not fail the chat; annotate coordinates as unavailable
                personal_details["coordinates"] = f"Geocoding unavailable: {geo_err.__class__.__name__}"
        # Ensure we always serialize, regardless of address presence or geocode success
        personal_details_str = json.dumps(personal_details)

        user_message_with_details = f"{user_message} | Personal Details: {personal_details_str}"

        if not user_message.strip():
            return jsonify({"error": "Empty message"}), 400

        # --- STEP 1: first-pass reasoning prompt ---
        # You can optionally incorporate history in a smarter way.
        step1_prompt = build_mistral_prompt(
            SYSTEM_PROMPT_1,
            user_message_with_details,
        )

        intermediate_reasoning = _sanitize_model_output(generate_text(step1_prompt))
        print("[MODEL OUTPUT] Intermediate reasoning:\n" + intermediate_reasoning + "\n---")

        # --- STEP 2: query your non-relational DB based on this reasoning ---
        db_results = query_non_relational_db(intermediate_reasoning)

        # --- STEP 3: second-pass final answer prompt ---
        step2_user_block = f"""Patient's original message:
{user_message}

Model's internal clinical reasoning:
{intermediate_reasoning}

Structured information from database:
{db_results}
"""

        step2_prompt = build_mistral_prompt(
            SYSTEM_PROMPT_2,
            step2_user_block,
        )

        final_answer = _sanitize_model_output(generate_text(step2_prompt))
        print("[MODEL OUTPUT] Final answer:\n" + final_answer + "\n=== End Response ===")

        return jsonify({
            "reply": final_answer,
            "chat_id": chat_id,
            "debug": {
                "intermediate_reasoning": intermediate_reasoning,
                "db_results": db_results,
            }
        })

    except Exception as e:
        print("Error in /chat:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run on 127.0.0.1:5000 (matches your frontend fetch URL)
    print(DEVICE, "will be used for model inference.")
    app.run(host="127.0.0.1", port=5000, debug=False)