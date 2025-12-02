import os
from typing import List, Dict, Any
import json

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from geopy.geocoders import Nominatim

# ----------- CONFIG -----------

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # change if you're using a local path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# System prompt for the FIRST pass (diagnosis + reasoning)
SYSTEM_PROMPT_1 = """You are a medical reasoning assistant.
You receive a patient's message describing their symptoms and context.
Your job in this first step is to:
1. Carefully interpret the symptoms.
2. List likely differential diagnoses (as hypotheses, not certainties).
3. List the kinds of tests, exams, or follow-up questions a doctor might use.
4. Keep language clinical and structured, not conversational.

Do NOT give final recommendations to the patient yet.
Just structure the reasoning clearly for downstream tools.
"""

# System prompt for the SECOND pass (final patient-facing answer)
SYSTEM_PROMPT_2 = """You are a compassionate, evidence-informed medical assistant.
You are given:
- The patient's original message.
- The model's internal clinical reasoning.
- Additional structured information retrieved from a database.

Your job:
1. Provide a clear, empathetic explanation in plain language.
2. Summarize likely possibilities (NOT definitive diagnoses).
3. Suggest reasonable next steps (e.g., see a doctor, urgent care, ER flags).
4. Encourage the patient to seek professional medical care and clarify
   that this is NOT a substitute for a real doctor.
5. If there are any red-flag symptoms, highlight them gently but firmly.

Be kind, concise, and supportive.
"""


# ----------- MODEL LOADING (once at startup) -----------

print("Loading tokenizer and model... this may take a bit the first time.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
if DEVICE != "cuda":
    model.to(DEVICE)

model.eval()
print("Model loaded.")


# ----------- HELPER FUNCTIONS -----------

def build_mistral_prompt(system_prompt: str, user_message: str) -> str:
    """
    Build a simple Mistral-Instruct style prompt:
    <s>[INST] <system + user> [/INST]
    You can make this more elaborate if you want to include history.
    """
    full_content = f"{system_prompt}\n\nUser message:\n{user_message}"
    return f"<s>[INST] {full_content} [/INST]"


def generate_text(prompt: str,
                  max_new_tokens: int = 512,
                  temperature: float = 0.7,
                  top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt part so we only return new tokens
    generated_ids = outputs[0]
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[prompt_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def query_non_relational_db(intermediate_reasoning: str) -> Dict[str, Any]:
    """
    Placeholder for your non-relational DB (Mongo, Qdrant, Redis, etc.)
    For now, just return a dummy structure. Later you can:
      - parse the intermediate reasoning
      - query your DB for guidelines, similar cases, etc.
    """
    # TODO: replace with real DB logic
    # e.g., use embeddings + vector DB, or directly match conditions
    return {
        "db_notes": "This is a placeholder. Connect your DB here.",
        "matched_conditions": [],
        "extra_recommendations": [],
    }


# ----------- FLASK APP -----------

app = Flask(__name__)
CORS(app)  # allow requests from your React dev server (http://localhost:5173)
geolocator = Nominatim(user_agent="geoapi")  # Initialize the geolocator

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "")
        history = data.get("history", [])
        chat_id = data.get("chat_id", None)
        personal_details = data.get("personal_details", {})  # Extract personal details
        address = personal_details.get("address", "")  # Extract the address
        if address:  # Check if the address is provided
            location = geolocator.geocode(address)  # Get the location details
            if location:  # If geolocation is successful
                personal_details["coordinates"] = {
                "latitude": location.latitude,
                "longitude": location.longitude
            }
            else:
                personal_details["coordinates"] = "Address not found"  # Handle geolocation failure
                personal_details_str = json.dumps(personal_details)  # Convert to JSON string

        user_message_with_details = f"{user_message} | Personal Details: {personal_details_str}"

        if not user_message.strip():
            return jsonify({"error": "Empty message"}), 400

        # --- STEP 1: first-pass reasoning prompt ---
        # You can optionally incorporate history in a smarter way.
        step1_prompt = build_mistral_prompt(
            SYSTEM_PROMPT_1,
            user_message_with_details,
        )

        intermediate_reasoning = generate_text(step1_prompt)

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

        final_answer = generate_text(step2_prompt)

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
    app.run(host="127.0.0.1", port=5000, debug=True)