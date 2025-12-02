import { useState, useRef, useEffect } from 'react'
import './ChatInterface.css'

function ChatInterface({ activeChat, onUpdateChat, onCreateNewChat }) {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const isUpdatingRef = useRef(false)

  useEffect(() => {
    if (!isUpdatingRef.current) {
      if (activeChat) {
        setMessages(activeChat.messages || [])
      } else {
        setMessages([])
      }
    }
  }, [activeChat?.id, activeChat?.messages])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    }

    const messageContent = input.trim()
    setInput('')

    // Create new chat if none exists
    let currentChatId = activeChat?.id
    if (!activeChat) {
      currentChatId = onCreateNewChat()
    }

    const newMessages = [...messages, userMessage]
    isUpdatingRef.current = true
    setMessages(newMessages)
    setIsLoading(true)

    // Update chat with user message
    if (currentChatId) {
      onUpdateChat(currentChatId, {
        messages: newMessages,
        title:
          messages.length === 0 &&
          (!activeChat || activeChat.title === 'New Chat')
            ? messageContent.substring(0, 50)
            : activeChat?.title || 'New Chat'
      })
    }

    try {
      // Retrieve personal details from localStorage
      const personalDetails = JSON.parse(localStorage.getItem('personalDetails'))

      // Call local Flask backend
      const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: messageContent,
          chat_id: currentChatId,
          history: newMessages,
          personal_details: personalDetails // Attach personal details here
        })
      })

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`)
      }

      const data = await response.json()

      const aiMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content:
          data.reply ||
          'The backend returned an empty response. Please check your model server.',
        timestamp: new Date()
      }

      const updatedMessages = [...newMessages, aiMessage]
      setMessages(updatedMessages)

      if (currentChatId) {
        onUpdateChat(currentChatId, { messages: updatedMessages })
      }
    } catch (error) {
      console.error('Error contacting model API:', error)
      const errorMessage = {
        id: Date.now() + 2,
        role: 'assistant',
        content: `Error contacting model API: ${error.message}`,
        timestamp: new Date()
      }
      const updatedMessages = [...newMessages, errorMessage]
      setMessages(updatedMessages)

      if (currentChatId) {
        onUpdateChat(currentChatId, { messages: updatedMessages })
      }
    } finally {
      isUpdatingRef.current = false
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="chat-interface">
      <div className="chat-container">
        {messages.length === 0 ? (
          <div className="landing-page">
            <div className="landing-content">
              <h1 className="landing-title">Welcome to dAIgnosis</h1>
              <div className="suggestions">
                <button 
                  className="suggestion-button"
                  onClick={() => setInput("Am I going to die?")}
                >
                  Am I going to die?
                </button>
                <button 
                  className="suggestion-button"
                  onClick={() => setInput("Explain Cancer.")}
                >
                  Explain Cancer.
                </button>
                <button 
                  className="suggestion-button"
                  onClick={() => setInput("Lebron?")}
                >
                  Lebron?
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="messages-container">
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.role}`}>
                <div className="message-content">
                  <div className="message-role">
                    {message.role === 'user' ? 'You' : 'Assistant'}
                  </div>
                  <div className="message-text">
                    {message.content.split('\n').map((line, i) => (
                      <span key={i}>
                        {line}
                        {i < message.content.split('\n').length - 1 && <br />}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="input-container">
        <form onSubmit={handleSubmit} className="input-form">
          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message..."
              className="chat-input"
              rows={1}
            />
            <button 
              type="submit" 
              className="send-button"
              disabled={!input.trim() || isLoading}
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" style={{ display: 'block' }}>
                <path d="M8.99992 16V6.41407L5.70696 9.70704C5.31643 10.0976 4.68342 10.0976 4.29289 9.70704C3.90237 9.31652 3.90237 8.6835 4.29289 8.29298L9.29289 3.29298L9.36907 3.22462C9.76184 2.90427 10.3408 2.92686 10.707 3.29298L15.707 8.29298L15.7753 8.36915C16.0957 8.76192 16.0731 9.34092 15.707 9.70704C15.3408 10.0732 14.7618 10.0958 14.3691 9.7754L14.2929 9.70704L10.9999 6.41407V16C10.9999 16.5523 10.5522 17 9.99992 17C9.44764 17 8.99992 16.5523 8.99992 16Z" fill="currentColor"/>
              </svg>
            </button>
          </div>
        </form>
        <div className="input-footer">
          <p className="disclaimer">Business Analytics Club NYU Fall 2025</p>
        </div>
      </div>
    </div>
  )
}

export default ChatInterface