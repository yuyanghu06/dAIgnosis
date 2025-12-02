import { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import './App.css'

function App() {
  const [chats, setChats] = useState([])
  const [activeChatId, setActiveChatId] = useState(null)
  const [personalDetails, setPersonalDetails] = useState(
    JSON.parse(localStorage.getItem('personalDetails')) || null
  )

  const createNewChat = () => {
    const newChat = {
      id: Date.now(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date()
    }
    setChats([newChat, ...chats])
    setActiveChatId(newChat.id)
    return newChat.id
  }

  const updateChat = (chatId, updates) => {
    setChats(chats.map(chat => 
      chat.id === chatId ? { ...chat, ...updates } : chat
    ))
  }

  const activeChat = chats.find(chat => chat.id === activeChatId)

  const handleSavePersonalDetails = (details) => {
    setPersonalDetails(details)
    localStorage.setItem('personalDetails', JSON.stringify(details))
  }

  if (!personalDetails) {
    return (
      <div className="app">
        <PersonalDetailsForm onSave={handleSavePersonalDetails} />
      </div>
    )
  }

  return (
    <div className="app">
      <ChatInterface 
        activeChat={activeChat}
        onUpdateChat={updateChat}
        onCreateNewChat={createNewChat}
      />
    </div>
  )
}

function PersonalDetailsForm({ onSave }) {
  const [formData, setFormData] = useState({
    height: '',
    weight: '',
    race: '',
    address: ''
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData({ ...formData, [name]: value })
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    onSave(formData)
  }

  return (
    <div className="personal-details-form">
      <h1>Enter Your Personal Details</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Height (cm):
          <input
            type="text"
            name="height"
            value={formData.height}
            onChange={handleChange}
            required
          />
        </label>
        <label>
          Weight (kg):
          <input
            type="text"
            name="weight"
            value={formData.weight}
            onChange={handleChange}
            required
          />
        </label>
        <label>
          Race:
          <input
            type="text"
            name="race"
            value={formData.race}
            onChange={handleChange}
            required
          />
        </label>
        <label>
          Address:
          <input
            type="text"
            name="address"
            value={formData.address}
            onChange={handleChange}
            required
          />
        </label>
        <button type="submit">Save</button>
      </form>
    </div>
  )
}

export default App