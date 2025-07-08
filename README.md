# AyurVeda Wellness Assistant using LangChain, FAISS & LLaMA 3

Welcome to the AyurVeda Wellness Assistant! This intelligent chatbot harnesses the power of LangChain, FAISS, and LLaMA 3 to deliver personalized Ayurvedic guidance rooted in ancient wisdom, all within a beautiful and secure web interface.

---

## 🌟 Project Overview

The AyurVeda Wellness Assistant is designed to:
- 🩺 **Provide Ayurvedic knowledge** about doshas, herbal remedies, daily routines, seasonal wellness, yoga, and more.
- 🧠 **Maintain personalized context** across sessions using Redis-powered memory.
- 🔐 **Enable secure login** with Google OAuth for personalized chat history.
- 🔎 **Search across vector documents** and live web using Tavily and FAISS for accurate, contextual answers.

---

## 🛠️ Tech Stack

- **Python** (core logic and orchestration)
- **LangChain** (agent, tools, memory handling)
- **FAISS** (vector similarity search)
- **HuggingFace Embeddings** (semantic similarity)
- **LLaMA 3 (via Together AI)** (LLM for generating responses)
- **Streamlit** (frontend interface)
- **Upstash Redis** (chat memory persistence)
- **Tavily API** (real-time web search)
- **Google OAuth** (secure login)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AyurVeda-Wellness-Assistant.git
cd AyurVeda-Wellness-Assistant  
```
---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ayurveda-chatbot.git
cd ayurveda-chatbot
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```
### 3. Setup Environment Variables
```bash
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
TOGETHER=your-together-ai-key
TAVILY=your-tavily-api-key
UPSTASH_URL=your-upstash-redis-url
UPSTASH_TOKEN=your-upstash-redis-token
```
### 4. Prepare Vector Store
```bash
AyurVeda-Wellness-Assistant/
└── vectorstore/
```

### 5. Run the App
```bash
streamlit run app.py
```

---
## 📂 Project Structure
```
Ayurveda Chatbot/
├── app.py # Main Streamlit app
├── custom.css # Custom styled theme for Ayurveda UI
├── requirements.txt # All required Python packages
└── vectorstore/ # Local FAISS vector database (not included in repo)
```
---
## 🧪 Example Usage

1. **Log in** with your Google account  
2. Ask questions like:
   - "What is a good morning routine for Kapha dosha?"
   - "Which herbs help with digestion?"
   - "Tell me about Panchakarma detoxification"
3. Receive **context-aware responses** rooted in Ayurvedic knowledge

---

## 🤝 Contributing

We welcome contributions to make this assistant even more helpful!

1. **Fork** the repository  
2. **Create** your feature branch  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
---

## ⚠️ Disclaimers

- This assistant provides **educational information only** about Ayurveda.  
- It is **not a substitute for professional medical advice**.  
- Please consult qualified **Ayurvedic practitioners** for any treatments.  
- It does **not diagnose or treat serious medical conditions**.  
- Always seek **immediate help** in case of emergencies.  

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)  
- [Streamlit](https://streamlit.io/)  
- [HuggingFace](https://huggingface.co/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [Together AI](https://www.together.ai/)  
- [Tavily Search API](https://www.tavily.com/)  
- [Upstash Redis](https://upstash.com/)

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙋‍♂️ Contact

- **GitHub**: [RUGVEDADHIKARI](https://github.com/RUGVEDADHIKARI)  
- **LinkedIn**: [www.linkedin.com/in/rugved-adhikari](https://www.linkedin.com/in/rugved-adhikari)

---
