import requests
import streamlit as st
import langchain
import os
import time
import re
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain_together import ChatTogether
from langchain_community.chat_models import ChatOllama
import datetime
import hashlib

# Google OAuth
from streamlit_oauth import OAuth2Component

# Custom CSS with improved Ayurvedic theme
def load_css(file_path):
    try:
        with open(file_path, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found. Using default styling.")

# Load environment variables
load_dotenv()
UPSTASH_REDIS_URL = os.getenv("UPSTASH_URL")
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_TOKEN")

oauth2 = OAuth2Component(
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
    token_endpoint="https://oauth2.googleapis.com/token"
)

# Initialize Tavily search
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY")
search = TavilySearchResults()

# Initialize Langchain components with Together AI
if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER")
from langchain_together import ChatTogether

llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.7,
    max_tokens=1024,
)

# System message for Ayurveda ChatBot
system_message = """You are AyurVeda Wellness Assistant, an AI consultant specialized in Ayurvedic medicine, wellness practices, and holistic health.

You provide guidance on:
- Ayurvedic principles and doshas (Vata, Pitta, Kapha)
- Herbal remedies and natural treatments
- Lifestyle recommendations and daily routines (Dinacharya)
- Diet and nutrition based on Ayurvedic principles
- Yoga and meditation practices
- Seasonal wellness practices (Ritucharya)
- Body constitution analysis
- Panchakarma and detoxification methods

You must ONLY provide information related to Ayurveda. If the user asks anything unrelated, politely refuse to answer.
Example refusal: "I'm here to provide information related to Ayurveda. Please let me know how I can assist you with Ayurvedic wellness."

IMPORTANT DISCLAIMERS:
- All advice is for educational purposes only.
- Always recommend consulting qualified Ayurvedic practitioners for personalized treatment.
- Never diagnose or treat serious medical conditions.
- Suggest seeking immediate medical attention for emergencies.
- Emphasize that Ayurveda complements but does not replace modern medicine.

If asked about non-Ayurvedic topics, gently redirect to Ayurvedic wellness and holistic health, and provide a disclaimer that you are only designed to provide information related to Ayurveda.

Provide practical, safe, and authentic Ayurvedic guidance rooted in traditional texts and modern research.
"""

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Load the embeddings model
embeddings = HuggingFaceEmbeddings()
try:
    vectorstore = FAISS.load_local(
        "D:\Ayurveda Chatbot\vectorstore",  
        embeddings, 
        allow_dangerous_deserialization=True
    )
    # Create retriever and tools
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever_tools = create_retriever_tool(
        retriever,
        "Ayurveda_knowledge_search",
        "Search for information about Ayurvedic medicine, treatments, herbs, and wellness practices"
    )
    tools = [search, retriever_tools]
except:
    tools = [search]

# Create the agent
agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "message_history" not in st.session_state:
    st.session_state.message_history = None
if "is_typing" not in st.session_state:
    st.session_state.is_typing = False
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

def create_session_id(email):
    """Create unique session ID from email"""
    return hashlib.md5(email.encode()).hexdigest()

def show_typing_indicator():
    """Enhanced typing indicator with Ayurvedic theme"""
    return st.markdown("""
        <div class="typing-indicator">
            <span class="typing-text">🕉️ AyurVeda Assistant is thinking...</span>
            <div class="loading-dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_loading_spinner():
    """Show loading spinner"""
    return st.markdown("""
        <div class="loading-spinner">
            <div class="spinner"></div>
        </div>
    """, unsafe_allow_html=True)

def history(session_input):
    return UpstashRedisChatMessageHistory(
        url=UPSTASH_REDIS_URL,
        token=UPSTASH_REDIS_TOKEN,
        session_id=session_input,
        ttl=0
    )

def memory(id):
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=history(session_input=id)
    )

def process_chat(agentExecutor, user_input, chat_history):
    """Process chat with enhanced error handling and typing indicator"""
    try:
        response = agentExecutor.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })
        return response["output"]
    except Exception as e:
        error_message = f"🙏 I apologize, but I encountered an issue while processing your request. Please try rephrasing your question about Ayurvedic wellness. \n\nNote: {str(e)}"
        return error_message

def logout_user():
    """Handle user logout"""
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.user_name = ""
    st.session_state.chat_history = []
    st.session_state.message_history = None
    st.session_state.is_typing = False
def main():
    # Load custom CSS before any UI rendering
    load_css("custom.css")
    # Page config
    st.set_page_config(
        page_title="AyurVeda Wellness Assistant",
        page_icon="🕉️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="ayurveda-title">🕉️ AyurVeda Wellness Assistant</h1>
            <p class="ayurveda-subtitle">Your Personal Guide to Holistic Health & Ancient Wisdom</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-title">🌿 Ayurveda Topics</div>
            <div class="sidebar-item">• Dosha Constitution Analysis</div>
            <div class="sidebar-item">• Herbal Remedies & Treatments</div>
            <div class="sidebar-item">• Daily Routines (Dinacharya)</div>
            <div class="sidebar-item">• Seasonal Wellness (Ritucharya)</div>
            <div class="sidebar-item">• Yoga & Meditation</div>
            <div class="sidebar-item">• Ayurvedic Nutrition</div>
            <div class="sidebar-item">• Panchakarma Detox</div>
            <div class="sidebar-item">• Mind-Body Balance</div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="sidebar-title">⚠️ Important Notice</div>
            <div style="font-size: 0.8rem; color: #666; line-height: 1.5;">
                This assistant provides educational information about Ayurveda. 
                Always consult qualified practitioners for personalized treatment 
                and never delay seeking medical attention for serious conditions.
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.logged_in:
            st.markdown(f"""
                <div class="sidebar-title">👤 User Info</div>
                <div style="font-size: 0.9rem; color: #666; line-height: 1.5;">
                    <strong>Name:</strong> {st.session_state.user_name}<br>
                    <strong>Email:</strong> {st.session_state.user_email}<br>
                    <strong>Status:</strong> <span style="color: #28a745;">● Online</span>
                </div>
            """, unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.markdown("""
            <div class="login-wrapper">
                <div class="login-card">
                    <div class="login-title">🚪 Welcome to AyurVeda Wellness</div>
                    <div class="login-subtitle">
                        Please login with your Google account to begin your personalized Ayurvedic wellness journey.<br>
                        Your chat history will be securely saved and accessible across sessions.
                    </div>
                </div>
            </div>
            <div class="login-btn-container"></div>
        """, unsafe_allow_html=True)

        login_btn_container = st.empty()
        with login_btn_container:
            result = oauth2.authorize_button(
                "Login with Google",
                redirect_uri="http://localhost:8501",  
                scope=" ".join(["email", "profile"]),
                key="google_login"
            )
        if result and "token" in result:
            access_token = result["token"]["access_token"]
            try:
                userinfo_response = requests.get(
                    "https://openidconnect.googleapis.com/v1/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                if userinfo_response.status_code == 200:
                    user_info = userinfo_response.json()
                    st.session_state.logged_in = True
                    st.session_state.user_email = user_info["email"]
                    st.session_state.user_name = user_info.get("name", user_info["email"])

                    session_id = create_session_id(st.session_state.user_email)
                    st.session_state.message_history = history(session_id)

                    st.success(f"Welcome, {st.session_state.user_name}!")
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error("Failed to fetch user info from Google.")
                    st.stop()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()
        else:
            st.warning("Please click the login button to authenticate.")
    else:
        session_id = create_session_id(st.session_state.user_email)
        chat_memory = memory(session_id)
        agentExecutor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=chat_memory,
            verbose=True,
            handle_parsing_errors=True
        )
        
        if len(st.session_state.chat_history) == 0:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "🙏 Welcome to AyurVeda Wellness Assistant! How can I support your holistic health journey today?"
            })
        if len(st.session_state.chat_history) >= 2:
            last_user = st.session_state.chat_history[-2]
            last_bot = st.session_state.chat_history[-1]
            if last_user["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><div class="message-header user-header">You</div><div class="message-content">{last_user["content"]}</div></div>',
                    unsafe_allow_html=True
                )
            if last_bot["role"] == "assistant":
                st.markdown(
                    f'<div class="chat-message bot-message"><div class="message-header bot-header">AyurVeda Assistant</div><div class="message-content">{last_bot["content"]}</div></div>',
                    unsafe_allow_html=True
                )
        elif len(st.session_state.chat_history) == 1:
            last_msg = st.session_state.chat_history[-1]
            if last_msg["role"] == "assistant":
                st.markdown(
                    f'<div class="chat-message bot-message"><div class="message-header bot-header">AyurVeda Assistant</div><div class="message-content">{last_msg["content"]}</div></div>',
                    unsafe_allow_html=True
                )

        with st.form("chat_input_form", clear_on_submit=True):
            user_input = st.text_area(
                "Type your question about Ayurveda...",
                placeholder="Ask about doshas, herbs, routines, or any wellness topic...",
                key="user_input"
            )
            send_clicked = st.form_submit_button("Send")

        if send_clicked and user_input.strip():
            # Show typing indicator
            st.session_state.is_typing = True
            show_typing_indicator()
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Process chat and get assistant response
            response = process_chat(agentExecutor, user_input, [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in st.session_state.chat_history if msg["role"] in ["user", "assistant"]
            ])
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.is_typing = False

            # Save to Redis message history if available
            if st.session_state.message_history:
                try:
                    st.session_state.message_history.add_user_message(user_input)
                    st.session_state.message_history.add_ai_message(response)
                except Exception as e:
                    st.warning("Could not save chat history to Redis.")

            st.rerun()

        # Logout button
        st.markdown('<div class="control-buttons">', unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            logout_user()
            st.success("You have been logged out.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
