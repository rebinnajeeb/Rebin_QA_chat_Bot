from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import PyPDF2

# ---------- LOAD ENV ----------
load_dotenv()

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Rebin's Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 20px;
    }
    .user-msg {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: right;
        color: white;
    }
    .bot-msg {
        background-color: #111827;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        color: #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<div class='title'>🤖 Rebin's Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask anything... I'm here to help 🚀</div>", unsafe_allow_html=True)

st.divider()

# ---------- CHAT HISTORY ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("📄 Upload a file (txt or pdf)", type=["txt", "pdf"])

# ---------- HANDLE FILE ----------
if uploaded_file and "file_added" not in st.session_state:
    file_text = ""

    if uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")

    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            file_text += page.extract_text()

    # ✅ ADD FILE INTO CHAT HISTORY (ONLY ONCE)
    st.session_state.chat_history.append({
        "role": "user",
        "content": f"[FILE UPLOADED]\n{file_text}"
    })

    st.session_state.file_added = True

# ---------- DISPLAY CHAT ----------
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"<div class='user-msg'>👤 {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {message['content']}</div>", unsafe_allow_html=True)

# ---------- LLM ----------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
)

# ---------- INPUT ----------
user_prompt = st.chat_input("Type your message...")

if user_prompt:
    # save user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_prompt
    })

    st.markdown(f"<div class='user-msg'>👤 {user_prompt}</div>", unsafe_allow_html=True)

    # prepare messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        *st.session_state.chat_history
    ]

    # loader
    with st.spinner("Thinking... 🤔"):
        response = llm.invoke(messages)

    assistant_response = response.content

    # save bot response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_response
    })

    st.markdown(f"<div class='bot-msg'>🤖 {assistant_response}</div>", unsafe_allow_html=True)
