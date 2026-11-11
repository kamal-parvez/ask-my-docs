"""Streamlit web UI"""
import ollama
import streamlit as st

from chat import Chat, ModelNotFound, OllamaUnavailable

st.set_page_config(page_title="AI Chatbot", page_icon="💬")
st.title("💬 AI Chatbot")

# One Chat (history + model) per browser session
if "chat" not in st.session_state:
    st.session_state.chat = Chat()
chat: Chat = st.session_state.chat

NOT_RUNNING = "Can't reach Ollama. Is it running? (try `ollama serve`)"


def on_model_change():
    try:
        chat.set_model(st.session_state.model_select)
    except ModelNotFound:
        st.session_state.model_error = f"Model '{st.session_state.model_select}' not found."
    except OllamaUnavailable:
        st.session_state.model_error = NOT_RUNNING


# Sidebar: model picker + reset
with st.sidebar:
    st.header("Settings")
    try:
        names = [m.name for m in chat.list_models()]
    except OllamaUnavailable:
        names = []
        st.error(NOT_RUNNING)
    if names:
        if chat.model not in names:
            names.insert(0, chat.model)
        st.selectbox(
            "Model",
            names,
            index=names.index(chat.model),
            key="model_select",
            on_change=on_model_change,
        )
    if err := st.session_state.pop("model_error", None):
        st.error(err)
    if st.button("Reset conversation", use_container_width=True):
        chat.reset()
        st.rerun()

# Past messages (index 0 is the system prompt)
for msg in chat.history[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New message
if prompt := st.chat_input("Message the bot..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            st.write_stream(chat.send(prompt))
        except OllamaUnavailable:
            st.error(NOT_RUNNING)
        except ollama.ResponseError as e:
            st.error(f"Ollama error: {e.error}")
