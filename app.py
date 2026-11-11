"""Streamlit web UI"""
import ollama
import streamlit as st

import storage
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


def new_chat():
    chat.reset()
    st.session_state.pending_delete = None


def open_chat(chat_id):
    try:
        chat.load(storage.load(chat_id))
    except storage.ChatNotFound:
        st.session_state.chat_error = "Couldn't read that chat."
        return
    st.session_state.model_select = chat.model  # keep the model picker in sync
    st.session_state.pending_delete = None


def ask_delete(chat_id):
    st.session_state.pending_delete = chat_id


def cancel_delete():
    st.session_state.pending_delete = None


def confirm_delete(chat_id):
    storage.delete(chat_id)
    if chat.id == chat_id:
        chat.reset()  # the chat on screen was deleted, so start fresh
    st.session_state.pending_delete = None


# Sidebar: model picker, new chat, saved chats
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

    st.button("➕ New chat", on_click=new_chat, use_container_width=True)

    st.header("Chats")
    if err := st.session_state.pop("chat_error", None):
        st.error(err)
    saved = storage.list_chats()
    if not saved:
        st.caption("No saved chats yet.")
    for c in saved:
        open_col, delete_col = st.columns([5, 1])
        open_col.button(
            c.title,
            key=f"open-{c.id}",
            on_click=open_chat,
            args=(c.id,),
            type="primary" if c.id == chat.id else "secondary",  # highlight the current chat
            use_container_width=True,
        )
        delete_col.button("🗑", key=f"del-{c.id}", on_click=ask_delete, args=(c.id,))
        if st.session_state.get("pending_delete") == c.id:
            st.warning(f"Delete “{c.title}” permanently?")
            yes_col, no_col = st.columns(2)
            yes_col.button("Delete", key=f"yes-{c.id}", on_click=confirm_delete, args=(c.id,))
            no_col.button("Cancel", key=f"no-{c.id}", on_click=cancel_delete)

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
        else:
            is_new = not storage.exists(chat.id)
            try:
                storage.save(chat)  # auto-save after every completed reply
            except OSError as e:
                st.warning(f"Couldn't save this chat: {e}")
            else:
                if is_new:
                    st.rerun()  # show the new chat in the sidebar list
