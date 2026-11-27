"""Streamlit web UI"""
import tempfile
from pathlib import Path

import ollama
import streamlit as st

import rag
import storage
from chat import Chat, ModelNotFound, OllamaUnavailable
from logging_setup import setup_logging

setup_logging()
st.set_page_config(page_title="Ask My Docs", page_icon="💬", layout="wide")
st.title("💬 Ask My Docs")

# One Chat (history + model) per browser session
if "chat" not in st.session_state:
    st.session_state.chat = Chat(retriever=rag.retrieve_with_prompt)
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


def add_documents(files):
    """Ingest the uploaded files"""
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            path = Path(tmp) / Path(f.name).name  # ingest names the document after the file
            path.write_bytes(f.getvalue())
            try:
                n = rag.ingest(path)
            except rag.RAG_ERRORS as e:
                results.append(("error", f"{path.name}: {rag.error_message(e)}"))
            else:
                results.append(("success", f"Added {path.name} ({n} chunks)."))
    st.session_state.doc_results = results
    st.session_state.uploader_key += 1  # new key = empty uploader


def ask_delete_doc(name):
    st.session_state.pending_doc_delete = name


def cancel_delete_doc():
    st.session_state.pending_doc_delete = None


def confirm_delete_doc(name):
    try:
        rag.delete_document(name)
    except rag.RAG_ERRORS as e:
        st.session_state.doc_results = [("error", rag.error_message(e))]
    st.session_state.pending_doc_delete = None


def show_sources(message):
    if sources := rag.visible_sources(message):
        st.caption("Sources: " + " · ".join(sources))


st.session_state.setdefault("uploader_key", 0)

# Sidebar: new chat, saved chats
with st.sidebar:
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
        delete_col.button("", icon=":material/delete:", key=f"del-{c.id}", help="Delete chat",
                          on_click=ask_delete, args=(c.id,))
        if st.session_state.get("pending_delete") == c.id:
            st.warning(f"Delete “{c.title}” permanently?")
            yes_col, no_col = st.columns(2)
            yes_col.button("Delete", key=f"yes-{c.id}", on_click=confirm_delete, args=(c.id,))
            no_col.button("Cancel", key=f"no-{c.id}", on_click=cancel_delete)

    # Documents the bot can answer from
    st.header("Documents")
    for kind, text in st.session_state.pop("doc_results", []):
        (st.success if kind == "success" else st.error)(text)
    try:
        docs = rag.list_documents()
    except rag.RAG_ERRORS as e:
        docs = None
        st.caption(f"⚠️ {rag.error_message(e)}")
    if docs is not None:
        uploads = st.file_uploader(
            "Add documents", type=sorted(ext.lstrip(".") for ext in rag.SUPPORTED_TYPES),
            accept_multiple_files=True, key=f"uploader-{st.session_state.uploader_key}",
        )
        if uploads and st.button("Add", use_container_width=True):
            with st.spinner("Reading and indexing..."):
                add_documents(uploads)
            st.rerun()
        if not docs:
            st.caption("No documents yet.")
        for name, chunks in sorted(docs.items()):
            name_col, delete_col = st.columns([5, 1])
            name_col.markdown(f"{name}  \n:gray[{chunks} chunks]")
            delete_col.button("", icon=":material/delete:", key=f"deldoc-{name}", help="Remove document",
                              on_click=ask_delete_doc, args=(name,))
            if st.session_state.get("pending_doc_delete") == name:
                st.warning(f"Remove “{name}” permanently?")
                yes_col, no_col = st.columns(2)
                yes_col.button("Remove", key=f"yesdoc-{name}", on_click=confirm_delete_doc, args=(name,))
                no_col.button("Cancel", key=f"nodoc-{name}", on_click=cancel_delete_doc)

# Past messages (index 0 is the system prompt)
for msg in chat.history[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg)

# Chat box with a small model picker underneath
prompt = st.chat_input("Message the bot...")
with st.bottom:
    try:
        names = [m.name for m in chat.list_models()]
    except OllamaUnavailable:
        names = []
    if names:
        if chat.model not in names:
            names.insert(0, chat.model)
        st.session_state.model_select = chat.model  # the chat's model is the source of truth
        _, picker_col = st.columns([4, 1])  # empty left side pushes the picker to the right
        picker_col.selectbox(
            "Model",
            names,
            key="model_select",
            on_change=on_model_change,
            label_visibility="collapsed",
        )
    else:
        st.caption(f"⚠️ {NOT_RUNNING}")
    if err := st.session_state.pop("model_error", None):
        st.error(err)

# New message
if prompt:
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
            if chat.rag_warning:
                st.caption(f"⚠️ {chat.rag_warning}")
            show_sources(chat.history[-1])
            is_new = not storage.exists(chat.id)
            try:
                storage.save(chat)  # auto-save after every completed reply
            except OSError as e:
                st.warning(f"Couldn't save this chat: {e}")
            else:
                if is_new:
                    st.rerun()  # show the new chat in the sidebar list
