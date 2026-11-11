import ollama

import storage
from chat import Chat, ModelNotFound, OllamaUnavailable

HELP = """Commands:
  /reset         start a new chat (the current one stays saved)
  /chats         list saved chats
  /load <n>      continue saved chat number n from /chats
  /model         list installed models
  /model <name>  switch to another Ollama model
  /help          show this help
  /exit          quit (also: exit, quit, Ctrl+C)
"""

NOT_RUNNING = "Can't reach Ollama. Is it running? (try `ollama serve`)\n"


def print_models(chat):
    """Print installed models, marking the active one."""
    try:
        models = chat.list_models()
    except OllamaUnavailable:
        print(NOT_RUNNING)
        return False
    print("Available models:")
    for m in models:
        marker = "*" if m.name == chat.model else " "
        print(f"  {marker} {m.name}  ({m.size_gb:.1f} GB)")
    print("(* = current)\n")
    return True


def handle_model_command(chat, name):
    if not name:
        if print_models(chat):
            print("Usage: /model <name>\n")
        return
    try:
        print(f"Switched to {chat.set_model(name)}.\n")
    except ModelNotFound:
        print(f"Model '{name}' not found.")
        print_models(chat)
    except OllamaUnavailable:
        print(NOT_RUNNING)


def print_chats():
    chats = storage.list_chats()
    if not chats:
        print("No saved chats yet.\n")
        return
    print("Saved chats:")
    for i, c in enumerate(chats, 1):
        when = c.updated[:16].replace("T", " ")
        print(f"  {i}. {c.title}  ({when}, {c.model})")
    print("Use /load <n> to continue one.\n")


def handle_load_command(chat, arg):
    chats = storage.list_chats()
    if not arg.isdigit() or not 1 <= int(arg) <= len(chats):
        print("Usage: /load <n>  (see /chats for the numbers)\n")
        return
    try:
        chat.load(storage.load(chats[int(arg) - 1].id))
    except storage.ChatNotFound:
        print("Couldn't read that chat.\n")
        return
    print(f"Loaded: {chats[int(arg) - 1].title}\n")
    for m in chat.history[1:]:
        print(f"{'You' if m['role'] == 'user' else 'Bot'}: {m['content']}\n")


def reply(chat, user_input):
    print("Bot: ", end="", flush=True)
    try:
        for piece in chat.send(user_input):
            print(piece, end="", flush=True)
    except OllamaUnavailable:
        print(f"\n{NOT_RUNNING}")
        return
    except ollama.ResponseError as e:
        print(f"\nOllama error: {e.error}\n")
        return
    except KeyboardInterrupt:
        print("\n[interrupted]\n")
        return
    print("\n")
    try:
        storage.save(chat)  # auto-save after every completed reply
    except OSError as e:
        print(f"(Couldn't save this chat: {e})\n")


def main():
    chat = Chat()
    print(f"Chatting with {chat.model}. Type /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue

        # Commands
        if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
            break
        if user_input == "/help":
            print(HELP)
        elif user_input == "/reset":
            chat.reset()
            print("Started a new chat.\n")
        elif user_input == "/chats":
            print_chats()
        elif user_input.startswith("/load"):
            handle_load_command(chat, user_input[len("/load"):].strip())
        elif user_input.startswith("/model"):
            handle_model_command(chat, user_input[len("/model"):].strip())
        elif user_input.startswith("/"):
            print("Unknown command. Type /help.\n")
        else:
            # Chat
            reply(chat, user_input)


if __name__ == "__main__":
    main()
