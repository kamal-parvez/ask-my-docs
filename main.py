import ollama

from chat import Chat, ModelNotFound, OllamaUnavailable

HELP = """Commands:
  /reset         clear conversation history
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
            print("History cleared.\n")
        elif user_input.startswith("/model"):
            handle_model_command(chat, user_input[len("/model"):].strip())
        elif user_input.startswith("/"):
            print("Unknown command. Type /help.\n")
        else:
            # Chat
            reply(chat, user_input)


if __name__ == "__main__":
    main()
