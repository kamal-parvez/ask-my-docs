import httpx
import ollama

MODEL = "qwen3.5:4b"
SYSTEM_PROMPT = "You are a helpful, concise assistant. Keep answers short unless asked for detail."
MAX_TURNS = 10  # keep the last N user/assistant exchanges

HELP = """Commands:
  /reset         clear conversation history
  /model         list installed models
  /model <name>  switch to another Ollama model
  /help          show this help
  /exit          quit (also: exit, quit, Ctrl+C)
"""

NOT_RUNNING = "Can't reach Ollama. Is it running? (try `ollama serve`)\n"


def new_history():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def trim(history):
    """Keep the system prompt plus the last MAX_TURNS exchanges."""
    system, rest = history[:1], history[1:]
    return system + rest[-MAX_TURNS * 2:]


def stream_reply(model, history):
    reply = ""
    # Stream the response and print each chunk as it arrives
    # think=False disables reasoning mode for faster replies
    for chunk in ollama.chat(model=model, messages=history, stream=True, think=False):
        piece = chunk["message"]["content"]
        reply += piece    # accumulate the full reply for the chat history
        print(piece, end="", flush=True)
    return reply


def print_models(current):
    """Print installed models, marking the active one."""
    try:
        models = ollama.list().models
    except (ConnectionError, httpx.ConnectError):
        print(NOT_RUNNING)
        return False
    print("Available models:")
    for m in models:
        marker = "*" if m.model == current else " "
        size_gb = (m.size or 0) / 1e9
        print(f"  {marker} {m.model}  ({size_gb:.1f} GB)")
    print("(* = current)\n")
    return True


def switch_model(current, name):
    try:
        installed = [m.model for m in ollama.list().models]
    except (ConnectionError, httpx.ConnectError):
        print(NOT_RUNNING)
        return current

    # Ollama treats a bare name like "qwen3.5" as "qwen3.5:latest"
    full = name if ":" in name else f"{name}:latest"
    if full not in installed:
        print(f"Model '{name}' not found.")
        print_models(current)
        return current

    print(f"Switched to {full}.\n")
    return full


def main():
    model = MODEL
    history = new_history()
    print(f"Chatting with {model}. Type /help for commands.\n")

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
            continue
        if user_input == "/reset":
            history = new_history()
            print("History cleared.\n")
            continue
        if user_input.startswith("/model"):
            name = user_input[len("/model"):].strip()
            if name:
                model = switch_model(model, name)
            elif print_models(model):
                print("Usage: /model <name>\n")
            continue
        if user_input.startswith("/"):
            print("Unknown command. Type /help.\n")
            continue

        # Chat
        history.append({"role": "user", "content": user_input})
        history = trim(history)

        print("Bot: ", end="", flush=True)
        try:
            reply = stream_reply(model, history)
        except (ConnectionError, httpx.ConnectError):
            history.pop()
            print(f"\n{NOT_RUNNING}")
            continue
        except ollama.ResponseError as e:
            history.pop()
            print(f"\nOllama error: {e.error}\n")
            continue
        except KeyboardInterrupt:
            history.pop()
            print("\n[interrupted]\n")
            continue
        print("\n")

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
