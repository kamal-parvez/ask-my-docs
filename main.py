import ollama

MODEL = "qwen3.5:4b"


def main():
    history = []
    print(f"Chatting with {MODEL}. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user_input})

        print("Bot: ", end="", flush=True)
        reply = ""
        # Stream the response and print each chunk as it arrives
        # think=False disables reasoning mode for faster replies
        for chunk in ollama.chat(model=MODEL, messages=history, stream=True, think=False):
            piece = chunk["message"]["content"]
            reply += piece    # accumulate the full reply for the chat history
            print(piece, end="", flush=True)
        print("\n")

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
