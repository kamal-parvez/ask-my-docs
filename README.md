# Ask My Docs

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Ask My Docs is a private question-answering assistant for your documents. It answers questions about your PDF, TXT, and Markdown files and cites the source pages behind each answer.

<!-- Add a screenshot of the web UI here, e.g. ![Ask My Docs web UI](docs/screenshot.png) -->

## Contents

- [How it works](#how-it-works)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Project structure](#project-structure)
- [License](#license)

## How it works

```
Ingestion:  file ─► extract and clean pages ─► chunk ─► embed (nomic-embed-text model) ─► store in Weaviate

Query:      question ─► embed ─► hybrid search (vector + BM25, top k)
                     ─► keep chunks with greater cosine similarity  or a keyword match
                     ─► add chunks to the prompt ─► local LLM (qwen3.5:4b)
                     ─► streamed answer (+ sources when retrieval is confident)
```

## Getting started

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| [Ollama](https://ollama.com/download) | Must be running (`ollama serve`) |
| [Docker](https://docs.docker.com/get-docker/) | Runs the Weaviate vector store |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kamal-parvez/ask-my-docs.git
cd ask-my-docs

# 2. Download the chat and embedding models
ollama pull qwen3.5:4b
ollama pull nomic-embed-text

# 3. Start Weaviate (data is persisted in ./knowledge)
docker compose up -d

# 4. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate       
pip install -r requirements.txt

# 5. (Optional) Create a local settings file
cp .env.example .env
```

### Running

```bash
streamlit run app.py     # web UI, opens at http://localhost:8501
python main.py           # terminal client
```

## Usage

### Web UI

1. Upload one or more documents from the sidebar.
2. Ask questions in the chat. Answers grounded in your documents show a **Sources** line beneath the reply.
3. Use the sidebar to resume a previous chat, delete one (🗑) or manage your documents.
4. Switch models with the selector below the chat input.

### Terminal client

| Command | Description |
|---|---|
| `/ingest <path>` | Add a `.pdf`, `.txt` or `.md` file to the knowledge base |
| `/docs` | List ingested documents |
| `/rmdoc <name>` | Remove a document (asks for confirmation) |
| `/chats` | List saved chats |
| `/load <n>` | Resume saved chat number `n` |
| `/reset` | Start a new chat (the current one remains saved) |
| `/model [name]` | List installed models, or switch to `name` |
| `/help` | Show available commands |
| `/exit` | Quit (also `exit`, `quit` or Ctrl+C) |

## Configuration

All settings can be set as environment variables or in a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `CHATBOT_TEMPERATURE` | `0.7` | Sampling temperature |
| `CHATBOT_MAX_TURNS` | `10` | Number of past exchanges sent to the model |
| `CHATBOT_CHUNK_SIZE` / `CHATBOT_CHUNK_OVERLAP` | `1200` / `120` | Chunk length and overlap, in characters |
| `CHATBOT_RAG_TOP_K` | `4` | Number of chunks retrieved per question |
| `CHATBOT_RAG_ALPHA` | `0.6` | Hybrid search weighting: `1` = vector only, `0` = keyword only |


## Evaluation

[`evals/run_eval.py`](evals/run_eval.py) benchmarks the pipeline on 100 questions from [`philschmid/easyrag-mini-wikipedia`](https://huggingface.co/datasets/philschmid/easyrag-mini-wikipedia), a Wikipedia-based RAG dataset.

### Metrics

- **LLM judge (primary metric):** a model from a different family (`gemma3:12b`) grades each answer as CORRECT or INCORRECT against the reference answer.
- **Retrieval hit:** at least 80% of the reference answer's words appear in a single retrieved chunk. 

### Running the evaluation

```bash
ollama pull gemma3:12b

python evals/run_eval.py --limit 5        # quick check on the first 5 questions
python evals/run_eval.py                  # full run
python evals/run_eval.py --reingest       # rebuild the eval collection after changing chunk settings
python evals/run_eval.py --no-baseline    # skip the answers without RAG
python evals/run_eval.py --rejudge evals/results/<file>.jsonl [--judge-model <model>]   # re-score a saved run
```

Results are written to `evals/results/` as JSON Lines

### Results

Full run with `qwen3.5:4b`, 1200/120 chunks, top-k = 4, alpha = 0.6, temperature = 0:

| Metric | With RAG | Without RAG |
|---|---|---|
| **Judge accuracy** | **88%** | 46% |
| Average answer time | 5.9 s | 3.9 s |

- **Retrieval hit rate:** 81%

## Project structure

```
ask-my-docs/
├── app.py                # Streamlit web UI
├── main.py               # Terminal client
├── chat.py               # Chat session: history, model switching, streaming
├── storage.py            # Save, load and delete chats (JSON)
├── config.py             # Settings from environment variables / .env
├── logging_setup.py      # Logging configuration
├── rag/
│   ├── documents.py      # Read PDF / TXT / MD files and clean pages
│   ├── chunking.py       # Split text into overlapping chunks
│   ├── embeddings.py     # Embeddings via Ollama
│   ├── store.py          # Weaviate storage and hybrid search
│   ├── retrieval.py      # Select relevant chunks and build source citations
│   └── prompting.py      # Build the prompt with retrieved context
├── evals/
│   ├── run_eval.py       # Evaluation runner
│   ├── scoring.py        # Metrics and judge prompt
│   ├── documents.txt     # Evaluation corpus
│   └── questions.csv     # Evaluation questions and reference answers
├── docker-compose.yml    # Weaviate service
├── requirements.txt
└── .env.example
```

## License

The source code is released under the [MIT License](LICENSE).

The evaluation data (`evals/documents.txt` and `evals/questions.csv`) is taken from [`philschmid/easyrag-mini-wikipedia`](https://huggingface.co/datasets/philschmid/easyrag-mini-wikipedia), which is derived from Wikipedia and the [Question-Answer Dataset](https://www.kaggle.com/datasets/rtatman/questionanswer-dataset) by Noah Smith, Michael Heilman, Rebecca Hwa and others (Carnegie Mellon University and the University of Pittsburgh). It is distributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), **not** the MIT License. If you use this data in academic work, please cite:

> Smith, N. A., Heilman, M., & Hwa, R. (2008). *Question generation as a competitive undergraduate course project.* In Proceedings of the NSF Workshop on the Question Generation Shared Task and Evaluation Challenge.
