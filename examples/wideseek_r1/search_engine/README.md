# Search Engine Setup

We provide **two search engine backends**:

* **Online search** (web-based)
* **Offline search** (Wikipedia corpus + local vector DB)

During **training**, we use **offline mode** to improve efficiency and reduce cost. During **evaluation**, we switch to **online mode** to maximize retrieval accuracy.



## Online Mode

Online mode uses **[Serper](https://serper.dev)** for `search` and **[Jina](https://jina.ai)** for `access`.

### API Keys

1. **Serper API Key**
   Sign up at [Serper](https://serper.dev), create an API key, then set:

   ```bash
   export SERPER_API_KEY=your_serper_api_key
   ```

2. **Jina API Key**
   Sign up at [Jina](https://jina.ai), create an API key, then set:

   ```bash
   export JINA_API_KEY=your_jina_api_key
   ```

### Configuration

In your training/eval config YAML, set:

```yaml
tools:
  online: True
  use_jina: True
  enable_cache: True
  cache_file: './webpage_cache.json'
```



## Offline Mode

Offline mode uses a **local Qdrant vector database** for retrieval.

### Prerequisites

If you installed the environment via Docker image or `uv`, you only need to install:

```bash
uv pip install qdrant-client
```

### Step 1: Download the Vector DB and Wiki Corpus

Download the corpus from: **[WideSeek-R1-Corpus](https://huggingface.co/datasets/RLinf/WideSeek-R1-Corpus)**

The repository includes three components:

* **`wiki_corpus.jsonl`**: Powers the **Search** tool. Given a query, it returns the most relevant snippets.
* **`wiki_webpages.jsonl`**: Powers the **Access** tool. Given a URL, it returns the full page content.
* **`qdrant/`**: A local **Qdrant** vector database built by embedding `wiki_corpus.jsonl`. This is the core backend for efficient retrieval.

### Step 2: Download the Retriever Model

We use **[E5](https://huggingface.co/intfloat/e5-base-v2)** as the retriever model. Download it from Hugging Face first.

### Step 3: Launch the Retrieval Server

1. **Start the Qdrant database**

   ```bash
   cd /PATH/TO/WideSeek-R1-Corpus/qdrant
   ./qdrant
   ```

   This process stays running. We recommend starting it inside a `tmux` session.

2. **Start the search server**

   Edit the config variables at the top of `launch_qdrant.sh`:

   ```bash
   pages_file=/PATH/TO/WideSeek-R1-Corpus/wiki_webpages.jsonl
   retriever_path=/PATH/TO/e5-model
   ```

   Then run:

   ```bash
   bash examples/search_engine/launch_qdrant.sh
   ```

   This process also stays running, so we recommend running it in another `tmux` session. It continuously listens for agent search requests, embeds queries with E5, retrieves results from Qdrant, and returns them to the agent.

#### Server Endpoints

The server runs on port **8000** and provides two endpoints:

* `POST /retrieve` — vector search
* `POST /access` — full page content lookup

### Configuration

In your training/eval config YAML, set:

```yaml
tools:
  online: False
  search:
    server_addr: "127.0.0.1:8000"
    topk: 3
```

---

## Test Tool Worker

You can test the `search` and `access` tools directly:

### Online mode

Requires `SERPER_API_KEY` and `JINA_API_KEY`:

```bash
python rlinf/agents/tools/tool_worker.py --is_online true
```

### Offline mode

Requires the local retrieval server running on port `8000`:

```bash
python rlinf/agents/tools/tool_worker.py --is_online false
```
