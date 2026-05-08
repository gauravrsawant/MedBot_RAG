# MedBot_RAG

MedBot_RAG is a retrieval-augmented generation (RAG) web app that answers medical questions using your PDF corpus as grounded context. It builds a Pinecone vector index from local PDFs, then serves a Flask chat UI that retrieves relevant chunks and generates responses with a Hugging Face model.

## What this project does

- Ingests PDFs from the data/ folder, cleans metadata, splits text into chunks, and embeds them.
- Stores embeddings in a Pinecone index for fast similarity search.
- Runs a Flask app with a chat UI and a RAG pipeline (retriever + LLM).
- Shows response sources (file names) when available.

## Tech stack

- Python, Flask
- LangChain (retrieval and chains)
- Pinecone (vector database)
- Hugging Face Inference Endpoint (Llama-3.1-8B-Instruct)
- SentenceTransformers embeddings (all-MiniLM-L6-v2)

## Architecture (high level)

1. store_index.py builds and populates the Pinecone index from PDFs.
2. app.py loads the index, sets up retrieval and the LLM, and serves the chat UI.
3. templates/chat.html and static/style.css provide the frontend.

## Project structure

- app.py: Flask app and RAG chain wiring.
- store_index.py: PDF ingestion and Pinecone indexing.
- src/helper.py: data loading, chunking, and embedding model.
- src/prompt.py: system prompt template.
- templates/chat.html: chat UI.
- static/style.css: UI styles.
- data/: place your PDF files here.

## Setup

### 1) Create and activate a virtual environment (optional)

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Set environment variables

Create a .env file in the project root:

```env
PINECONE_API_KEY=your_pinecone_key
HF_TOKEN=your_huggingface_token
```

Optional:

```env
FLASK_DEBUG=true
```

## Index your data

Place PDFs in the data/ folder, then run:

```bash
python store_index.py
```

This creates a Pinecone index named medbot-rag (if it does not exist) and uploads embeddings.

## Run the app

```bash
python app.py
```

The app runs on:

- http://localhost:8080

## Docker

```bash
docker build -t medbot-rag .
docker run -p 8080:8080 --env-file .env medbot-rag
```

## AWS deployment (EC2 + ECR + GitHub Actions)

High-level flow: GitHub Actions builds the Docker image, pushes it to ECR, and the self-hosted runner on EC2 pulls and runs the container.

### 1) Create resources in the AWS Console

- Create an IAM user with access to ECR and EC2.
- Create an ECR repository.
- Create or select an EC2 instance and install Docker.
- Register a GitHub Actions self-hosted runner on the EC2 instance.

### 2) Add GitHub repository secrets

Set these secrets in your GitHub repo:

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION
- ECR_REPO
- PINECONE_API_KEY
- HF_TOKEN

### 3) CI/CD flow

On every push of selected files to GitHub, the workflow:

- Builds the Docker image.
- Pushes the image to ECR.
- Pulls and runs the container on EC2 via the self-hosted runner.

### 4) Security group

Allow inbound TCP 8080 on the EC2 security group.

## Notes

- This project is for informational use and is not a substitute for professional medical advice.
- The LLM response quality depends on the PDF corpus and prompt design.

## Troubleshooting

- Missing API keys: ensure PINECONE_API_KEY and HF_TOKEN are set.
- Empty or weak answers: verify PDFs exist in data/ and re-run store_index.py.
