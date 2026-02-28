from flask import Flask, render_template, jsonify, request
from src.helper import embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os 

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not PINECONE_API_KEY:
    raise RuntimeError("Missing required environment variable: PINECONE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("Missing required environment variable: HF_TOKEN")

embeddings = embedding_model()

index_name = "medbot-rag"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-3B-Instruct",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=HF_TOKEN,
    provider="nscale",
    task="conversational",
)
llm = ChatHuggingFace(llm=llm_endpoint)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    if not msg.strip():
        return "Please enter a valid question.", 400
    try:
        response = rag_chain.invoke({"input": msg})
        answer = str(response["answer"])

        # Extract unique sources
        sources = set()
        for doc in response.get("context", []):
            src = doc.metadata.get("source", "")
            if src:
                sources.add(os.path.basename(src))  # just filename, not full path

        if sources:
            answer += "\n\nSources: " + ", ".join(sorted(sources))

        answer += "\n\n⚠️ Disclaimer: This is not a substitute for professional medical advice. Please consult a qualified healthcare provider for serious concerns."

        return answer
    except Exception as e:
        app.logger.error(f"Error processing query: {e}")
        return "Sorry, something went wrong on our end. Please try again later.", 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)