from flask import Flask, render_template, jsonify, request
from src.helper import embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from src.prompt import *
import os 

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
RETRIEVAL_SCORE_THRESHOLD = float(os.environ.get("RETRIEVAL_SCORE_THRESHOLD", "0.78"))

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
retriever = docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": RETRIEVAL_SCORE_THRESHOLD},
)

llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=200,
    temperature=0.1,
    huggingfacehub_api_token=HF_TOKEN,
    task="conversational",
)
llm = ChatHuggingFace(llm=llm_endpoint)

# Chat history for conversation memory
chat_history = []

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)


@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    global chat_history
    msg = request.form.get("msg", "")
    if not msg.strip():
        return "Please enter a valid question.", 400
    try:
        docs = retriever.get_relevant_documents(msg)
        if not docs:
            refusal = "I don't know based on the provided documents."
            chat_history.append(HumanMessage(content=msg))
            chat_history.append(AIMessage(content=refusal))
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
            return refusal

        response = qa_chain.invoke({"input": msg, "chat_history": chat_history, "context": docs})
        if isinstance(response, str):
            answer = response
        elif hasattr(response, "content"):
            answer = response.content
        elif isinstance(response, dict):
            answer = response.get("answer") or response.get("output_text") or str(response)
        else:
            answer = str(response)

        # Extract unique sources
        sources = set()
        for doc in docs:
            src = doc.metadata.get("source", "")
            if src:
                sources.add(os.path.basename(src))  # just filename, not full path

        if sources:
            answer += "\n\nSources: " + ", ".join(sorted(sources))

        answer += "\n\n⚠️ Disclaimer: This is not a substitute for professional medical advice. Please consult a qualified healthcare provider for serious concerns."

        # Update chat history
        chat_history.append(HumanMessage(content=msg))
        chat_history.append(AIMessage(content=answer))

        # Keep only last 10 exchanges (20 messages) to avoid token overflow
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        return answer
    except Exception as e:
        app.logger.error(f"Error processing query: {e}")
        return "Sorry, something went wrong on our end. Please try again later.", 500


@app.route("/clear", methods=["POST"])
def clear_chat():
    global chat_history
    chat_history = []
    return "Chat history cleared.", 200


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)