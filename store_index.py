
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from pinecone import ServerlessSpec
from src.helper import load_pdfs, clean_metadata, split_data, embedding_model
from langchain_pinecone import PineconeVectorStore

load_dotenv() 
pinecone_key = os.getenv("PINECONE_API_KEY")


dataset = load_pdfs("data/")
filered_dataset = clean_metadata(dataset)
text_chunks = split_data(filered_dataset)
model_embedding = embedding_model()

# Initialize the Pinecone client
pc_client = Pinecone(api_key=pinecone_key)

index_name = "medbot-rag"

if not pc_client.has_index(index_name):
    pc_client.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embeddings
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  
    )
index = pc_client.Index(index_name)


addocs = PineconeVectorStore.from_documents(
    documents=text_chunks,     
    embedding=model_embedding,        
    index_name=index_name      
)