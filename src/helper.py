from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch


def load_pdfs(pdf_directory):
    # Load PDFs from the specified directory
    loader = DirectoryLoader(pdf_directory, glob="*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents




def clean_metadata(documents: List[Document]) -> List[Document]:
    # Filter documents based on specific criteria (e.g., keywords, metadata)
    filtered_documents: List[Document] = []
    for doc in documents:
        src = doc.metadata.get("source", "")
        # page_num = doc.metadata.get("page_number")
        filtered_documents.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return filtered_documents


def split_data(filered_dataset):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(filered_dataset)
    return text_chunks


def embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name,
                                            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
    return embedding_model