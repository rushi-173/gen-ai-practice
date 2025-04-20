import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

# Data Ingesting and Preprocessing for RAG

node_pdf_path = Path(__file__).parent / "node-docs.pdf"
node_pdf_loader = PyPDFLoader(file_path=node_pdf_path)
node_docs = node_pdf_loader.load()

express_pdf_path = Path(__file__).parent / "express-docs.pdf"
express_pdf_loader = PyPDFLoader(file_path=express_pdf_path)
express_docs = express_pdf_loader.load()

# Splitting the documents into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Chunk size in tokens
    chunk_overlap=200, # Chunk overlap in tokens
)

split_node_docs = text_splitter.split_documents(documents=node_docs)
split_express_docs = text_splitter.split_documents(documents=express_docs)

# Embedding the chunks and storing in Qdrant

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url=os.getenv("QDRANT_URL"),
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embedding=embedder
)

vector_store.add_documents(documents=split_node_docs)
vector_store.add_documents(documents=split_express_docs)

print("Indexing Process is Done")
