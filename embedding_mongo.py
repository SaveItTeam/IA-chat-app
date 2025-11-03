from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uteis.db_connection import conectar_banco,mongo_db,redis_db
from dotenv import load_dotenv
import os

load_dotenv()

PDF_PATH = "SAVEITFuncionalidades.pdf"

def salvar_embeddings_mongo():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    collection = mongo_db["faq_embeddings"]
    collection.delete_many({})  # limpa anteriores

    for doc in chunks:
        texto = doc.page_content
        vector = embeddings.embed_query(texto)
        collection.insert_one({"texto": texto, "embedding": vector})

    print("embeddings salvos no MongoDB com sucesso.")

if __name__ == "__main__":
    salvar_embeddings_mongo()
