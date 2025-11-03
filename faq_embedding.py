import os
import re
from tqdm import tqdm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uteis.db_connection import mongo_db

def gerar_faq_embeddings_txt(txt_path: str):
    """
    Lê um arquivo .txt contendo pares de PERGUNTA/RESPOSTA,
    gera embeddings apenas da PERGUNTA e insere o texto completo no MongoDB.
    """
    print(f"Lendo arquivo: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"PERGUNTA:\s*(.*?)\s*RESPOSTA:\s*(.*?)(?:\n---|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        print("Nenhuma pergunta encontrada. Verifique o formato do arquivo.")
        return

    print(f"{len(matches)} perguntas encontradas.")

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    faq_collection = mongo_db["faq_embeddings"]

    print("🧠 Gerando embeddings e salvando no MongoDB...")
    for pergunta, resposta in tqdm(matches, total=len(matches)):
        pergunta = pergunta.strip()
        resposta = resposta.strip()

        texto_completo = f"Pergunta: {pergunta}\nResposta: {resposta}"

        if faq_collection.find_one({"text": texto_completo}):
            continue

        try:
            vector = embeddings_model.embed_query(pergunta)

            faq_collection.insert_one({
                "text": texto_completo,
                "embedding": vector,
                "metadata": {
                    "source": txt_path,
                }
            })
        except Exception as e:
            print(f"Erro ao processar: {pergunta[:60]}... => {e}")

    total = faq_collection.count_documents({'metadata.source': txt_path})
    print(f"\nEmbeddings inseridos com sucesso ({total} perguntas).")

if __name__ == "__main__":
    gerar_faq_embeddings_txt("questions.txt")
