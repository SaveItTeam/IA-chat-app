import os
import re
import numpy as np
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------
# Conexão com MongoDB
# ------------------------------------------------------
mongo_client = MongoClient(os.getenv("MONGO_URI"))
faq_collection = mongo_client["faq_db"]["faq_embeddings"]

# ------------------------------------------------------
# Embeddings e LLM
# ------------------------------------------------------
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm_faq = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ------------------------------------------------------
# Prompt corrigido (sem variáveis extras como "nome")
# ------------------------------------------------------
prompt_faq = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Você é o assistente oficial do Save It, responsável por responder perguntas com base no conteúdo do FAQ armazenado no banco de dados.

        Regras:
        - Responda **apenas** com base no contexto fornecido.
        - Seja **claro, direto e educado**.
        - Se a pergunta não tiver relação com o FAQ ou Save It, diga:
          "Essa pergunta foge do escopo do sistema Save It. Posso te ajudar com algo sobre o funcionamento do sistema?"
        - Se o contexto não tiver informações suficientes, diga:
          "Não encontrei informações suficientes no FAQ para responder com segurança."
        """
    ),
    (
        "human",
        "Pergunta: {pergunta}\n\nContexto do FAQ:\n{contexto}\n\nResponda de forma objetiva e confiável."
    )
])

# ------------------------------------------------------
# Função: Busca os textos mais próximos no Mongo
# ------------------------------------------------------
def buscar_contexto_faq(pergunta: str, top_k: int = 5) -> list[str]:
    """Busca os trechos mais similares no MongoDB"""
    pergunta_emb = np.array(embeddings_model.embed_query(pergunta))
    docs = list(faq_collection.find({}, {"text": 1, "embedding": 1}))

    if not docs:
        return ["Nenhum contexto encontrado no FAQ."]

    for doc in docs:
        emb = np.array(doc["embedding"])
        doc["similaridade"] = np.dot(pergunta_emb, emb) / (
            np.linalg.norm(pergunta_emb) * np.linalg.norm(emb)
        )

    melhores = sorted(docs, key=lambda d: d["similaridade"], reverse=True)[:top_k]

    # Debug: imprime similaridades
    for i, m in enumerate(melhores):
        print(f"[{i}] Similaridade: {m['similaridade']:.4f} | Texto: {m['text'][:60]}...")

    return [m["text"] for m in melhores]

# ------------------------------------------------------
# Função: Cria o agente RAG do FAQ
# ------------------------------------------------------
def criar_faq_agent():
    """Cria o agente RAG para FAQ"""
    def responder(pergunta: str):
        trechos = buscar_contexto_faq(pergunta)
        contexto = "\n---\n".join(trechos)
        mensagem = prompt_faq.format_messages(pergunta=pergunta, contexto=contexto)
        resposta = llm_faq.invoke(mensagem)
        return resposta.content.strip()
    return responder

# ------------------------------------------------------
# Teste opcional
# ------------------------------------------------------
if __name__ == "__main__":
    faq_agent = criar_faq_agent()
    pergunta = "Como funciona o Save It?"
    resposta = faq_agent(pergunta)
    print("\n🔹 Resposta final:")
    print(resposta)
