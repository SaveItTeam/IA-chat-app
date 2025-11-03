from zoneinfo import ZoneInfo
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import os
from pymongo import MongoClient
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from agentes.juiz_agent import criar_juiz_agent
from agentes.guardrail_agent import criar_guardrail_agent
from fastapi.middleware.cors import CORSMiddleware
from uteis.utils import get_session_history,salvar_mensagem,build_config
from uteis.db_connection import conectar_banco,mongo_db,redis_db
TZ = ZoneInfo("America/Sao_Paulo")
load_dotenv()
TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()


prompt_analitico = (
    ChatPromptTemplate.from_messages([
        ("system", """
Você é um analista de dados do Aplicativo Save it,.
Seu papel é gerar respostas analíticas e consultas SQL válidas para PostgreSQL com base nas perguntas do usuário.

Aqui está o esquema de tabelas e suas explicações:

1. Product:
   - Armazena informações sobre cada produto.
   - Campos: id, name, description, category, brand, enterprise_id.
   - Cada produto pertence a uma empresa.

2. Batch:
   - Armazena os lotes de produtos.
   - Campos: id, unit_measure, entry_date, batch_code, expiration_date, quantity, max_quantity, product_id.
   - Cada lote está associado a um produto.

3. Stock:
   - Registra movimentações de produtos no estoque.
   - Campos: id, product_id, batch_id, quantity_input, quantity_output, discard_quantity, discard_reason, created_at.
   - Permite calcular vendas, perdas e estoque disponível.

4. Showcase:
   - Representa produtos expostos na Vitrine Virtual.
   - Campos: id, description, batch_id, quantity_showcase, entrance_showcase.
   - Cada vitrine está ligada a um lote.

Regras importantes:
- Sempre filtre os resultados por empresa {enterprise_id}.
- Use aliases curtos (p, b, s, sh).
- Para consultas SQL, retorne apenas a query, sem explicações.
- Para respostas em linguagem natural, transforme o resultado em uma frase clara para o usuário.
"""),
        ("human", "{input}")
    ])
)

llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY")
)


analitico_chain = RunnableWithMessageHistory(
    prompt_analitico | llm_fast | StrOutputParser(),
    get_session_history=lambda session_id: get_session_history(*session_id.split(":")),
    input_messages_key="input",
    history_messages_key="chat_history",
)

def executar_query_analitica(pergunta, empresa_id, funcionario_id, session_id):
    config = build_config(empresa_id, funcionario_id, session_id)

    # Mantemos o mesmo input que seu analítico esperava
    query_sql = analitico_chain.invoke(
        {"input": f"ROUTE=analitico\nPERGUNTA_ORIGINAL={pergunta}\nCLARIFY=", 
         "enterprise_id": empresa_id},  # <- variável agora passada
        config=config
    )

    # Substituímos o placeholder do prompt pela empresa atual
    query_sql = query_sql.replace("{enterprise_id}", str(empresa_id)).replace("```sql", "").replace("```", "").strip()


    try:
        conn = conectar_banco()
        inicio = datetime.now()
        df = pd.read_sql(query_sql, conn)
        fim = datetime.now()
        conn.close()

        if df.empty:
            resposta_texto = "Nenhum dado encontrado para essa consulta."
        else:
            resultado_texto = df.to_string(index=False)
            resposta_texto = llm_fast.invoke([
                ("system", "Você é um assistente que resume resultados SQL em linguagem natural."),
                ("human", f"Pergunta: {pergunta}\nResultado:\n{resultado_texto}\nResuma em uma frase natural e direta.")
            ]).content.strip()

        mongo_db.query_logs.insert_one({
            "empresa_id": empresa_id,
            "funcionario_id": funcionario_id,
            "session_id": session_id,
            "pergunta": pergunta,
            "query_sql": query_sql,
            "resultado": df.to_dict(orient="records"),
            "resposta_gerada": resposta_texto,
            "duracao_ms": (fim - inicio).total_seconds() * 1000,
            "timestamp": datetime.now(TZ)
        })

        return {"query_sql": query_sql, "resposta": resposta_texto}

    except psycopg2.Error as e:
        return {"query_sql": query_sql, "resposta": f"Erro SQL: {e.pgerror}"}
    except Exception as e:
        return {"query_sql": query_sql, "resposta": f"Erro inesperado: {e}"}
