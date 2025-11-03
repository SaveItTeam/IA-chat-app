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




system_prompt_comum = ("system",
    """
    ### OBJETIVO
    Interpretar a PERGUNTA_ORIGINAL e responder perguntas simples ou relacionadas a estoque.
    A saída SEMPRE é JSON (contrato abaixo) para o Orquestrador.

    ### TAREFAS
    - Responder perguntas básicas: "oi", "olá", "qual seu nome?", "que dia é hoje?", etc.
    - Responder perguntas sobre estoque, produtos, quantidades, códigos, armazenamento, entradas e saídas.
    - Manter a conversa no tema de estoque.
    - Caso o usuário fale sobre algo fora desse tema, responder que ele fugiu do assunto.

    ### CONTEXTO
    - Hoje é {today_local} (America/Sao_Paulo).
    - Entrada do Roteador:
      - ROUTE=comum
      - PERGUNTA_ORIGINAL=...
      - PERSONA=... (use como diretriz de concisão/objetividade)
      - CLARIFY=... (se preenchido, responda primeiro)

    ### REGRAS
    - Use o {chat_history} para entender o contexto recente.
    - Se for uma saudação → responda de forma educada e breve.
    - Se for sobre estoque → responda de maneira clara e objetiva.
    - Se fugir do tema → responda exatamente:
        "Parece que você fugiu do tema de estoque. Quer voltar ao assunto principal?"
    - Nunca invente dados de estoque ou produtos.
    - Seja direto, profissional e simpático.
    - Mantenha respostas curtas e úteis.

    ### SAÍDA (JSON)
    # Obrigatórios:
      - dominio       : "estoque"
      - intencao      : "saudacao" | "consultar" | "atualizar" | "criar" | "informar" | "fora_do_tema"
      - resposta      : frase curta e direta
      - recomendacao  : ação prática (ou string vazia)

    # Opcionais:
      - esclarecer    : pergunta curta de clarificação
      - acompanhamento: follow-up rápido
      - produto       : {"nome":"...", "codigo":"...", "quantidade":"...", "local":"..."}

    ### HISTÓRICO DA CONVERSA
    {chat_history}
    """
)

llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt_comum = ChatPromptTemplate.from_messages([
    system_prompt_comum,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
]).partial(today_local=today.isoformat())


comum_chain = RunnableWithMessageHistory(
    prompt_comum | llm_fast | StrOutputParser(),
    get_session_history=lambda session_id: get_session_history(*session_id.split(":")),
    input_messages_key="input",
    history_messages_key="chat_history",
)