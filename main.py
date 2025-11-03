from zoneinfo import ZoneInfo
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from uuid import uuid4
from pydantic import BaseModel

from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from agentes.juiz_agent import criar_juiz_agent
from agentes.guardrail_agent import criar_guardrail_agent
from fastapi.middleware.cors import CORSMiddleware
from uteis.utils import get_session_history,salvar_mensagem,build_config
from uteis.db_connection import conectar_banco,mongo_db,redis_db
from agentes.analytical import executar_query_analitica
from agentes.faq_agent import criar_faq_agent

TZ = ZoneInfo("America/Sao_Paulo")
load_dotenv()
TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()
from agentes.comun_agent import comum_chain
# ==============================
# 🧠 MODELOS LLM
# ==============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ==============================
# 🧭 ROTEADOR
# ==============================
system_prompt_roteador = ("system", """
Você é o ROteador .

Sua tarefa é **classificar a intenção do usuário** com base na pergunta.  
Você deve escolher **apenas uma rota** entre as três abaixo:

- **comum** → perguntas gerais, explicações simples ou dúvidas sobre o uso do sistema (ex: "como cadastro um produto?", "como ver o estoque atual?").
- **analitico** → perguntas que exigem análise de dados, cálculos ou geração de SQL (ex: "qual produto mais vendeu este mês?", "qual o total vendido no mês passado?").
- **faq** → perguntas que envolvem políticas, informações gerais, instruções ou dúvidas institucionais presentes no documento de FAQ (ex: "qual é o horário de funcionamento?", "como recuperar a senha?", "quem posso contatar para suporte?").

Responda **exatamente neste formato**:

ROUTE=<comum|analitico|faq>  
PERGUNTA_ORIGINAL=<mensagem>  
CLARIFY=<pergunta mínima se precisar de mais contexto, senão deixe vazio>

{chat_history}
""")



example_prompt_base = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}"),
])

shots_roteador = [
    {"human": "Oi", "ai": "Olá! Posso te ajudar com dúvidas gerais ou análises do estoque?"},
    {"human": "Como cadastro um produto?", "ai": "ROUTE=comum\nPERGUNTA_ORIGINAL=Como cadastro um produto?\nCLARIFY="},
    {"human": "Qual produto mais saiu esse mês?", "ai": "ROUTE=analitico\nPERGUNTA_ORIGINAL=Qual produto mais saiu esse mês?\nCLARIFY="},
    {"human": "Qual o total do mês passado?", "ai": "ROUTE=analitico\nPERGUNTA_ORIGINAL=Qual o total do mês passado?\nCLARIFY="},
    {"human": "Como posso recuperar minha senha?", "ai": "ROUTE=faq\nPERGUNTA_ORIGINAL=Como posso recuperar minha senha?\nCLARIFY="}
]


fewshots_roteador = FewShotChatMessagePromptTemplate(
    examples=shots_roteador,
    example_prompt=example_prompt_base
)

prompt_roteador = ChatPromptTemplate.from_messages([
    system_prompt_roteador,
    fewshots_roteador,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
]).partial(today_local=today.isoformat())



# 📊 AGENTE ANALÍTICO (RESTAURO DO PROMPT ORIGINAL)
# =

# ==============================
# ⚙️ CHAINS
# ==============================

router_chain = RunnableWithMessageHistory(
    prompt_roteador | llm_fast | StrOutputParser(),
    get_session_history=lambda session_id: get_session_history(*session_id.split(":")),
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ==============================
# 🤖 EXECUÇÃO PRINCIPAL
# ==============================
def executar_fluxo_estoque(user_input, empresa_id, funcionario_id, session_id):
    config = build_config(empresa_id, funcionario_id, session_id)
    salvar_mensagem(empresa_id, funcionario_id, session_id, "user", user_input)

    rota = router_chain.invoke({"input": user_input, "nome": ""}, config=config)

    if "ROUTE=comum" in rota:
        resposta = comum_chain.invoke({"input": rota}, config=config)

    elif "ROUTE=analitico" in rota:
        pergunta = rota.split("PERGUNTA_ORIGINAL=")[-1].split("\n")[0]
        resultado = executar_query_analitica(pergunta, empresa_id, funcionario_id, session_id)
        resposta = f"Query SQL:\n{resultado['query_sql']}\n\nResposta:\n{resultado['resposta']}"
    elif "ROUTE=faq" in rota:
        pergunta = rota.split("PERGUNTA_ORIGINAL=")[-1].split("\n")[0]
        faq_chain = criar_faq_agent()
        resposta = faq_chain(pergunta)

    else:
        resposta = rota

    guardrail_chain = criar_guardrail_agent(empresa_id)
    if (veredito_guardrail := guardrail_chain.invoke({"input": resposta}, config=config)) != "OK":
        resposta = f"Guardrail: {veredito_guardrail}"

    juiz_chain = criar_juiz_agent(empresa_id)
    if (veredito_juiz := juiz_chain.invoke({"input": resposta}, config=config)) != "OK":
        resposta = f"Alerta do Juiz: {veredito_juiz}"

    salvar_mensagem(empresa_id, funcionario_id, session_id, "assistant", resposta)
    return resposta


app = FastAPI(
    title="SaveIt AI API",
    description="API do Chatbot do aplicativo SaveIt",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens (para teste local)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    user_input: str
    empresa_id: int
    funcionario_id: int
    session_id: str

class IniciarChatRequest(BaseModel):
    empresa_id: int
    funcionario_id: int
@app.get("/")
def root():
    return {"status": "ok", "message": "SaveIt AI API is running"}

# 🔹 Iniciar novo chat
@app.post("/iniciar_chat")
def iniciar_chat(request: IniciarChatRequest):
    try:
        session_id = f"{request.empresa_id}:{request.funcionario_id}:{uuid4()}"
        return JSONResponse(
            content={
                "status": "success",
                "mensagem": "Nova sessão criada",
                "empresa_id": request.empresa_id,
                "funcionario_id": request.funcionario_id,
                "session_id": session_id
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "mensagem": f"Erro ao iniciar chat: {str(e)}"},
            status_code=500
        )

# 🔹 Executar fluxo (POST com JSON)
@app.post("/executar_fluxo")
def executar_fluxo(request: ChatRequest):
    try:
        resultado = executar_fluxo_estoque(
            request.user_input,
            request.empresa_id,
            request.funcionario_id,
            request.session_id
        )
        return JSONResponse(
            content={
                "status": "success",
                "mensagem_usuario": request.user_input,
                "resposta_assistente": resultado,
                "empresa_id": request.empresa_id,
                "funcionario_id": request.funcionario_id,
                "session_id": request.session_id,
                "timestamp": datetime.now(TZ).isoformat()
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "mensagem": f"Erro ao processar: {str(e)}"},
            status_code=500
        )

# 🔹 Obter histórico da sessão
# ==========================================
# 🔹 Histórico por funcionário + sessão
# ==========================================
from fastapi import Body

@app.post("/historico_sessao")
def obter_historico_sessao(
    funcionario_id: int = Body(...),
    session_id: str = Body(...)
):
    """
    Retorna o histórico de uma sessão específica de um funcionário.
    """
    try:
        chat_history = get_session_history(*session_id.split(":"))
        historico_formatado = [
            {
                "tipo": "user" if isinstance(msg, HumanMessage) else "assistant",
                "mensagem": msg.content
            }
            for msg in chat_history.messages
        ]

        return JSONResponse(
            content={
                "status": "success",
                "funcionario_id": funcionario_id,
                "session_id": session_id,
                "historico": historico_formatado
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "mensagem": f"Erro ao obter histórico da sessão: {str(e)}"
            },
            status_code=500
        )


# ==========================================
# 🔹 Histórico geral do funcionário
# ==========================================
@app.post("/historico_funcionario")
def obter_historico_funcionario(funcionario_id: int = Body(...)):
    """
    Retorna o histórico de todas as sessões de um funcionário.
    """
    try:
        # Acessa o Mongo e busca todas as mensagens do funcionário
        collection = mongo_db["chat_history"]
        historicos = list(collection.find({"funcionario_id": funcionario_id}, {"_id": 0}))

        if not historicos:
            return JSONResponse(
                content={
                    "status": "vazio",
                    "mensagem": f"Nenhum histórico encontrado para o funcionário {funcionario_id}."
                },
                status_code=200
            )

        return JSONResponse(
            content={
                "status": "success",
                "funcionario_id": funcionario_id,
                "total_sessoes": len(historicos),
                "historicos": historicos
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "mensagem": f"Erro ao obter histórico geral: {str(e)}"
            },
            status_code=500
        )