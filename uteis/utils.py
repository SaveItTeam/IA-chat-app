from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from zoneinfo import ZoneInfo
from uteis.db_connection import redis_db, mongo_db

TZ = ZoneInfo("America/Sao_Paulo")

def get_session_history(empresa_id, funcionario_id, session_id, *args, **kwargs):
    redis_key = f"chat:{empresa_id}:{funcionario_id}:{session_id}"
    mensagens = redis_db.lrange(redis_key, 0, -1)
    history = ChatMessageHistory()
    for msg in mensagens:
        try:
            role, content = msg.split("||", 1)
            if role == "user":
                history.add_message(HumanMessage(content=content))
            else:
                history.add_message(AIMessage(content=content))
        except Exception:
            continue
    return history


def salvar_mensagem(empresa_id, funcionario_id, session_id, role, content):
    redis_key = f"chat:{empresa_id}:{funcionario_id}:{session_id}"
    redis_db.rpush(redis_key, f"{role}||{content}")
    redis_db.ltrim(redis_key, -50, -1)

    mongo_db.chat_history.insert_one({
        "empresa_id": empresa_id,
        "funcionario_id": funcionario_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(TZ)
    })
def build_config(empresa_id, funcionario_id, session_id):
    return {
        "configurable": {
            "session_id": f"{empresa_id}:{funcionario_id}:{session_id}"
        }
    }
