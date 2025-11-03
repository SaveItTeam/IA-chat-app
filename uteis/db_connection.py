from dotenv import load_dotenv
import os
import redis
import psycopg
from pymongo import MongoClient

load_dotenv()

def conectar_redis():
    redis_url = os.getenv("REDIS_URL")
    return redis.from_url(redis_url, decode_responses=True)

def conectar_mongo():
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    return client["saveit_db"]

def conectar_banco():
    try:
        return psycopg.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
    except psycopg.Error as e:
        raise ConnectionError(f"Erro ao conectar ao banco: {e.pgerror}")

redis_db = conectar_redis()
mongo_db = conectar_mongo()
