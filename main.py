import uvicorn
import webbrowser  # <-- ИМПОРТ ДЛЯ ОТКРЫТИЯ БРАУЗЕРА
import httpx
import chromadb
import json
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates  # <-- НОВЫЙ ИМПОРТ
from fastapi.staticfiles import StaticFiles  # <-- НОВЫЙ ИМПОРТ
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()  # Эта команда читает ваш .env файл

# Теперь код может использовать ключи
API_KEY = os.getenv("API_KEY")
# --- Конфигурация API (из документа) ---
BASE_URL = "https://openai-hub.neuraldeep.tech"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
# ----------------------------------------

# --- Инициализация ---
app = FastAPI(title="Zaman Bank AI Assistant Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- НОВЫЙ КОД: КОНФИГУРАЦИЯ ФРОНТЕНДА ---
# Монтируем статические файлы (CSS, JS, если они появятся)
# Сейчас мы не используем static, но эта строка обязательна, если вы захотите ее добавить
app.mount("/static", StaticFiles(directory="static"), name="static")

# Инициализируем шаблонизатор Jinja2
templates = Jinja2Templates(directory="templates")
# -----------------------------------------


USER_STATE_CACHE: Dict[str, Dict[str, Any]] = {}

http_client = httpx.AsyncClient(timeout=60.0)

try:
    db_client = chromadb.PersistentClient(path="./zaman_db")
    collection = db_client.get_collection(name="zaman_products")
    print("Успешно подключено к ChromaDB.")
except Exception as e:
    print(f"Критическая ошибка: Не удалось загрузить ChromaDB. Запустите rag_prep.py. Ошибка: {e}")
    collection = None


# --- Модели данных (Pydantic) ---
class AnalyzeRequest(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    role: str
    content: str


class AnalyzeResponse(BaseModel):
    summary: str
    categories: Dict[str, float]


# --- Логика Анализа (Хак) ---
def analyze_mock_transactions() -> AnalyzeResponse:
    """
    (Гилфойл): Это наша имитация "Интеллектуального анализа".
    Читает мок-файл и генерирует саммари.
    """
    try:
        with open("mock_transactions.json", "r", encoding="utf-8") as f:
            transactions = json.load(f)
    except FileNotFoundError:
        return AnalyzeResponse(summary="Ошибка: mock_transactions.json не найден.", categories={})

    categories: Dict[str, float] = {}
    total_income = 0
    total_expense = 0

    for tx in transactions:
        amount = tx["amount"]
        category = tx["category"]
        if amount > 0:
            total_income += amount
        else:
            if category not in categories:
                categories[category] = 0
            categories[category] += abs(amount)
            total_expense += abs(amount)

    # Сортируем для наглядности
    sorted_categories = dict(sorted(categories.items(), key=lambda item: item[1], reverse=True))

    # (Ричард): Это саммари пойдет в LLM как "контекст"
    summary = (
            f"Клиент получил {total_income} дохода и потратил {total_expense}. "
            f"Основные траты: " + ", ".join([f"{k} ({v} KZT)" for k, v in sorted_categories.items()])
    )

    return AnalyzeResponse(summary=summary, categories=sorted_categories)


# --- Логика RAG (Ядро) ---
async def get_embedding(text: str) -> list[float]:
    """Получает эмбеддинг для одного текста."""
    try:
        response = await http_client.post(
            f"{BASE_URL}/v1/embeddings",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"input": [text], "model": EMBEDDING_MODEL}
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        print(f"Ошибка API эмбеддинга: {e}")
        return []


def query_vector_db(embedding: list[float]) -> str:
    """Ищет в ChromaDB релевантные документы."""
    if not collection or not embedding:
        return "Информация о продуктах банка временно недоступна."

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3  # 3 самых релевантных документа
    )
    return " ".join(results['documents'][0])


async def get_llm_response(session_id: str, user_message: str) -> str:
    """
    (Ричард): Это мозг. Он собирает все вместе.
    """
    # 1. Получаем финансовый контекст (из /analyze)
    user_context = USER_STATE_CACHE.get(session_id, {}).get("summary", "Финансовый анализ еще не проводился.")

    # 2. RAG: Ищем релевантные продукты/советы
    query_embedding = await get_embedding(f"{user_message} {user_context}")
    retrieved_docs = query_vector_db(query_embedding)

    # 3. (Гилфойл): Собираем "Мастер-промпт". Здесь мы программируем "человечность".
    MASTER_PROMPT = f"""
    Ты - Zaman, "умный, человекоподобный и персональный" AI-ассистент банка.
    Твоя миссия - не просто отвечать, а помогать принимать ОСОЗНАННЫЕ финансовые решения.
    Ты должен вызывать доверие. Будь эмпатичным, но профессиональным.

    ТЕКУЩИЙ ФИНАНСОВЫЙ КОНТЕКСТ КЛИЕНТА:
    {user_context}

    СПРАВОЧНАЯ ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ ZAMAN BANK (Исламское финансирование):
    {retrieved_docs}

    ЗАДАЧА:
    Ответь на последнее сообщение клиента.
    1. Всегда учитывай его финансовый контекст.
    2. Если уместно, ИСПОЛЬЗУЙ информацию из базы знаний, чтобы предложить продукты банка (депозиты, кредиты, инвестиции).
    3. Если клиент говорит о стрессе или тратах, предложи альтернативные способы борьбы со стрессом, не связанные с покупками (например, прогулка, медитация, хобби).
    4. Если клиент ставит цель (квартира, машина, обучение), помоги ему.
    5. Говори на языке клиента (русский).
    """

    try:
        response = await http_client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": MASTER_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    except httpx.HTTPStatusError as e:
        print(f"Ошибка API LLM: {e.response.status_code} - {e.response.text}")
        return "Произошла ошибка при обращении к AI-модели. Попробуйте позже."
    except Exception as e:
        print(f"Неожиданная ошибка LLM: {e}")
        return "Я столкнулся с внутренней ошибкой. Пожалуйста, перефразируйте."


# --- Эндпоинты API ---

# --- НОВЫЙ ЭНДПОИНТ: ГЛАВНАЯ СТРАНИЦА ---
@app.get("/", include_in_schema=False)
def serve_frontend(request: Request):
    """Отдает главную HTML-страницу."""
    return templates.TemplateResponse("index.html", {"request": request})


# -----------------------------------------


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_transactions_endpoint(request: AnalyzeRequest):
    """
    (Фронтенд): Кнопка "Загрузить выписку" вызывает этот эндпоинт.
    Он анализирует МОК-данные и сохраняет саммари в кэш.
    """
    print(f"Запрос на анализ для сессии: {request.session_id}")
    response = analyze_mock_transactions()

    # Сохраняем саммари для RAG
    if request.session_id not in USER_STATE_CACHE:
        USER_STATE_CACHE[request.session_id] = {}
    USER_STATE_CACHE[request.session_id]["summary"] = response.summary

    return response


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    """
    (Фронтенд): Главный эндпоинт чата.
    """
    print(f"Сообщение от {request.session_id}: {request.message}")
    if not collection:
        raise HTTPException(status_code=500, detail="Векторная база данных не загружена. Запустите rag_prep.py")

    ai_response_content = await get_llm_response(request.session_id, request.message)

    return ChatResponse(role="assistant", content=ai_response_content)


# --- Запуск (С авто-открытием браузера) ---
if __name__ == "__main__":
    SERVER_URL = "http://localhost:8000"


    def start_server():
        uvicorn.run(app, host="0.0.0.0", port=8000)


    # 1. Открываем браузер с фронтендом
    print(f"✅ Вся система Zaman AI запущена на: {SERVER_URL}")
    print(f"🌐 Открываем главный интерфейс в браузере...")
    # Используем threading.Timer, чтобы браузер открылся ПОСЛЕ старта сервера
    import threading

    threading.Timer(1.5, lambda: webbrowser.open(SERVER_URL)).start()

    # 2. Запускаем сервер, который блокирует терминал
    start_server()