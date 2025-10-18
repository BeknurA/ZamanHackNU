import uvicorn
import webbrowser
import httpx
import chromadb
import json
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
import threading

load_dotenv()

# --- КОНФИГУРАЦИЯ БЭКЕНДА И API ---
# Убедитесь, что у вас есть файл .env с ключом API_KEY
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://openai-hub.neuraldeep.tech"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
# -----------------------------------

# --- Инициализация FastAPI ---
app = FastAPI(title="Zaman Bank AI Assistant Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- КОНФИГУРАЦИЯ СТАТИКИ И ШАБЛОНОВ ---
# 1. Монтируем StaticFiles на /static/
# Все файлы в папке static будут доступны по пути /static/...
app.mount("/static", StaticFiles(directory="static"), name="static")

# ПУТИ К HTML-файлам (Они находятся в static)
INDEX_HTML_PATH = "./static/index.html"
CHAT_HTML_PATH = "./static/chat.html"
# ---------------------------------------

USER_STATE_CACHE: Dict[str, Dict[str, Any]] = {}
http_client = httpx.AsyncClient(timeout=60.0)

# --- Инициализация ChromaDB (RAG) ---
try:
    db_client = chromadb.PersistentClient(path="./zaman_db")
    collection = db_client.get_collection(name="zaman_products")
    print("✅ Успешно подключено к ChromaDB.")
except Exception as e:
    print(f"⚠️ Критическая ошибка: Не удалось загрузить ChromaDB. Запустите rag_prep.py. Ошибка: {e}")
    collection = None


# --- ЗАГРУЗКА ПЕРСОНАЛИЗИРОВАННОГО КОНТЕКСТА КЛИЕНТА ---
def load_personalized_client_context() -> str:
    """Загружает статический профиль клиента (Айгерим) из RAG JSON-файла."""
    try:
        # Убедитесь, что файл лежит в корне проекта
        with open("zaman_personalized_rag_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            client_profile = next(item for item in data if item.get("id") == 0)

            details = client_profile.get("client_details", {})
            summary = client_profile.get("financial_summary_kzt", {})

            context_str = (
                f"Клиент: {details.get('name', 'N/A')}, Возраст: {details.get('age', 'N/A')}, Город: {details.get('city', 'N/A')}. "
                f"Текущий неисламский продукт: {details.get('current_product', 'N/A')}. "
                f"Ежемесячный доход: {summary.get('monthly_salary_in_kzt', 'N/A')} KZT. "
                f"Ежемесячные платежи по займам: {summary.get('loan_payment_out_avg', 'N/A')} KZT. "
                f"Главный потенциал для продаж (Upsell): {', '.join(summary.get('key_sales_opportunities', []))}."
            )
            return context_str
    except (FileNotFoundError, IndexError):
        return "Статическая информация о клиенте недоступна."
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке персонализированного контекста: {e}")
        return "Статическая информация о клиенте недоступна."


STATIC_CLIENT_PROFILE = load_personalized_client_context()
print("👤 Статический профиль клиента загружен для персонализации.")


# --- ЗАГРУЗКА СРАВНИТЕЛЬНОЙ АНАЛИТИКИ (БЕНЧМАРКИ) ---
def load_benchmark_data() -> str:
    """Загружает данные для сравнительной аналитики (бенчмарков) из JSON-файла."""
    try:
        # Убедитесь, что файл лежит в корне проекта
        with open("zaman_benchmark_data.json", "r", encoding="utf-8") as f:
            benchmarks = json.load(f)
            formatted_benchmarks = []
            for item in benchmarks:
                top_spends = ", ".join([f"{k} ({v:.0f} KZT)" for k, v in item["top_spending_categories"].items()])
                goals = ", ".join(item["common_goals"])

                formatted_benchmarks.append(
                    f"СЕГМЕНТ: {item['segment_name']} | Средний Доход: {item['avg_monthly_income_kzt']:.0f} KZT. "
                    f"Топ-траты: {top_spends}. "
                    f"Типичные Цели: {goals}. "
                    f"ГОТОВЫЙ ВЫВОД ДЛЯ МОТИВАЦИИ: {item['motivational_insight']}"
                )

            return "\n\n---\n\n".join(formatted_benchmarks)

    except FileNotFoundError:
        print("⚠️ ПРЕДУПРЕЖДЕНИЕ: zaman_benchmark_data.json не найден. Сравнительная аналитика будет недоступна.")
        return "Сравнительная аналитика недоступна."
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке бенчмарков: {e}")
        return "Сравнительная аналитика недоступна."


BENCHMARK_DATA = load_benchmark_data()
print("📈 Сравнительная аналитика (бенчмарки) загружена.")


# --------------------------------------------------------

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


# --- Логика Анализа (Мок) ---
def analyze_mock_transactions() -> AnalyzeResponse:
    """Имитация интеллектуального анализа."""
    try:
        # Убедитесь, что файл лежит в корне проекта
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

    sorted_categories = dict(sorted(categories.items(), key=lambda item: item[1], reverse=True))

    # Формируем ДИНАМИЧЕСКИЙ саммари
    dynamic_summary = (
            f"Клиент получил {total_income} KZT дохода и потратил {total_expense} KZT. "
            f"Основные траты за последний период: " + ", ".join(
        [f"{k} ({v:.0f} KZT)" for k, v in sorted_categories.items()])
    )

    # ОБЪЕДИНЯЕМ СТАТИЧЕСКИЙ И ДИНАМИЧЕСКИЙ КОНТЕКСТ
    full_context = (
        f"СТАТИЧЕСКИЙ ПРОФИЛЬ (Из базы знаний продаж): {STATIC_CLIENT_PROFILE}\n\n"
        f"ДИНАМИЧЕСКИЙ АНАЛИЗ ТЕКУЩИХ ТРАНЗАКЦИЙ: {dynamic_summary}"
    )

    return AnalyzeResponse(summary=full_context, categories=sorted_categories)


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
        n_results=3
    )
    return "\n---\n".join(results['documents'][0])


async def get_llm_response(session_id: str, user_message: str) -> str:
    """
    Главная функция LLM. Собирает контекст, ищет RAG и формирует промпт.
    """
    # 1. Получаем ПОЛНЫЙ финансовый контекст (статика + динамика)
    user_context = USER_STATE_CACHE.get(session_id, {}).get("summary",
                                                            "Финансовый анализ еще не проводился. Попросите клиента нажать 'Загрузить выписку'.")

    # 2. RAG: Ищем релевантные продукты/советы
    query_embedding = await get_embedding(f"Сообщение клиента: {user_message}. Его контекст: {user_context}")
    retrieved_docs = query_vector_db(query_embedding)

    # 3. Собираем "Мастер-промпт"
    MASTER_PROMPT = f"""
    Ты - Zaman, "умный, человекоподобный и персональный" AI-ассистент исламского банка.
    Твоя миссия - не просто отвечать, а помогать принимать ОСОЗНАННЫЕ финансовые решения, строго соблюдая принципы этики и Шариата.
    Ты должен вызывать доверие. Будь эмпатичным, но профессиональным. Ты ПРОАКТИВНЫЙ продавец услуг.

    ТЕКУЩИЙ, ПОЛНЫЙ КОНТЕКСТ КЛИЕНТА (Включая статический профиль и анализ трат):
    {user_context}

    --- СРАВНИТЕЛЬНАЯ АНАЛИТИКА (БЕНЧМАРКИ) ---
    Используй эти анонимные данные других клиентов для мотивации и сравнения. Выбери наиболее релевантный сегмент, чтобы дать индивидуальный совет по оптимизации или целям.
    {BENCHMARK_DATA}
    --- КОНЕЦ БЕНЧМАРКОВ ---

    СПРАВОЧНАЯ ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ ZAMAN BANK (Продукты и Исламское финансирование):
    {retrieved_docs}

    ЗАДАЧА:
    Ответь на последнее сообщение клиента.
    1. Всегда учитывай его финансовый контекст для персонализации.
    2. Если уместно, ИСПОЛЬЗУЙ информацию из базы знаний, чтобы ПРОАКТИВНО предложить продукты Zaman Bank.
    3. Если клиент спрашивает о своих тратах или целях, обязательно сравни его с подходящим сегментом из СРАВНИТЕЛЬНОЙ АНАЛИТИКИ (бенчмарков), чтобы мотивировать его и дать реалистичный совет по сбережениям.
    4. Объясняй принципы Исламского финансирования простыми словами.
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


# --- Эндпоинты Фронтенда ---

@app.get("/", include_in_schema=False)
def serve_main_frontend():
    """Отдает главную HTML-страницу (static/index.html)."""
    if not os.path.exists(INDEX_HTML_PATH):
        raise HTTPException(status_code=404, detail=f"Главный фронтенд файл не найден: {INDEX_HTML_PATH}")
    return FileResponse(INDEX_HTML_PATH, media_type="text/html")


@app.get("/chat", include_in_schema=False)
def serve_chat_frontend():
    """Отдает страницу чата (static/chat.html)."""
    if not os.path.exists(CHAT_HTML_PATH):
        raise HTTPException(status_code=404, detail=f"Фронтенд файл чата не найден: {CHAT_HTML_PATH}")
    return FileResponse(CHAT_HTML_PATH, media_type="text/html")


# --- Эндпоинты API (Остаются без изменений) ---

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_transactions_endpoint(request: AnalyzeRequest):
    """
    Анализирует МОК-данные и сохраняет ПОЛНЫЙ контекст (статика+динамика) в кэш.
    """
    print(f"Запрос на анализ для сессии: {request.session_id}")
    response = analyze_mock_transactions()

    # Сохраняем ПОЛНЫЙ саммари для RAG
    if request.session_id not in USER_STATE_CACHE:
        USER_STATE_CACHE[request.session_id] = {}
    USER_STATE_CACHE[request.session_id]["summary"] = response.summary

    return response


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    """
    Главный эндпоинт чата.
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


    print(f"✅ Вся система Zaman AI запущена на: {SERVER_URL}")
    print(f"🌐 Открываем главный интерфейс в браузере...")

    # Открываем браузер на корневом пути, который теперь отдает index.html
    threading.Timer(1.5, lambda: webbrowser.open(SERVER_URL)).start()

    start_server()