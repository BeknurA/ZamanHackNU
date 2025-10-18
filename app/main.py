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
from typing import Dict, Any, List
from dotenv import load_dotenv
import threading
from datetime import datetime
from goals_habits import GoalsHabitsManager

load_dotenv()

# Инициализируем менеджер целей
goals_manager = GoalsHabitsManager()

# --- КОНФИГУРАЦИЯ ---
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://openai-hub.neuraldeep.tech"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- FastAPI ---
app = FastAPI(title="Zaman Bank AI Assistant Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
INDEX_HTML_PATH = "./static/index.html"
CHAT_HTML_PATH = "./static/chat.html"

USER_STATE_CACHE: Dict[str, Dict[str, Any]] = {}
http_client = httpx.AsyncClient(timeout=60.0)

# --- ChromaDB ---
try:
    db_client = chromadb.PersistentClient(path="./zaman_db")
    collection = db_client.get_collection(name="zaman_products")
    print("✅ ChromaDB подключена.")
except Exception as e:
    print(f"⚠️ Ошибка ChromaDB: {e}")
    collection = None


# --- ЗАГРУЗКА ДАННЫХ ---
def load_json_safe(filepath: str, default: Any = None) -> Any:
    """Безопасная загрузка JSON с обработкой ошибок"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Файл не найден: {filepath}")
        return default
    except json.JSONDecodeError as e:
        print(f"⚠️ Ошибка JSON в {filepath}: {e}")
        return default


def load_personalized_client_context() -> str:
    """Загружает профиль клиента"""
    data = load_json_safe("data/zaman_personalized_rag_data.json", [])
    if not data:
        return "Информация о клиенте недоступна."
    
    try:
        client_profile = next(item for item in data if item.get("id") == 0)
        details = client_profile.get("client_details", {})
        summary = client_profile.get("financial_summary_kzt", {})
        
        return (
            f"Клиент: {details.get('name')}, {details.get('age')} лет, {details.get('city')}. "
            f"Статус: {details.get('status')}. "
            f"Текущий продукт: {details.get('current_product')}. "
            f"Средний баланс: {details.get('avg_monthly_balance_kzt')} KZT. "
            f"Ежемесячный доход: {summary.get('monthly_salary_in_kzt')} KZT. "
            f"Платежи по займам: {summary.get('loan_payment_out_avg')} KZT/мес."
        )
    except (StopIteration, KeyError, TypeError):
        return "Информация о клиенте недоступна."


def load_benchmark_data() -> str:
    """Загружает бенчмарки"""
    benchmarks = load_json_safe("data/zaman_benchmark_data.json", [])
    if not benchmarks:
        return "Сравнительная аналитика недоступна."
    
    formatted = []
    for item in benchmarks:
        top_spends = ", ".join([f"{k} ({v:.0f} KZT)" for k, v in item.get("top_spending_categories", {}).items()])
        goals = ", ".join(item.get("common_goals", []))
        
        formatted.append(
            f"СЕГМЕНТ: {item['segment_name']} | "
            f"Средний доход: {item['avg_monthly_income_kzt']:.0f} KZT. "
            f"Топ-траты: {top_spends}. "
            f"Цели: {goals}. "
            f"ИНСАЙТ: {item['motivational_insight']}"
        )
    
    return "\n\n---\n\n".join(formatted)


# --- Эмоциональный интеллект бота ---
def detect_emotional_state(message: str) -> str:
    """Определяет эмоциональное состояние из сообщения"""
    stress_keywords = ["стресс", "переживаю", "волнуюсь", "тревожно", "нервничаю", "устал", "проблем"]
    positive_keywords = ["спасибо", "отлично", "замечательно", "рад", "благодарен"]
    
    msg_lower = message.lower()
    
    if any(word in msg_lower for word in stress_keywords):
        return "stressed"
    if any(word in msg_lower for word in positive_keywords):
        return "positive"
    
    return "neutral"


def get_wellness_advice() -> str:
    """Советы по борьбе со стрессом без трат"""
    tips = [
        "💚 Попробуйте бесплатную медитацию: приложение 'Insight Timer' предлагает тысячи бесплатных сессий на русском языке.",
        "🚶 Прогулка в парке — доказанный способ снизить кортизол (гормон стресса) на 25%.",
        "📝 Ведение финансового дневника помогает снизить тревожность на 30% (исследование Cambridge University).",
        "☕ Встретьтесь с другом за чашкой чая дома — социальная поддержка важнее, чем дорогие развлечения.",
        "🧘 Практика благодарности: каждый вечер записывайте 3 вещи, за которые вы благодарны сегодня.",
    ]
    import random
    return random.choice(tips)


STATIC_CLIENT_PROFILE = load_personalized_client_context()
BENCHMARK_DATA = load_benchmark_data()

print("👤 Профиль клиента загружен." if "недоступна" not in STATIC_CLIENT_PROFILE else "⚠️ Профиль клиента НЕ загружен.")
print("📈 Бенчмарки загружены." if "недоступна" not in BENCHMARK_DATA else "⚠️ Бенчмарки НЕ загружены.")


# --- Pydantic Models ---
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


# --- Анализ транзакций ---
def analyze_mock_transactions() -> AnalyzeResponse:
    transactions = load_json_safe("data/mock_transactions.json", [])
    if not transactions:
        return AnalyzeResponse(
            summary="Ошибка: файл транзакций не найден.",
            categories={}
        )
    
    categories: Dict[str, float] = {}
    total_income = 0
    total_expense = 0
    
    for tx in transactions:
        amount = tx["amount"]
        category = tx["category"]
        if amount > 0:
            total_income += amount
        else:
            categories[category] = categories.get(category, 0) + abs(amount)
            total_expense += abs(amount)
    
    sorted_categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
    
    dynamic_summary = (
        f"Клиент получил {total_income:.0f} KZT дохода и потратил {total_expense:.0f} KZT. "
        f"Основные траты: " + ", ".join([f"{k} ({v:.0f} KZT)" for k, v in list(sorted_categories.items())[:3]])
    )
    
    full_context = (
        f"СТАТИЧЕСКИЙ ПРОФИЛЬ: {STATIC_CLIENT_PROFILE}\n\n"
        f"ДИНАМИЧЕСКИЙ АНАЛИЗ: {dynamic_summary}"
    )
    
    return AnalyzeResponse(summary=full_context, categories=sorted_categories)


# --- RAG ---
async def get_embedding(text: str) -> List[float]:
    try:
        response = await http_client.post(
            f"{BASE_URL}/v1/embeddings",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"input": [text], "model": EMBEDDING_MODEL}
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        print(f"Ошибка эмбеддинга: {e}")
        return []


def query_vector_db(embedding: List[float]) -> str:
    if not collection or not embedding:
        return "База знаний недоступна."
    
    results = collection.query(query_embeddings=[embedding], n_results=3)
    return "\n---\n".join(results['documents'][0])


# --- LLM Response ---
async def get_llm_response(session_id: str, user_message: str) -> str:
    user_context = USER_STATE_CACHE.get(session_id, {}).get(
        "summary", 
        "Финансовый анализ не проведен. Попросите загрузить выписку."
    )
    
    # Определяем эмоциональное состояние
    emotional_state = detect_emotional_state(user_message)
    
    # Если клиент в стрессе, добавляем совет по wellness
    wellness_tip = ""
    if emotional_state == "stressed":
        wellness_tip = f"\n\n**🌿 Совет по заботе о себе:**\n{get_wellness_advice()}"
    
    # RAG
    query_embedding = await get_embedding(f"{user_message}. Контекст: {user_context}")
    retrieved_docs = query_vector_db(query_embedding)
    
    # Текущее время
    current_time = datetime.now().strftime("%H:%M, %d.%m.%Y")
    
    MASTER_PROMPT = f"""
Ты — Zaman, персональный AI-ассистент исламского банка. Твоя миссия — быть настоящим финансовым другом клиента.

**ПРИНЦИПЫ ОБЩЕНИЯ:**
1. **Эмпатия прежде всего** — если клиент переживает, сначала поддержи его эмоционально, потом говори о финансах.
2. **Никогда не используй Bold или Italic** — общайся естественно, как живой человек.
3. **Будь проактивным** — не жди вопросов, предлагай решения на основе данных.
4. **Говори на языке клиента** — избегай банковского жаргона, объясняй простыми словами.

**ТЕКУЩИЙ КОНТЕКСТ КЛИЕНТА:**
{user_context}

**СРАВНИТЕЛЬНАЯ АНАЛИТИКА (для мотивации):**
{BENCHMARK_DATA}

**БАЗА ЗНАНИЙ О ПРОДУКТАХ:**
{retrieved_docs}

**ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ КЛИЕНТА:** {emotional_state}

**ТЕКУЩЕЕ ВРЕМЯ:** {current_time}

**ЗАДАЧА:**
Ответь на сообщение клиента, учитывая:
- Его финансовую ситуацию
- Эмоциональное состояние
- Сравнительную аналитику (мотивируй, сравнивая с похожими клиентами)
- Продукты банка (предлагай, где уместно)
- Принципы исламского финансирования (объясняй просто)

Если клиент в стрессе, обязательно предложи способ справиться без лишних трат.
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
                "temperature": 0.8,  # Повышена для более "человечных" ответов
                "max_tokens": 600
            }
        )
        response.raise_for_status()
        ai_response = response.json()['choices'][0]['message']['content']
        
        # Добавляем wellness совет, если нужно
        return ai_response + wellness_tip
        
    except Exception as e:
        print(f"Ошибка LLM: {e}")
        return "Извините, возникла техническая проблема. Попробуйте еще раз через минуту."


# --- ENDPOINTS ---

@app.get("/", include_in_schema=False)
def serve_main_frontend():
    if not os.path.exists(INDEX_HTML_PATH):
        raise HTTPException(status_code=404, detail="index.html не найден")
    return FileResponse(INDEX_HTML_PATH, media_type="text/html")


@app.get("/chat", include_in_schema=False)
def serve_chat_frontend():
    if not os.path.exists(CHAT_HTML_PATH):
        raise HTTPException(status_code=404, detail="chat.html не найден")
    return FileResponse(CHAT_HTML_PATH, media_type="text/html")


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_transactions_endpoint(request: AnalyzeRequest):
    print(f"Анализ для {request.session_id}")
    response = analyze_mock_transactions()
    
    if request.session_id not in USER_STATE_CACHE:
        USER_STATE_CACHE[request.session_id] = {}
    USER_STATE_CACHE[request.session_id]["summary"] = response.summary
    
    return response


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    print(f"[{request.session_id}] {request.message}")
    
    if not collection:
        raise HTTPException(status_code=500, detail="База знаний не загружена")
    
    ai_response = await get_llm_response(request.session_id, request.message)
    
    return ChatResponse(role="assistant", content=ai_response)


# --- ЗАПУСК ---
if __name__ == "__main__":
    SERVER_URL = "http://localhost:8000"
    
    def start_server():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    print(f"✅ Zaman AI запущен: {SERVER_URL}")
    print("🌐 Открываем браузер...")
    
    threading.Timer(1.5, lambda: webbrowser.open(SERVER_URL)).start()
    start_server()