import uvicorn
import webbrowser
import httpx
import chromadb
import json
import os
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import threading
import random
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- КОНФИГУРАЦИЯ ---
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://openai-hub.neuraldeep.tech"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
WHISPER_MODEL = "whisper-1"

logger.info(f"API_KEY установлен: {'✓' if API_KEY else '✗'}")
logger.info(f"BASE_URL: {BASE_URL}")
logger.info(f"LLM_MODEL: {LLM_MODEL}")

# --- FastAPI ---
app = FastAPI(title="Zaman Bank AI Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- КОНФИГУРАЦИЯ СТАТИКИ ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.abspath(os.path.join(current_dir, "static"))
    if not os.path.exists(static_dir):
        static_dir = os.path.abspath(os.path.join(current_dir, "..", "static"))
        if not os.path.exists(static_dir):
            raise FileNotFoundError("Папка static не найдена")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    INDEX_HTML_PATH = os.path.join(static_dir, "index.html")
    CHAT_HTML_PATH = os.path.join(static_dir, "chat.html")
    logger.info(f"✅ Статика: {static_dir}")
except Exception as e:
    logger.error(f"⚠️ Ошибка статики: {e}")
    INDEX_HTML_PATH = None
    CHAT_HTML_PATH = None

USER_STATE_CACHE: Dict[str, Dict[str, Any]] = {}
http_client = httpx.AsyncClient(timeout=120.0)

# --- ChromaDB ---
try:
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../zaman_db")
    db_client = chromadb.PersistentClient(path=db_path)
    collection = db_client.get_collection(name="zaman_products")
    logger.info(f"✅ ChromaDB: {db_path}")
except Exception as e:
    logger.error(f"⚠️ ChromaDB ошибка: {e}")
    collection = None


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def get_data_path(filename: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    if not os.path.exists(data_dir):
        data_dir = current_dir
    return os.path.join(data_dir, filename)


def load_json_safe(filepath: str, default: Any = None) -> Any:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки {filepath}: {e}")
        return default


def load_personalized_client_context() -> str:
    filepath = get_data_path("zaman_personalized_rag_data.json")
    data = load_json_safe(filepath, [])
    if not data:
        return "Информация о клиенте недоступна."

    try:
        client_profile = next(item for item in data if item.get("id") == 0)
        details = client_profile.get("client_details", {})
        summary = client_profile.get("financial_summary_kzt", {})

        context_str = (
            f"Клиент: {details.get('name', 'N/A')}, Возраст: {details.get('age', 'N/A')}, Город: {details.get('city', 'N/A')}. "
            f"Статус: {details.get('status', 'N/A')}. Текущий продукт: {details.get('current_product', 'N/A')}. "
            f"Сред. баланс: {details.get('avg_monthly_balance_kzt', 'N/A')} KZT. "
            f"Ежемесячный доход: {summary.get('monthly_salary_in_kzt', 'N/A')} KZT. "
            f"Платежи по займам: {summary.get('loan_payment_out_avg', 'N/A')} KZT/мес. "
            f"Ключевые возможности продаж: {', '.join(summary.get('key_sales_opportunities', ['N/A']))}."
        )
        return context_str
    except Exception as e:
        logger.error(f"Ошибка парсинга профиля: {e}")
        return "Информация о клиенте недоступна."


def load_benchmark_data() -> str:
    filepath = get_data_path("zaman_benchmark_data.json")
    benchmarks = load_json_safe(filepath, [])
    if not benchmarks:
        return "Сравнительная аналитика недоступна."

    formatted = []
    try:
        for item in benchmarks:
            top_spends = ", ".join([f"{k} ({v:.0f} KZT)" for k, v in item.get("top_spending_categories", {}).items()])
            goals = ", ".join(item.get("common_goals", []))
            formatted.append(
                f"СЕГМЕНТ: {item['segment_name']} | "
                f"Средний доход: {item['avg_monthly_income_kzt']:.0f} KZT. "
                f"Топ-траты: {top_spends}. "
                f"Типичные Цели: {goals}. "
                f"ИНСАЙТ ДЛЯ МОТИВАЦИИ: {item['motivational_insight']}"
            )
        return "\n\n---\n\n".join(formatted)
    except KeyError as e:
        logger.error(f"Ошибка парсинга бенчмарков: {e}")
        return "Сравнительная аналитика недоступна."


def detect_emotional_state(message: str) -> str:
    stress_keywords = ["стресс", "переживаю", "волнуюсь", "тревожно", "устал", "проблем", "расстроен", "не могу"]
    if any(word in message.lower() for word in stress_keywords):
        return "stressed"
    return "neutral"


def get_wellness_advice() -> str:
    tips = [
        "Попробуйте короткую дыхательную практику: 4 секунды вдох, 4 задержка, 6 выдох. Повторите 5 раз.",
        "Сделайте перерыв на 5 минут и послушайте спокойную музыку без слов.",
        "Прогуляйтесь 10 минут на свежем воздухе, если есть возможность.",
        "Запишите 3 вещи, которые у вас хорошо получились сегодня, даже маленькие.",
        "Выпейте стакан чистой воды медленными глотками.",
    ]
    return random.choice(tips)


STATIC_CLIENT_PROFILE = load_personalized_client_context()
BENCHMARK_DATA = load_benchmark_data()

logger.info(f"👤 Профиль: {'✓' if 'недоступна' not in STATIC_CLIENT_PROFILE else '✗'}")
logger.info(f"📈 Бенчмарки: {'✓' if 'недоступна' not in BENCHMARK_DATA else '✗'}")


# ==================== PYDANTIC MODELS ====================

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    role: str
    content: str


class AnalyzeRequest(BaseModel):
    session_id: str


class AnalyzeResponse(BaseModel):
    summary: str
    categories: Dict[str, float]


class SuggestGoalResponse(BaseModel):
    formatted_message: str


# ==================== УНИВЕРСАЛЬНЫЙ ПРОКСИ ====================

async def proxy_request(
        method: str,
        path: str,
        body: Optional[Dict] = None,
        files: Optional[Dict] = None
) -> Dict:
    url = f"{BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    try:
        if method == "GET":
            response = await http_client.get(url, headers=headers)
        elif method == "POST":
            if files:
                processed_files = {}
                for key, value in files.items():
                    if isinstance(value, tuple) and len(value) == 3:
                        processed_files[key] = value
                    elif hasattr(value, 'read'):
                        filename = getattr(value, 'name', 'unknown_file')
                        content_type = getattr(value, 'content_type', 'application/octet-stream')
                        processed_files[key] = (filename, value, content_type)
                    else:
                        raise ValueError(f"Некорректный формат файла для {key}")

                response = await http_client.post(url, headers=headers, files=processed_files, data=body, timeout=120.0)
            else:
                response = await http_client.post(url, headers=headers, json=body)
        elif method == "DELETE":
            response = await http_client.delete(url, headers=headers)
        else:
            raise HTTPException(status_code=405, detail="Method not allowed")

        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Ошибка прокси {method} {path}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Неожиданная ошибка прокси {method} {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ZAMAN BANK ENDPOINTS ====================

async def get_embedding(text: str) -> List[float]:
    if not API_KEY:
        logger.warning("API_KEY не установлен для эмбеддинга")
        return []
    try:
        logger.info(f"Запрос эмбеддинга для текста длиной: {len(text)}")
        response = await http_client.post(
            f"{BASE_URL}/v1/embeddings",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"input": [text], "model": EMBEDDING_MODEL}
        )
        response.raise_for_status()
        embedding = response.json()['data'][0]['embedding']
        logger.info(f"Эмбеддинг получен, размерность: {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Ошибка эмбеддинга: {e}")
        return []


def query_vector_db(embedding: List[float]) -> str:
    if not collection:
        logger.warning("ChromaDB недоступна")
        return "База знаний недоступна."
    if not embedding:
        logger.warning("Пустой эмбеддинг")
        return "База знаний недоступна."

    try:
        logger.info("Запрос к ChromaDB...")
        results = collection.query(query_embeddings=[embedding], n_results=3)
        if results and results.get('documents') and results['documents'][0]:
            docs = "\n---\n".join(results['documents'][0])
            logger.info(f"Найдено документов: {len(results['documents'][0])}")
            return docs
        else:
            logger.warning("В ChromaDB ничего не найдено")
            return "В базе знаний не найдено релевантной информации."
    except Exception as e:
        logger.error(f"Ошибка запроса к ChromaDB: {e}")
        return "Ошибка доступа к базе знаний."


async def get_llm_response(session_id: str, user_message: str) -> str:
    logger.info(f"🔵 LLM запрос от session: {session_id}, message: '{user_message[:50]}...'")

    if not API_KEY:
        logger.error("API_KEY не настроен!")
        return "Ошибка: API ключ не настроен."

    user_context = USER_STATE_CACHE.get(session_id, {}).get(
        "summary",
        "Финансовый анализ не проведен."
    )

    logger.info(f"Контекст пользователя: {user_context[:100]}...")

    emotional_state = detect_emotional_state(user_message)
    wellness_tip = ""
    if emotional_state == "stressed":
        wellness_tip = f"\n\n🌿 **Совет для душевного равновесия:**\n{get_wellness_advice()}"

    query_embedding = await get_embedding(f"{user_message}. Контекст: {user_context}")
    retrieved_docs = query_vector_db(query_embedding)

    logger.info(f"Retrieved docs length: {len(retrieved_docs)}")

    MASTER_PROMPT = f"""
Ты — Zaman, персональный AI-ассистент исламского банка.

Твоя цель — помогать клиентам принимать осознанные финансовые решения в соответствии с принципами Шариата, проактивно предлагая продукты банка и мотивируя на улучшение финансового здоровья. Будь эмпатичным, профессиональным и используй Markdown для форматирования (например, **жирный**, *курсив*, списки).

**КОНТЕКСТ КЛИЕНТА (Айгерим):**
{user_context}

---
**СРАВНИТЕЛЬНАЯ АНАЛИТИКА (БЕНЧМАРКИ ПОХОЖИХ КЛИЕНТОВ):**
{BENCHMARK_DATA}
---

**СПРАВОЧНАЯ ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ БАНКА:**
{retrieved_docs}
---

**ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ КЛИЕНТА:** {emotional_state}

**ТВОЯ ЗАДАЧА:**
1.  **Ответь** на последнее сообщение клиента (`{user_message}`).
2.  **Персонализируй** ответ, используя КОНТЕКСТ КЛИЕНТА.
3.  **Сравни** траты или цели клиента с БЕНЧМАРКАМИ, если это уместно, чтобы **мотивировать** его. Используй ГОТОВЫЙ ВЫВОД ДЛЯ МОТИВАЦИИ из релевантного сегмента.
4.  **Проактивно предложи** подходящий продукт из БАЗЫ ЗНАНИЙ, объясни его **преимущества** и исламский **принцип** (Мудараба, Мурабаха, Вакала).
5.  Если ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ = stressed, интегрируй совет по wellness (он будет добавлен автоматически).
6.  Используй **Markdown** для лучшей читаемости.
7.  Отвечай на **русском языке**.
"""

    try:
        logger.info("Отправка запроса к LLM API...")
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
                "max_tokens": 700
            },
            timeout=60.0
        )
        response.raise_for_status()
        ai_response = response.json()['choices'][0]['message']['content']
        logger.info(f"✅ LLM ответ получен, длина: {len(ai_response)}")
        return ai_response + (wellness_tip if emotional_state == "stressed" else "")
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ HTTP ошибка LLM: {e.response.status_code} - {e.response.text}")
        error_detail = e.response.json().get("error", {}).get("message", e.response.text)
        return f"Произошла ошибка при обращении к AI-модели: {error_detail}"
    except httpx.TimeoutException as e:
        logger.error(f"❌ Timeout LLM: {e}")
        return "Превышено время ожидания ответа от AI. Попробуйте еще раз."
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка LLM: {e}", exc_info=True)
        return "Я столкнулся с внутренней ошибкой. Пожалуйста, перефразируйте ваш запрос."


def analyze_mock_transactions() -> AnalyzeResponse:
    filepath = get_data_path("mock_transactions.json")
    transactions = load_json_safe(filepath, [])
    if not transactions:
        return AnalyzeResponse(summary="Ошибка: файл mock_transactions.json не найден.", categories={})

    categories: Dict[str, float] = {}
    total_income = 0
    total_expense = 0

    for tx in transactions:
        amount = tx.get("amount", 0)
        category = tx.get("category", "Неизвестно")
        if not isinstance(amount, (int, float)): continue

        if amount > 0:
            total_income += amount
        else:
            categories[category] = categories.get(category, 0) + abs(amount)
            total_expense += abs(amount)

    sorted_categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))

    dynamic_summary = (
            f"Клиент получил {total_income:.0f} KZT дохода и потратил {total_expense:.0f} KZT. "
            f"Основные траты за последний период: " + ", ".join(
        [f"{k} ({v:.0f} KZT)" for k, v in list(sorted_categories.items())[:3]])
    )

    full_context = f"СТАТИЧЕСКИЙ ПРОФИЛЬ: {STATIC_CLIENT_PROFILE}\n\nДИНАМИКА: {dynamic_summary}"

    return AnalyzeResponse(summary=full_context, categories=sorted_categories)


# ==================== ФРОНТЕНД РОУТЫ ====================

@app.get("/", include_in_schema=False)
def serve_main_frontend():
    if not INDEX_HTML_PATH or not os.path.exists(INDEX_HTML_PATH):
        raise HTTPException(status_code=404, detail="index.html не найден")
    return FileResponse(INDEX_HTML_PATH, media_type="text/html")


@app.get("/chat", include_in_schema=False)
def serve_chat_frontend():
    if not CHAT_HTML_PATH or not os.path.exists(CHAT_HTML_PATH):
        raise HTTPException(status_code=404, detail="chat.html не найден")
    return FileResponse(CHAT_HTML_PATH, media_type="text/html")


# ==================== API ENDPOINTS ====================

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_transactions_endpoint(request: AnalyzeRequest):
    logger.info(f"📊 Анализ для session: {request.session_id}")
    response = analyze_mock_transactions()
    if request.session_id not in USER_STATE_CACHE:
        USER_STATE_CACHE[request.session_id] = {}
    USER_STATE_CACHE[request.session_id]["summary"] = response.summary
    logger.info(f"✅ Анализ завершен для session: {request.session_id}")
    return response


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    logger.info(f"💬 /api/chat вызван: session={request.session_id}, message='{request.message[:50]}...'")

    if not collection:
        logger.error("❌ ChromaDB не загружена!")
        raise HTTPException(status_code=500, detail="База знаний не загружена")

    try:
        ai_response = await get_llm_response(request.session_id, request.message)
        logger.info(f"✅ /api/chat ответ отправлен: {len(ai_response)} символов")
        return ChatResponse(role="assistant", content=ai_response)
    except Exception as e:
        logger.error(f"❌ Ошибка в /api/chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice")
async def voice_chat_endpoint(session_id: str = Form(...), file: UploadFile = File(...)):
    logger.info(f"🎤 /api/voice вызван: session={session_id}, file={file.filename}")

    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY не настроен")
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Аудиофайл не предоставлен")

    try:
        audio_bytes = await file.read()
        logger.info(f"Аудио прочитан: {len(audio_bytes)} байт")

        files_for_proxy = {"file": (file.filename, audio_bytes, file.content_type)}
        data_for_proxy = {"model": WHISPER_MODEL}

        transcription_result = await proxy_request(
            "POST",
            "/v1/audio/transcriptions",
            body=data_for_proxy,
            files=files_for_proxy
        )

        transcribed_text = transcription_result.get("text", "").strip()
        logger.info(f"Транскрипция: '{transcribed_text}'")

        if not transcribed_text:
            return JSONResponse(
                status_code=400,
                content={"detail": "Не удалось распознать речь. Попробуйте еще раз."}
            )

        ai_response_text = await get_llm_response(session_id, transcribed_text)

        return JSONResponse(content={
            "transcribed_text": transcribed_text,
            "ai_response_text": ai_response_text
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Ошибка /api/voice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера при обработке голоса.")


@app.post("/api/suggest-goal", response_model=SuggestGoalResponse)
async def suggest_financial_goal(request: AnalyzeRequest):
    logger.info(f"🎯 /api/suggest-goal для session: {request.session_id}")

    session_id = request.session_id
    user_context = USER_STATE_CACHE.get(session_id, {}).get(
        "summary",
        "Финансовый анализ не проведен."
    )
    if "не проведен" in user_context:
        raise HTTPException(status_code=400, detail="Сначала нужно загрузить выписку для анализа.")

    goal_prompt = f"""
Проанализируй следующий финансовый контекст клиента:
{user_context}

На основе его трат, дохода и потенциала для продаж, предложи ОДНУ реалистичную финансовую цель (например, накопить на первый взнос, создать резервный фонд, рефинансировать долг). Сформулируй предложение в виде мотивирующего сообщения для клиента, включая название цели и примерный срок ее достижения, если это возможно.

Пример ответа: "Айгерим, учитывая ваши траты на кафе (~35,500 KZT/мес), предлагаю цель: 'Накопить 500,000 KZT на резервный фонд за 10-12 месяцев'. Это даст вам уверенность в завтрашнем дне. Мы можем начать с перевода части сэкономленных средств на наш вклад Мудараба."
"""
    try:
        response = await http_client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": goal_prompt}],
                "temperature": 0.8,
                "max_tokens": 150
            }
        )
        response.raise_for_status()
        goal_suggestion = response.json()['choices'][0]['message']['content']
        logger.info(f"✅ Цель сгенерирована")
        return SuggestGoalResponse(formatted_message=goal_suggestion)

    except Exception as e:
        logger.error(f"❌ Ошибка генерации цели: {e}")
        raise HTTPException(status_code=500, detail="Не удалось сгенерировать финансовую цель.")


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    SERVER_URL = "http://localhost:8000"


    def start_server():
        try:
            uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
        except RuntimeError as e:
            if "Cannot run uvicorn.run()" in str(e):
                logger.error("⚠️ Запустите: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
            else:
                raise e


    logger.info(f"✅ Zaman AI запущен: {SERVER_URL}")
    logger.info(f"📚 Документация: {SERVER_URL}/docs")

    threading.Timer(2.0, lambda: webbrowser.open(SERVER_URL)).start()
    start_server()