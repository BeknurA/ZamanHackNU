"""
Полный API роутер для Zaman Bank
Поддерживает все эндпоинты OpenAI-совместимого API
"""

import uvicorn
import webbrowser
import httpx
import chromadb
import json
import os
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import threading
from datetime import datetime
import random

load_dotenv()

# --- КОНФИГУРАЦИЯ ---
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://openai-hub.neuraldeep.tech"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- FastAPI ---
app = FastAPI(
    title="Zaman Bank AI Assistant API",
    description="Полный API роутер с поддержкой всех OpenAI эндпоинтов",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="../static"), name="static")
INDEX_HTML_PATH = "../static/index.html"
CHAT_HTML_PATH = "../static/chat.html"

USER_STATE_CACHE: Dict[str, Dict[str, Any]] = {}
http_client = httpx.AsyncClient(timeout=120.0)

# --- ChromaDB ---
try:
    db_client = chromadb.PersistentClient(path="./zaman_db")
    collection = db_client.get_collection(name="zaman_products")
    print("✅ ChromaDB подключена.")
except Exception as e:
    print(f"⚠️ Ошибка ChromaDB: {e}")
    collection = None


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def load_json_safe(filepath: str, default: Any = None) -> Any:
    """Безопасная загрузка JSON"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️ Ошибка загрузки {filepath}: {e}")
        return default


def load_personalized_client_context() -> str:
    """Загружает профиль клиента"""
    data = load_json_safe("../data/zaman_personalized_rag_data.json", [])
    if not data:
        return "Информация о клиенте недоступна."
    
    try:
        client_profile = next(item for item in data if item.get("id") == 0)
        details = client_profile.get("client_details", {})
        summary = client_profile.get("financial_summary_kzt", {})
        
        return (
            f"Клиент: {details.get('name')}, {details.get('age')} лет, {details.get('city')}. "
            f"Статус: {details.get('status')}. "
            f"Ежемесячный доход: {summary.get('monthly_salary_in_kzt')} KZT. "
            f"Платежи по займам: {summary.get('loan_payment_out_avg')} KZT/мес."
        )
    except (StopIteration, KeyError, TypeError):
        return "Информация о клиенте недоступна."


def load_benchmark_data() -> str:
    """Загружает бенчмарки"""
    benchmarks = load_json_safe("../data/zaman_benchmark_data.json", [])
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
            f"ИНСАЙТ: {item['motivational_insight']}"
        )
    return "\n\n---\n\n".join(formatted)


def detect_emotional_state(message: str) -> str:
    """Определяет эмоциональное состояние"""
    stress_keywords = ["стресс", "переживаю", "волнуюсь", "тревожно", "устал", "проблем"]
    if any(word in message.lower() for word in stress_keywords):
        return "stressed"
    return "neutral"


def get_wellness_advice() -> str:
    """Советы по wellness"""
    tips = [
        "💚 Попробуйте бесплатную медитацию: приложение 'Insight Timer'.",
        "🚶 Прогулка в парке снижает стресс на 25%.",
        "📝 Ведение финансового дневника снижает тревожность на 30%.",
        "☕ Встретьтесь с другом за чашкой чая дома.",
        "🧘 Практика благодарности: записывайте 3 вещи, за которые благодарны.",
    ]
    return random.choice(tips)


STATIC_CLIENT_PROFILE = load_personalized_client_context()
BENCHMARK_DATA = load_benchmark_data()

print("👤 Профиль клиента загружен." if "недоступна" not in STATIC_CLIENT_PROFILE else "⚠️ Профиль клиента НЕ загружен.")
print("📈 Бенчмарки загружены." if "недоступна" not in BENCHMARK_DATA else "⚠️ Бенчмарки НЕ загружены.")


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


# ==================== УНИВЕРСАЛЬНЫЙ ПРОКСИ ====================

async def proxy_request(
    method: str,
    path: str,
    body: Optional[Dict] = None,
    files: Optional[Dict] = None
) -> Dict:
    """Универсальная функция для проксирования запросов к API"""
    url = f"{BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        if method == "GET":
            response = await http_client.get(url, headers=headers)
        elif method == "POST":
            if files:
                response = await http_client.post(url, headers=headers, files=files, data=body)
            else:
                response = await http_client.post(url, headers=headers, json=body)
        elif method == "DELETE":
            response = await http_client.delete(url, headers=headers)
        else:
            raise HTTPException(status_code=405, detail="Method not allowed")
        
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CHAT COMPLETIONS ====================

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
@app.post("/engines/{model}/chat/completions")
@app.post("/openai/deployments/{model}/chat/completions")
async def chat_completions(request: Request, model: Optional[str] = None):
    """Chat Completion API"""
    body = await request.json()
    if model:
        body["model"] = model
    return await proxy_request("POST", "/v1/chat/completions", body)


# ==================== COMPLETIONS ====================

@app.post("/v1/completions")
@app.post("/completions")
@app.post("/engines/{model}/completions")
@app.post("/openai/deployments/{model}/completions")
async def completions(request: Request, model: Optional[str] = None):
    """Text Completion API"""
    body = await request.json()
    if model:
        body["model"] = model
    return await proxy_request("POST", "/v1/completions", body)


# ==================== EMBEDDINGS ====================

@app.post("/v1/embeddings")
@app.post("/embeddings")
@app.post("/engines/{model}/embeddings")
@app.post("/openai/deployments/{model}/embeddings")
async def embeddings(request: Request, model: Optional[str] = None):
    """Embeddings API"""
    body = await request.json()
    if model:
        body["model"] = model
    return await proxy_request("POST", "/v1/embeddings", body)


# ==================== IMAGES ====================

@app.post("/v1/images/generations")
@app.post("/images/generations")
async def image_generations(request: Request):
    """Image Generation API"""
    body = await request.json()
    return await proxy_request("POST", "/v1/images/generations", body)


@app.post("/v1/images/edits")
@app.post("/images/edits")
async def image_edits(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    mask: Optional[UploadFile] = File(None)
):
    """Image Edit API"""
    files = {"image": image.file}
    if mask:
        files["mask"] = mask.file
    data = {"prompt": prompt}
    return await proxy_request("POST", "/v1/images/edits", data, files)


# ==================== AUDIO ====================

@app.post("/v1/audio/transcriptions")
@app.post("/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1")
):
    """Audio Transcriptions API"""
    files = {"file": file.file}
    data = {"model": model}
    return await proxy_request("POST", "/v1/audio/transcriptions", data, files)


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def audio_speech(request: Request):
    """Audio Speech (TTS) API"""
    body = await request.json()
    return await proxy_request("POST", "/v1/audio/speech", body)


# ==================== MODERATIONS ====================

@app.post("/v1/moderations")
@app.post("/moderations")
async def moderations(request: Request):
    """Moderations API"""
    body = await request.json()
    return await proxy_request("POST", "/v1/moderations", body)


# ==================== FILES ====================

@app.post("/v1/files")
@app.post("/files")
async def create_file(file: UploadFile = File(...), purpose: str = Form(...)):
    """Create File"""
    files = {"file": file.file}
    data = {"purpose": purpose}
    return await proxy_request("POST", "/v1/files", data, files)


@app.get("/v1/files")
@app.get("/files")
async def list_files():
    """List Files"""
    return await proxy_request("GET", "/v1/files")


@app.get("/v1/files/{file_id}")
@app.get("/files/{file_id}")
async def get_file(file_id: str):
    """Get File"""
    return await proxy_request("GET", f"/v1/files/{file_id}")


@app.delete("/v1/files/{file_id}")
@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete File"""
    return await proxy_request("DELETE", f"/v1/files/{file_id}")


@app.get("/v1/files/{file_id}/content")
@app.get("/files/{file_id}/content")
async def get_file_content(file_id: str):
    """Get File Content"""
    return await proxy_request("GET", f"/v1/files/{file_id}/content")


# ==================== BATCHES ====================

@app.post("/v1/batches")
@app.post("/batches")
async def create_batch(request: Request):
    """Create Batch"""
    body = await request.json()
    return await proxy_request("POST", "/v1/batches", body)


@app.get("/v1/batches")
@app.get("/batches")
async def list_batches():
    """List Batches"""
    return await proxy_request("GET", "/v1/batches")


@app.get("/v1/batches/{batch_id}")
@app.get("/batches/{batch_id}")
async def retrieve_batch(batch_id: str):
    """Retrieve Batch"""
    return await proxy_request("GET", f"/v1/batches/{batch_id}")


# ==================== FINE-TUNING ====================

@app.post("/v1/fine_tuning/jobs")
@app.post("/fine_tuning/jobs")
async def create_fine_tuning_job(request: Request):
    """Create Fine-Tuning Job (Enterprise)"""
    body = await request.json()
    return await proxy_request("POST", "/v1/fine_tuning/jobs", body)


@app.get("/v1/fine_tuning/jobs")
@app.get("/fine_tuning/jobs")
async def list_fine_tuning_jobs():
    """List Fine-Tuning Jobs (Enterprise)"""
    return await proxy_request("GET", "/v1/fine_tuning/jobs")


@app.post("/v1/fine_tuning/jobs/{job_id}/cancel")
@app.post("/fine_tuning/jobs/{job_id}/cancel")
async def cancel_fine_tuning_job(job_id: str):
    """Cancel Fine-Tuning Job (Enterprise)"""
    return await proxy_request("POST", f"/v1/fine_tuning/jobs/{job_id}/cancel")


# ==================== ASSISTANTS ====================

@app.get("/v1/assistants")
@app.get("/assistants")
async def get_assistants():
    """Get Assistants"""
    return await proxy_request("GET", "/v1/assistants")


@app.post("/v1/assistants")
@app.post("/assistants")
async def create_assistant(request: Request):
    """Create Assistant"""
    body = await request.json()
    return await proxy_request("POST", "/v1/assistants", body)


@app.delete("/v1/assistants/{assistant_id}")
@app.delete("/assistants/{assistant_id}")
async def delete_assistant(assistant_id: str):
    """Delete Assistant"""
    return await proxy_request("DELETE", f"/v1/assistants/{assistant_id}")


# ==================== THREADS ====================

@app.post("/v1/threads")
@app.post("/threads")
async def create_thread(request: Request):
    """Create Thread"""
    body = await request.json()
    return await proxy_request("POST", "/v1/threads", body)


@app.get("/v1/threads/{thread_id}")
@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get Thread"""
    return await proxy_request("GET", f"/v1/threads/{thread_id}")


@app.post("/v1/threads/{thread_id}/messages")
@app.post("/threads/{thread_id}/messages")
async def add_messages(thread_id: str, request: Request):
    """Add Messages to Thread"""
    body = await request.json()
    return await proxy_request("POST", f"/v1/threads/{thread_id}/messages", body)


@app.get("/v1/threads/{thread_id}/messages")
@app.get("/threads/{thread_id}/messages")
async def get_messages(thread_id: str):
    """Get Messages from Thread"""
    return await proxy_request("GET", f"/v1/threads/{thread_id}/messages")


@app.post("/v1/threads/{thread_id}/runs")
@app.post("/threads/{thread_id}/runs")
async def run_thread(thread_id: str, request: Request):
    """Run Thread"""
    body = await request.json()
    return await proxy_request("POST", f"/v1/threads/{thread_id}/runs", body)


# ==================== RERANK ====================

@app.post("/v1/rerank")
@app.post("/v2/rerank")
@app.post("/rerank")
async def rerank(request: Request):
    """Rerank API"""
    body = await request.json()
    return await proxy_request("POST", "/v1/rerank", body)


# ==================== VECTOR STORES ====================

@app.post("/v1/vector_stores")
@app.post("/vector_stores")
async def create_vector_store(request: Request):
    """Create Vector Store"""
    body = await request.json()
    return await proxy_request("POST", "/v1/vector_stores", body)


@app.post("/v1/vector_stores/{vector_store_id}/search")
@app.post("/vector_stores/{vector_store_id}/search")
async def search_vector_store(vector_store_id: str, request: Request):
    """Search Vector Store"""
    body = await request.json()
    return await proxy_request("POST", f"/v1/vector_stores/{vector_store_id}/search", body)


# ==================== UTILS ====================

@app.post("/utils/token_counter")
async def token_counter(request: Request):
    """Token Counter Utility"""
    body = await request.json()
    return await proxy_request("POST", "/utils/token_counter", body)


@app.post("/utils/transform_request")
async def transform_request(request: Request):
    """Transform Request Utility"""
    body = await request.json()
    return await proxy_request("POST", "/utils/transform_request", body)


# ==================== WEBSOCKET ====================

@app.websocket("/v1/realtime")
@app.websocket("/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для realtime коммуникации"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        print(f"WebSocket error: {e}")


# ==================== ZAMAN BANK CUSTOM ENDPOINTS ====================

async def get_embedding(text: str) -> List[float]:
    """Получает эмбеддинг"""
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
    """Ищет в ChromaDB"""
    if not collection or not embedding:
        return "База знаний недоступна."
    results = collection.query(query_embeddings=[embedding], n_results=3)
    return "\n---\n".join(results['documents'][0])


async def get_llm_response(session_id: str, user_message: str) -> str:
    """Главная функция LLM с RAG"""
    user_context = USER_STATE_CACHE.get(session_id, {}).get(
        "summary", 
        "Финансовый анализ не проведен."
    )
    
    emotional_state = detect_emotional_state(user_message)
    wellness_tip = ""
    if emotional_state == "stressed":
        wellness_tip = f"\n\n**🌿 Совет:**\n{get_wellness_advice()}"
    
    query_embedding = await get_embedding(f"{user_message}. Контекст: {user_context}")
    retrieved_docs = query_vector_db(query_embedding)
    
    MASTER_PROMPT = f"""
Ты — Zaman, персональный AI-ассистент исламского банка.

не используй Bold или Italic в ответах.

КОНТЕКСТ КЛИЕНТА:
{user_context}

СРАВНИТЕЛЬНАЯ АНАЛИТИКА:
{BENCHMARK_DATA}

БАЗА ЗНАНИЙ:
{retrieved_docs}

ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ: {emotional_state}

Ответь на сообщение клиента, учитывая его ситуацию и продукты банка.
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
                "temperature": 0.8,
                "max_tokens": 600
            }
        )
        response.raise_for_status()
        ai_response = response.json()['choices'][0]['message']['content']
        return ai_response + wellness_tip
    except Exception as e:
        print(f"Ошибка LLM: {e}")
        return "Произошла ошибка. Попробуйте позже."


def analyze_mock_transactions() -> AnalyzeResponse:
    """Анализ транзакций"""
    transactions = load_json_safe("../data/mock_transactions.json", [])
    if not transactions:
        return AnalyzeResponse(summary="Ошибка: файл транзакций не найден.", categories={})
    
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
    
    full_context = f"СТАТИЧЕСКИЙ ПРОФИЛЬ: {STATIC_CLIENT_PROFILE}\n\nДИНАМИКА: {dynamic_summary}"
    
    return AnalyzeResponse(summary=full_context, categories=sorted_categories)


# ==================== ФРОНТЕНД ====================

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
    response = analyze_mock_transactions()
    if request.session_id not in USER_STATE_CACHE:
        USER_STATE_CACHE[request.session_id] = {}
    USER_STATE_CACHE[request.session_id]["summary"] = response.summary
    return response


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    if not collection:
        raise HTTPException(status_code=500, detail="База знаний не загружена")
    ai_response = await get_llm_response(request.session_id, request.message)
    return ChatResponse(role="assistant", content=ai_response)


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    SERVER_URL = "http://localhost:8000"
    
    def start_server():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    print(f"✅ Zaman AI с полным API роутером запущен: {SERVER_URL}")
    print(f"📚 Документация API: {SERVER_URL}/docs")
    print(f"🌐 Открываем браузер...")
    
    threading.Timer(1.5, lambda: webbrowser.open(SERVER_URL)).start()
    start_server()