import httpx
import chromadb
import os
import time
from dotenv import load_dotenv
import os

load_dotenv() # Эта команда читает ваш .env файл

# Теперь код может использовать ключи
API_KEY= os.getenv("API_KEY")
# --- Конфигурация API (из документа) ---
BASE_URL = "https://openai-hub.neuraldeep.tech"  #
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding_from_api(text: str, client: httpx.Client) -> list[float]:
    """
    Получает эмбеддинг для текста, используя предоставленный API.
    """
    print(f"Получение эмбеддинга для: {text[:50]}...")
    try:
        response = client.post(
            f"{BASE_URL}/v1/embeddings",  # Путь к API эмбеддингов
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "input": [text],
                "model": EMBEDDING_MODEL
            }
        )
        response.raise_for_status()  # Проверка на HTTP ошибки
        data = response.json()
        return data['data'][0]['embedding']
    except httpx.HTTPStatusError as e:
        print(f"Ошибка API эмбеддинга: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
    return []


def main():
    # 1. Чтение данных
    # (Гилфойл): Я не буду писать парсер. Просто скопируйте текст
    # с https://zamanbank.kz/ru/[cite: 27],
    # https://www.zamanbank.kz/ru/islamic-finance/islamskie-fi nansy[cite: 29],
    # https://www.zamanbank.kz/ru/islamic-finance/glossarij[cite: 31],
    # https://www.zamanbank.kz/ru/islamic-finance/otvety-na-v oprosy#question-faq-307 [cite: 34]
    # ...в один файл data.txt
    try:
        with open("../data.txt", "r", encoding="utf-8") as f:
            text_data = f.read()
    except FileNotFoundError:
        print("Ошибка: Файл data.txt не найден. Создайте его, скопировав текст с URL-адресов банка.")
        return

    # 2. Разделение на чанки (по абзацам)
    chunks = [chunk for chunk in text_data.split("\n\n") if len(chunk.strip()) > 100]
    print(f"Найдено {len(chunks)} текстовых чанков.")

    # 3. Инициализация ChromaDB (локально, бесплатно)
    client = chromadb.PersistentClient(path="./zaman_db")
    try:
        collection = client.create_collection(name="zaman_products")
    except chromadb.errors.UniqueConstraintError:
        print("Коллекция уже существует. Удаляем и создаем заново.")
        client.delete_collection(name="zaman_products")
        collection = client.create_collection(name="zaman_products")

    # 4. Векторизация и сохранение
    with httpx.Client(timeout=30.0) as http_client:
        embeddings = []
        doc_ids = []

        for i, chunk in enumerate(chunks):
            embedding = get_embedding_from_api(chunk, http_client)
            if embedding:
                embeddings.append(embedding)
                doc_ids.append(f"doc_{i}")
                # (Гилфойл): API может иметь лимит запросов.
                # Добавим небольшую задержку.
                time.sleep(0.5)

        if embeddings:
            collection.add(
                embeddings=embeddings,
                documents=chunks[:len(embeddings)],  # Убедимся, что кол-во совпадает
                ids=doc_ids
            )
            print(f"Успешно добавлено {len(embeddings)} документов в ChromaDB.")
        else:
            print("Не удалось получить эмбеддинги. База данных пуста.")


if __name__ == "__main__":
    main()