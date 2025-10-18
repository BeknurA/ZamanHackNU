'use client';

import React, { useState, useCallback, useMemo } from 'react';

// --- КОНФИГУРАЦИЯ БЭКЕНДА ---
// !!! ВНИМАНИЕ: Если вы разместили FastAPI на Render, замените URL !!!
const API_BASE_URL = 'http://localhost:8000';
const SESSION_ID = 'demo_user_123'; // Хардкодим ID для демо
// -----------------------------

// Типы данных
interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface AnalysisData {
  summary: string;
  categories: { [key: string]: number };
}

// Простой компонент для Чат-пузыря
const ChatBubble: React.FC<Message> = ({ role, content }) => {
  const isUser = role === 'user';
  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-xl p-3 my-1 rounded-lg shadow-md ${
          isUser
            ? 'bg-blue-500 text-white rounded-br-none'
            : 'bg-gray-100 text-gray-800 rounded-tl-none'
        }`}
      >
        {content}
      </div>
    </div>
  );
};

// Главная страница приложения
const HomePage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(false);
  
  // Добавляем сообщение в историю
  const addMessage = useCallback((message: Message) => {
    setMessages(prev => [...prev, message]);
  }, []);

  // 1. ФУНКЦИЯ АНАЛИЗА (Связь с /api/analyze)
  const handleAnalyze = async () => {
    if (loading) return;
    setLoading(true);
    addMessage({ role: 'user', content: 'Инициирован анализ финансовой выписки...' });

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: SESSION_ID }),
      });
      
      if (!response.ok) throw new Error('Ошибка API анализа');

      const data: AnalysisData = await response.json();
      setAnalysis(data);
      
      const categoryList = Object.entries(data.categories)
        .map(([key, value]) => `${key}: ${Math.round(value)} KZT`)
        .join(', ');

      addMessage({ 
        role: 'assistant', 
        content: `Я успешно проанализировал вашу выписку. ${data.summary}. Основные категории трат: ${categoryList}. Чем могу помочь дальше?`
      });

    } catch (error) {
      console.error(error);
      addMessage({ role: 'assistant', content: 'Ошибка анализа. Проверьте, запущен ли бэкенд.' });
    } finally {
      setLoading(false);
    }
  };

  // 2. ФУНКЦИЯ ЧАТА (Связь с /api/chat)
  const handleChat = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    addMessage({ role: 'user', content: userMessage });
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          session_id: SESSION_ID, 
          message: userMessage 
        }),
      });

      if (!response.ok) throw new Error('Ошибка API чата');

      const data = await response.json();
      addMessage({ role: 'assistant', content: data.content });

    } catch (error) {
      console.error(error);
      addMessage({ role: 'assistant', content: 'Произошла ошибка при получении ответа от AI.' });
    } finally {
      setLoading(false);
    }
  };

  // Визуализация категорий (Простой список для хакатона)
  const CategoryChart = useMemo(() => {
    if (!analysis) return null;
    const total = Object.values(analysis.categories).reduce((sum, val) => sum + val, 0);

    return (
      <div className="mt-4 p-4 bg-white rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-2">Обзор расходов (Визуализация)</h3>
        <div className="space-y-1">
          {Object.entries(analysis.categories).map(([category, amount]) => (
            <div key={category} className="flex justify-between text-sm">
              <span className="font-medium">{category}</span>
              <span className="text-gray-600">{((amount / total) * 100).toFixed(1)}% ({Math.round(amount)} KZT)</span>
            </div>
          ))}
        </div>
      </div>
    );
  }, [analysis]);


  return (
    <div className="flex h-screen bg-gray-50">
      {/* Левая панель: Цели (Визионерство) */}
      <div className="w-1/4 p-6 bg-white shadow-xl flex flex-col">
        <h2 className="text-xl font-bold mb-6 text-green-700">Zaman AI Assistant</h2>
        
        <div className="flex-grow">
            <h3 className="text-lg font-semibold mb-3">Мои Финансовые Цели</h3>
            <div className="border border-green-200 p-4 rounded-lg bg-green-50">
                <p className="font-medium">Первый взнос на квартиру</p>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                    <div className="bg-green-600 h-2.5 rounded-full" style={{ width: '35%' }}></div>
                </div>
                <p className="text-sm text-gray-500 mt-1">Прогресс: 35%</p>
            </div>

            <h3 className="text-lg font-semibold mt-6 mb-3">Функционал</h3>
            <button 
                onClick={handleAnalyze} 
                disabled={loading || analysis !== null}
                className="w-full bg-yellow-500 text-white py-2 rounded-lg hover:bg-yellow-600 disabled:bg-gray-400 transition"
            >
                {analysis ? '✅ Анализ загружен' : (loading ? 'Анализ...' : '1. Загрузить выписку (Анализ)')}
            </button>
            <button 
                onClick={() => addMessage({role: 'user', content: 'Я очень устал и расстроен, что не могу сэкономить. Дай совет по борьбе со стрессом.'})}
                disabled={loading}
                className="w-full bg-green-500 text-white py-2 rounded-lg hover:bg-green-600 disabled:bg-gray-400 transition mt-2"
            >
                2. Кнопка "Я в стрессе"
            </button>
            <button 
                onClick={() => alert('Архитектура готова (Whisper API), но для демо используйте текст.')}
                className="w-full bg-gray-500 text-white py-2 rounded-lg hover:bg-gray-600 transition mt-2"
            >
                [🎙️ Голосовой режим]
            </button>
        </div>
      </div>

      {/* Правая панель: Чат и Аналитика */}
      <div className="w-3/4 flex flex-col">
        {/* Аналитика/Визуализация */}
        {analysis && (
            <div className="p-6 border-b border-gray-200">
                {CategoryChart}
            </div>
        )}

        {/* Чат */}
        <div className="flex-grow overflow-y-auto p-6 space-y-4 bg-gray-50">
          {messages.length === 0 && (
            <ChatBubble role="assistant" content="Здравствуйте! Я ваш персональный AI-ассистент Zaman Bank. Нажмите 'Загрузить выписку', чтобы начать анализ ваших финансов." />
          )}
          {messages.map((msg, index) => (
            <ChatBubble key={index} role={msg.role} content={msg.content} />
          ))}
          {loading && <ChatBubble role="assistant" content="Печатаю..." />}
        </div>

        {/* Форма ввода */}
        <div className="p-6 border-t border-gray-200 bg-white">
          <div className="flex space-x-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleChat()}
              placeholder="Спросите меня о продуктах, целях или расходах..."
              disabled={loading}
              className="flex-grow p-3 border border-gray-300 rounded-lg focus:ring-green-500 focus:border-green-500"
            />
            <button
              onClick={handleChat}
              disabled={loading || !input.trim()}
              className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 disabled:bg-gray-400 transition"
            >
              Отправить
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;