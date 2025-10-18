# goals_habits.py
# Модуль для работы с финансовыми целями и формированием привычек

from typing import Dict, List, Any
from datetime import datetime, timedelta
import json


class GoalsHabitsManager:
    """Управление финансовыми целями и привычками клиента"""
    
    def __init__(self):
        self.goals_templates = {
            "savings": {
                "name": "Накопление средств",
                "icon": "💰",
                "milestones": [10000, 50000, 100000, 500000],
                "tips": [
                    "Переводите 10% от дохода сразу после зарплаты",
                    "Используйте правило 'Заплати себе первым'",
                    "Автоматизируйте переводы на накопительный счет"
                ]
            },
            "debt_free": {
                "name": "Освобождение от долгов",
                "icon": "🎯",
                "milestones": [25, 50, 75, 100],  # % погашения
                "tips": [
                    "Метод 'снежного кома': начните с малых долгов",
                    "Рефинансирование в исламский продукт снижает нагрузку",
                    "Отслеживайте прогресс еженедельно для мотивации"
                ]
            },
            "expense_reduction": {
                "name": "Снижение трат",
                "icon": "📉",
                "categories": ["Кафе и рестораны", "Развлечения/Хобби", "Такси"],
                "tips": [
                    "Правило 24 часов: откладывайте импульсивные покупки на сутки",
                    "Ведите дневник трат - это снижает расходы на 15-20%",
                    "Используйте принцип 'Одна покупка — одна вещь продана'"
                ]
            },
            "halal_transition": {
                "name": "Переход на халяльные продукты",
                "icon": "☪️",
                "steps": [
                    "Изучить принципы исламского финансирования",
                    "Проанализировать текущие продукты на соответствие Шариату",
                    "Постепенно заменить 1-2 продукта на исламские",
                    "Полностью перейти на этичные финансы"
                ],
                "benefits": [
                    "Прозрачность всех операций",
                    "Отсутствие скрытых комиссий и процентов",
                    "Душевное спокойствие"
                ]
            },
            "investment": {
                "name": "Начать инвестировать",
                "icon": "📈",
                "starting_amounts": [10000, 50000, 100000],
                "islamic_options": [
                    "Депозит Мудараба (совместное инвестирование)",
                    "Золотые слитки (защита от инфляции)",
                    "Сукук (исламские облигации)"
                ]
            }
        }
        
        self.habit_challenges = {
            "coffee_challenge": {
                "name": "7-дневный кофейный челлендж",
                "description": "Вместо кофе в кафе - готовьте дома",
                "duration_days": 7,
                "expected_savings": 15000,  # за неделю
                "difficulty": "easy",
                "tips": [
                    "День 1-2: Купите качественный кофе домой (3000 KZT)",
                    "День 3-4: Заведите красивую термокружку",
                    "День 5-7: Похвалите себя - вы сэкономили 12000 KZT!",
                    "Бонус: Переведите сэкономленное на депозит Мудараба"
                ]
            },
            "no_delivery_week": {
                "name": "Неделя без доставки еды",
                "description": "Готовьте дома или берите ланчбокс на работу",
                "duration_days": 7,
                "expected_savings": 20000,
                "difficulty": "medium",
                "tips": [
                    "Планируйте меню на неделю заранее",
                    "Готовьте большими порциями и замораживайте",
                    "Инвестируйте в хорошие контейнеры",
                    "Соцсети: следите за #mealprep для вдохновения"
                ]
            },
            "taxi_to_walk": {
                "name": "Пешком вместо такси",
                "description": "Маршруты до 2 км - пешком или на велосипеде",
                "duration_days": 14,
                "expected_savings": 12000,
                "difficulty": "easy",
                "health_benefits": [
                    "10 000 шагов в день снижают стресс на 47%",
                    "Экономия времени: в пробках пешком быстрее",
                    "Здоровье + экономия = двойная польза"
                ]
            },
            "subscription_audit": {
                "name": "Ревизия подписок",
                "description": "Отмените неиспользуемые подписки",
                "duration_days": 1,
                "expected_savings": 8000,  # в месяц
                "difficulty": "easy",
                "checklist": [
                    "Netflix, Spotify, YouTube Premium - используете ли?",
                    "Онлайн-кинотеатры: можно ли делить аккаунт с семьей?",
                    "Фитнес-приложения: заменить на бесплатные?",
                    "Облачные хранилища: проверьте лимиты бесплатных тарифов"
                ]
            }
        }
    
    def suggest_goal(self, client_context: Dict[str, Any]) -> Dict[str, Any]:
        """Предлагает цель на основе финансового профиля"""
        
        income = client_context.get("monthly_salary_in_kzt", 0)
        debt = client_context.get("loan_payment_out_avg", 0)
        top_category = client_context.get("top_spending_category_1", "")
        
        # Логика выбора цели
        if debt > income * 0.3:  # Долг больше 30% дохода
            return self._build_goal_plan("debt_free", client_context)
        
        elif "Кафе" in top_category or "Рестораны" in top_category:
            return self._build_goal_plan("expense_reduction", client_context)
        
        elif income > 500000 and debt == 0:
            return self._build_goal_plan("investment", client_context)
        
        else:
            return self._build_goal_plan("savings", client_context)
    
    def _build_goal_plan(self, goal_type: str, client_context: Dict) -> Dict:
        """Создает персонализированный план достижения цели"""
        
        template = self.goals_templates.get(goal_type)
        if not template:
            return {}
        
        income = client_context.get("monthly_salary_in_kzt", 0)
        
        plan = {
            "goal_type": goal_type,
            "name": template["name"],
            "icon": template["icon"],
            "personalized_target": self._calculate_target(goal_type, income),
            "timeline_months": self._estimate_timeline(goal_type, income),
            "action_steps": template.get("tips", []),
            "progress_tracking": self._create_progress_tracker(goal_type)
        }
        
        return plan
    
    def _calculate_target(self, goal_type: str, income: float) -> Dict:
        """Рассчитывает реалистичную цель"""
        
        if goal_type == "savings":
            # Цель: 3 месячных дохода (резервный фонд)
            return {
                "amount": income * 3,
                "description": "Резервный фонд на 3 месяца"
            }
        
        elif goal_type == "expense_reduction":
            # Снижение трат на 15%
            return {
                "percentage": 15,
                "description": "Снижение необязательных трат на 15%"
            }
        
        elif goal_type == "investment":
            # Начать с 10% от дохода
            return {
                "monthly_amount": income * 0.1,
                "description": "Ежемесячные инвестиции 10% от дохода"
            }
        
        return {}
    
    def _estimate_timeline(self, goal_type: str, income: float) -> int:
        """Оценивает реалистичные сроки"""
        
        if goal_type == "savings":
            return 6  # 6 месяцев на резервный фонд
        elif goal_type == "debt_free":
            return 12  # год на значительное снижение долга
        elif goal_type == "expense_reduction":
            return 2  # 2 месяца на формирование привычки
        elif goal_type == "halal_transition":
            return 3  # 3 месяца на плавный переход
        else:
            return 1  # начать инвестировать можно сразу
    
    def _create_progress_tracker(self, goal_type: str) -> Dict:
        """Создает систему отслеживания прогресса"""
        
        return {
            "frequency": "weekly",  # еженедельные чек-ины
            "metrics": self._get_metrics_for_goal(goal_type),
            "celebration_milestones": [25, 50, 75, 100],  # % прогресса
            "motivation": [
                "Вы на правильном пути! 🎉",
                "Половина пути позади! 💪",
                "Почти у цели - не останавливайтесь! 🚀",
                "Поздравляю, цель достигнута! 🏆"
            ]
        }
    
    def _get_metrics_for_goal(self, goal_type: str) -> List[str]:
        """Определяет метрики для отслеживания"""
        
        metrics_map = {
            "savings": ["Сумма накоплений", "% от целевой суммы"],
            "debt_free": ["Остаток долга", "% погашения"],
            "expense_reduction": ["Траты в категории", "% снижения"],
            "investment": ["Сумма инвестиций", "Доходность"],
            "halal_transition": ["Количество переведенных продуктов", "% халяльности"]
        }
        
        return metrics_map.get(goal_type, ["Прогресс"])
    
    def suggest_habit_challenge(self, spending_category: str) -> Dict[str, Any]:
        """Предлагает челлендж на основе категории трат"""
        
        category_to_challenge = {
            "Кафе и рестораны": "coffee_challenge",
            "Развлечения/Хобби": "no_delivery_week",
            "Транспорт/Такси": "taxi_to_walk",
            "Связь/Интернет": "subscription_audit"
        }
        
        challenge_key = category_to_challenge.get(spending_category, "coffee_challenge")
        challenge = self.habit_challenges[challenge_key]
        
        # Добавляем персонализацию
        challenge["start_date"] = datetime.now().strftime("%d.%m.%Y")
        challenge["end_date"] = (datetime.now() + timedelta(days=challenge["duration_days"])).strftime("%d.%m.%Y")
        
        return challenge
    
    def format_goal_for_chat(self, goal_plan: Dict) -> str:
        """Форматирует план цели для чата"""
        
        output = f"{goal_plan['icon']} {goal_plan['name']}\n\n"
        
        if "amount" in goal_plan.get("personalized_target", {}):
            target = goal_plan["personalized_target"]
            output += f"Цель: {target['amount']:,.0f} KZT ({target['description']})\n"
        
        output += f"Срок: {goal_plan['timeline_months']} месяцев\n\n"
        output += "Шаги к успеху:\n"
        
        for i, step in enumerate(goal_plan.get("action_steps", []), 1):
            output += f"{i}. {step}\n"
        
        return output
    
    def format_challenge_for_chat(self, challenge: Dict) -> str:
        """Форматирует челлендж для чата"""
        
        output = f"🎯 {challenge['name']}\n\n"
        output += f"{challenge['description']}\n\n"
        output += f"Длительность: {challenge['duration_days']} дней\n"
        output += f"Ожидаемая экономия: {challenge['expected_savings']:,.0f} KZT\n"
        output += f"Сложность: {challenge['difficulty']}\n\n"
        
        output += "План действий:\n"
        for i, tip in enumerate(challenge.get("tips", []), 1):
            output += f"{i}. {tip}\n"
        
        if "health_benefits" in challenge:
            output += "\nБонус для здоровья:\n"
            for benefit in challenge["health_benefits"]:
                output += f"💚 {benefit}\n"
        
        return output


# Пример использования
if __name__ == "__main__":
    manager = GoalsHabitsManager()
    
    # Тестовый профиль клиента
    test_client = {
        "monthly_salary_in_kzt": 488634,
        "loan_payment_out_avg": 117151,
        "top_spending_category_1": "Кафе и рестораны (102,000+)"
    }
    
    # Предложить цель
    goal = manager.suggest_goal(test_client)
    print("=== ПРЕДЛОЖЕННАЯ ЦЕЛЬ ===")
    print(manager.format_goal_for_chat(goal))
    
    # Предложить челлендж
    challenge = manager.suggest_habit_challenge("Кафе и рестораны")
    print("\n=== ПРЕДЛОЖЕННЫЙ ЧЕЛЛЕНДЖ ===")
    print(manager.format_challenge_for_chat(challenge))