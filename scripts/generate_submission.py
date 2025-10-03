#!/usr/bin/env python3
"""
Скрипт для генерации submission.csv на основе test.csv

Использует LLM для преобразования вопросов на естественном языке
в HTTP запросы к Finam TradeAPI.

Использование:
    python scripts/generate_submission.py [OPTIONS]

Опции:
    --test-file PATH      Путь к test.csv (по умолчанию: data/processed/test.csv)
    --train-file PATH     Путь к train.csv (по умолчанию: data/processed/train.csv)
    --output-file PATH    Путь к submission.csv (по умолчанию: data/processed/submission.csv)
    --num-examples INT    Количество примеров для few-shot (по умолчанию: 7)
    --batch-size INT      Размер батча для обработки (по умолчанию: 5)
"""

import csv
import random
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import click
from tqdm import tqdm  # type: ignore[import-untyped]
import concurrent.futures
from typing import List, Dict, Tuple

from src.app.core.llm import call_llm

VALID_ENDPOINTS = [
    "/v1/exchanges",
    "/v1/assets",
    "/v1/assets/{symbol}",
    "/v1/assets/{symbol}/params",
    "/v1/assets/{symbol}/schedule",
    "/v1/assets/{symbol}/options",
    "/v1/instruments/{symbol}/quotes/latest",
    "/v1/instruments/{symbol}/orderbook",
    "/v1/instruments/{symbol}/trades/latest",
    "/v1/instruments/{symbol}/bars",
    "/v1/accounts/{account_id}",
    "/v1/accounts/{account_id}/orders",
    "/v1/accounts/{account_id}/orders/{order_id}",
    "/v1/accounts/{account_id}/trades",
    "/v1/accounts/{account_id}/transactions",
    "/v1/sessions",
    "/v1/sessions/details",
    "/v1/accounts/{account_id}/orders",
    "/v1/accounts/{account_id}/orders/{order_id}",
]

TIMEFRAMES = [
    "TIME_FRAME_M1", "TIME_FRAME_M5", "TIME_FRAME_M15", "TIME_FRAME_M30",
    "TIME_FRAME_H1", "TIME_FRAME_H4", "TIME_FRAME_D", "TIME_FRAME_W", "TIME_FRAME_MN"
]

def build_faiss_index(train_examples: List[Dict[str, str]]) -> Tuple:
    model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")  
    questions = [ex["question"] for ex in train_examples]
    embeddings = model.encode(questions, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, model, train_examples

def find_similar_examples(question: str, index, embeddings, model, train_examples: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, str]]:
    q_emb = model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    return [train_examples[i] for i in I[0]]

def calculate_cost(usage: Dict, model: str) -> float:
    """Рассчитать стоимость запроса на основе usage и модели"""
    # Цены OpenRouter (примерные, в $ за 1M токенов)
    # Источник: https://openrouter.ai/models
    pricing = {
        "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "openai/gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "openai/gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
        "anthropic/claude-3-sonnet": {"prompt": 3.00, "completion": 15.00},
        "anthropic/claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
    }

    # Получаем цены для модели (по умолчанию как для gpt-4o-mini)
    prices = pricing.get(model, {"prompt": 0.15, "completion": 0.60})

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Считаем стоимость (цена за 1M токенов)
    prompt_cost = (prompt_tokens / 1_000_000) * prices["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * prices["completion"]

    return prompt_cost + completion_cost

def load_train_examples(train_file: Path, num_examples: int = 100) -> List[Dict[str, str]]:
    """Загрузить примеры из train.csv для few-shot learning"""
    examples = []
    with open(train_file, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            examples.append({"question": row["question"], "type": row["type"], "request": row["request"]})

    # Берем разнообразные примеры (GET, POST, DELETE)
    get_examples = [e for e in examples if e["type"] == "GET"]
    post_examples = [e for e in examples if e["type"] == "POST"]
    delete_examples = [e for e in examples if e["type"] == "DELETE"]

    # Формируем сбалансированный набор
    selected = []
    selected.extend(random.sample(get_examples, min(num_examples // 2, len(get_examples))))
    selected.extend(random.sample(post_examples, min(num_examples // 4, len(post_examples))))
    selected.extend(random.sample(delete_examples, min(num_examples // 4, len(delete_examples))))

    return selected[:num_examples]

def create_prompt(questions: List[str], examples: List[Dict[str, str]], similar_examples_list: List[List[Dict[str, str]]]) -> str:
    """Создать промпт для LLM с few-shot примерами для батча вопросов"""
    prompt = """Ты - эксперт по Finam TradeAPI. Твоя задача - преобразовать вопрос на русском языке в HTTP запрос к API.

API Documentation:
- GET /v1/exchanges - список бирж
- GET /v1/assets - поиск инструментов
- GET /v1/assets/{symbol} - информация об инструменте
- GET /v1/assets/{symbol}/params - параметры инструмента для счета
- GET /v1/assets/{symbol}/schedule - расписание торгов
- GET /v1/assets/{symbol}/options - опционы на базовый актив
- GET /v1/instruments/{symbol}/quotes/latest - последняя котировка
- GET /v1/instruments/{symbol}/orderbook - биржевой стакан
- GET /v1/instruments/{symbol}/trades/latest - лента сделок
- GET /v1/instruments/{symbol}/bars - исторические свечи
  (параметры: timeframe, interval.start_time, interval.end_time)
- GET /v1/accounts/{account_id} - информация о счете
- GET /v1/accounts/{account_id}/orders - список ордеров
- GET /v1/accounts/{account_id}/orders/{order_id} - информация об ордере
- GET /v1/accounts/{account_id}/trades - история сделок
- GET /v1/accounts/{account_id}/transactions - транзакции по счету
- POST /v1/sessions - создание новой сессии
- POST /v1/sessions/details - детали текущей сессии
- POST /v1/accounts/{account_id}/orders - создание ордера
- DELETE /v1/accounts/{account_id}/orders/{order_id} - отмена ордера

Timeframes: TIME_FRAME_M1, TIME_FRAME_M5, TIME_FRAME_M15, TIME_FRAME_M30,
TIME_FRAME_H1, TIME_FRAME_H4, TIME_FRAME_D, TIME_FRAME_W, TIME_FRAME_MN

Инструкции:
Сначала проанализируй каждый вопрос шаг за шагом:
1. Определи, какой эндпоинт API подходит (на основе документации).
2. Выдели параметры из вопроса (например, symbol, timeframe, dates). Даты форматируй как YYYY-MM-DD. Timeframe используй только из списка.
3. Выбери правильный метод (GET для чтения, POST для создания, DELETE для удаления).
4. Если вопрос не связан с API, используй fallback: GET /v1/assets

Примеры:

"""

    for ex in examples:
        prompt += f'Вопрос: "{ex["question"]}"\n'
        prompt += f"Ответ: {ex['type']} {ex['request']}\n\n"


    for idx, sim_exs in enumerate(similar_examples_list):
        prompt += f"\nПохожие примеры для вопроса {idx+1}:\n"
        for ex in sim_exs:
            prompt += f'Вопрос: "{ex["question"]}"\n'
            prompt += f"Ответ: {ex['type']} {ex['request']}\n\n"

    prompt += "\nТеперь обработай следующие вопросы. Для каждого выдай ответ только в формате: [ID] METHOD /path?params\n"
    for i, q in enumerate(questions):
        prompt += f'Вопрос [{i+1}]: "{q}"\n'

    prompt += "\nОтветы (только в указанном формате, без объяснений):"

    return prompt

def parse_llm_response(response: str) -> List[Tuple[str, str]]:
    """Парсинг ответа LLM для батча в список (type, request)"""
    lines = response.strip().split("\n")
    results = []
    for line in lines:
        if line.startswith("[") and "]" in line:
            try:
                # Извлекаем [ID] METHOD /path
                parts = line.split(" ", 2)
                method = parts[1].strip()
                request = parts[2].strip()
                results.append((method, request))
            except:
                results.append(("GET", "/v1/assets"))  # Fallback
    return results

def validate_request(method: str, request: str) -> Tuple[str, str]:
    """Валидация и пост-обработка запроса"""
    if method not in ["GET", "POST", "DELETE"]:
        method = "GET"

    if not request.startswith("/"):
        request = "/" + request

    base_path = request.split("?")[0]
    if base_path not in VALID_ENDPOINTS:
        from difflib import get_close_matches
        closest = get_close_matches(base_path, VALID_ENDPOINTS, n=1, cutoff=0.6)
        if closest:
            base_path = closest[0]

    return method, base_path + ("?" + request.split("?")[1] if "?" in request else "")

def generate_api_calls_batch(questions: List[str], examples: List[Dict[str, str]], similar_examples_list: List[List[Dict[str, str]]], model: str, max_retries: int = 3) -> Tuple[List[Dict[str, str]], float]:
    """Сгенерировать API запросы для батча вопросов"""
    total_cost = 0.0
    results = [{"type": "GET", "request": "/v1/assets"} for _ in questions]  # Fallback

    for attempt in range(max_retries):
        try:
            prompt = create_prompt(questions, examples, similar_examples_list)
            messages = [{"role": "user", "content": prompt}]
            response = call_llm(messages, temperature=0.0 if attempt == 0 else 0.1, max_tokens=500)
            llm_answer = response["choices"][0]["message"]["content"].strip()

            parsed = parse_llm_response(llm_answer)
            if len(parsed) == len(questions):
                for i, (method, request) in enumerate(parsed):
                    method, request = validate_request(method, request)
                    results[i] = {"type": method, "request": request}

            usage = response.get("usage", {})
            total_cost += calculate_cost(usage, model)
            break  # Успех
        except Exception as e:
            click.echo(f"⚠️  Ошибка при генерации батча (попытка {attempt+1}): {e}", err=True)
            if attempt == max_retries - 1:
                click.echo("🚨 Максимум попыток достигнут, использую fallback", err=True)

    return results, total_cost

@click.command()
@click.option(
    "--test-file",
    type=click.Path(exists=True, path_type=Path),
    default="data/processed/test.csv",
    help="Путь к test.csv",
)
@click.option(
    "--train-file",
    type=click.Path(exists=True, path_type=Path),
    default="data/processed/train.csv",
    help="Путь к train.csv",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default="data/processed/submission.csv",
    help="Путь к submission.csv",
)
@click.option("--num-examples", type=int, default=7, help="Количество примеров для few-shot")
@click.option("--batch-size", type=int, default=5, help="Размер батча для обработки")
def main(test_file: Path, train_file: Path, output_file: Path, num_examples: int, batch_size: int) -> None:
    """Генерация submission.csv для хакатона"""
    from src.app.core.config import get_settings

    click.echo("🚀 Генерация submission файла...")
    click.echo(f"📖 Загрузка примеров из {train_file}...")

    # Получаем настройки для определения модели
    settings = get_settings()
    model = settings.openrouter_model

    # Загружаем примеры для few-shot
    examples = load_train_examples(train_file, 100)
    index, emb_matrix, emb_model, train_examples = build_faiss_index(examples)
    click.echo(f"✅ Загружено {len(examples)} примеров для few-shot learning")
    click.echo(f"🤖 Используется модель: {model}")

    # Читаем тестовый набор
    click.echo(f"📖 Чтение {test_file}...")
    test_items = []
    with open(test_file, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            test_items.append({"uid": row["uid"], "question": row["question"]})

    click.echo(f"✅ Найдено {len(test_items)} вопросов для обработки")

    # Генерируем ответы батчами с параллелизмом
    click.echo("\n🤖 Генерация API запросов с помощью LLM...")
    results = []
    total_cost = 0.0

    def process_batch(batch_items):
        questions = [item["question"] for item in batch_items]
        similar_examples_list = [find_similar_examples(q, index, emb_matrix, emb_model, train_examples, top_k=3) for q in questions]
        api_calls, cost = generate_api_calls_batch(questions, random.sample(examples, num_examples), similar_examples_list, model)
        return [{"uid": batch_items[i]["uid"], **call} for i, call in enumerate(api_calls)], cost

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # Параллелизм для батчей
        futures = []
        for i in range(0, len(test_items), batch_size):
            batch = test_items[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch))

        progress_bar = tqdm(futures, desc="Обработка батчей")
        for future in progress_bar:
            batch_results, batch_cost = future.result()
            results.extend(batch_results)
            total_cost += batch_cost
            progress_bar.set_postfix({"cost": f"${total_cost:.4f}"})

    # Записываем в submission.csv
    click.echo(f"\n💾 Сохранение результатов в {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "type", "request"], delimiter=";")
        writer.writeheader()
        writer.writerows(results)

    click.echo(f"✅ Готово! Создано {len(results)} записей в {output_file}")
    click.echo(f"\n💰 Общая стоимость генерации: ${total_cost:.4f}")
    click.echo(f"   Средняя стоимость на запрос: ${total_cost / len(results):.6f}")
    click.echo("\n📊 Статистика по типам запросов:")
    type_counts: Dict[str, int] = {}
    for r in results:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
    for method, count in sorted(type_counts.items()):
        click.echo(f"  {method}: {count}")

if __name__ == "__main__":
    main()