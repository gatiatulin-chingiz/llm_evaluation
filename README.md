# LLM Evaluation Framework

Минималистичный фреймворк для оценки языковых моделей (LLM) с поддержкой базовой и расширенной оценки.

## Возможности
- Быстрая базовая оценка производительности (скорость генерации, ресурсы)
- Расширенная оценка с тестами точности (LM Evaluation Harness)
- Сохранение результатов в компактный JSON
- Мониторинг CPU, RAM, GPU
- Простое сравнение нескольких моделей

## Установка
```bash
pip install transformers torch lm-eval psutil GPUtil
```

## Быстрый старт

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from main import evaluate_basic_model, evaluate_full_model

# Загрузка модели
model_name = "./path/to/model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Базовая оценка (только производительность)
basic_results = evaluate_basic_model(model, tokenizer, "MyModel")

# Расширенная оценка (включая точность)
full_results = evaluate_full_model(model, tokenizer, "MyModel", tasks=["hellaswag", "gsm8k"])
```

## Основные функции

- `evaluate_basic_model(model, tokenizer, model_name=None, num_samples=10, save_results=True)`
- `evaluate_full_model(model, tokenizer, model_name=None, tasks=["hellaswag", "mmlu", "gsm8k"], batch_size=8, num_samples=10, save_results=True)`

## Пример сравнения моделей

```python
from main import compare_models_basic, compare_models_full

model_configs = [
    {"name": "Qwen3-0.6B", "path": "./Qwen3-0.6B"},
    {"name": "Qwen3-1.5B", "path": "./Qwen3-1.5B"}
]

# Сравнение производительности
results = compare_models_basic(model_configs)

# Сравнение с тестами точности
results = compare_models_full([
    {"name": "Qwen3-0.6B", "path": "./Qwen3-0.6B", "tasks": ["hellaswag"]},
    {"name": "Qwen3-1.5B", "path": "./Qwen3-1.5B", "tasks": ["hellaswag"]}
])
```

## Структура результатов

- Все результаты сохраняются в папку `results/`.
- Формат JSON содержит только ключевые метрики: скорость, ресурсы, точность (если есть), промпты и ответы.

## Требования
- Python 3.7+
- torch, transformers, lm-eval, psutil, GPUtil
- CUDA GPU (рекомендуется для больших моделей)

## Примечания
- Модель и токенизатор должны быть загружены заранее (через Hugging Face Transformers)
- Для расширенной оценки требуется lm-eval
- Все параметры имеют разумные значения по умолчанию

---

**Минимальный, быстрый и удобный инструмент для оценки LLM.**
