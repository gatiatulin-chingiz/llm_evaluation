# LLM Evaluation Framework

Фреймворк для оценки языковых моделей с поддержкой использования только с предзагруженными моделью и токенизатором.

## Возможности

- Оценка моделей с помощью LM Evaluation Harness
- Измерение скорости генерации
- Мониторинг системных ресурсов (CPU, RAM, GPU)
- Работа только с предзагруженными моделями
- Сохранение результатов в JSON формате
- Гибкая настройка параметров оценки

## Установка зависимостей

```bash
pip install transformers torch lm-eval psutil GPUtil
```

## Использование

### Из Jupyter Notebook

#### Простой способ:

```python
import sys
sys.path.append('./src')

from transformers import AutoTokenizer, AutoModelForCausalLM
from main import evaluate_preloaded_model

# Загрузка модели
model_name = "./Текстовые/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Оценка модели
results = evaluate_preloaded_model(
    model=model,
    tokenizer=tokenizer,
    model_name="Qwen3-0.6B",  # Опционально, для логирования
    tasks=["hellaswag", "gsm8k"],
    batch_size=4,
    num_samples=10,
    save_results=True
)
```

#### Расширенный способ:

```python
from main import ModelEvaluator

# Создание оценщика
evaluator = ModelEvaluator(
    model=model, 
    tokenizer=tokenizer, 
    model_name="Qwen3-0.6B"
)

# Отдельные этапы оценки
speed_metrics = evaluator.measure_generation_speed(num_samples=10)

# Полная оценка
full_results = evaluator.run_full_evaluation(
    tasks=["hellaswag", "mmlu", "gsm8k"],
    batch_size=8,
    num_samples=20,
    save_results=True
)
```

## Параметры

### evaluate_preloaded_model()

- `model`: Предзагруженная модель (обязательно)
- `tokenizer`: Предзагруженный токенизатор (обязательно)
- `model_name` (str, optional): Название модели для логирования и сохранения результатов
- `tasks` (list): Список задач для оценки (по умолчанию: ["hellaswag", "mmlu", "gsm8k"])
- `batch_size` (int): Размер батча (по умолчанию: 8)
- `num_samples` (int): Количество образцов для измерения скорости (по умолчанию: 10)
- `save_results` (bool): Сохранять ли результаты в файл (по умолчанию: True)

### ModelEvaluator.run_full_evaluation()

Те же параметры, что и у `evaluate_preloaded_model()`.

## Доступные задачи

- `hellaswag`: HellaSwag
- `mmlu`: Massive Multitask Language Understanding
- `gsm8k`: Grade School Math 8K
- `arc_easy`: AI2 Reasoning Challenge (Easy)
- `arc_challenge`: AI2 Reasoning Challenge (Challenge)
- `truthfulqa`: TruthfulQA
- `winogrande`: Winogrande
- `piqa`: Physical IQa

## Результаты

Функции возвращают словарь с результатами:

```python
{
    "model_name": "название_модели",
    "load_time": 0.0,  # Всегда 0, так как модель предзагружена
    "evaluation_time": 120.5,  # Время оценки
    "generation_speed": 15.2,  # Скорость генерации (токенов/сек)
    "total_tokens_generated": 1000,  # Общее количество сгенерированных токенов
    "system_metrics": {...},  # Системные метрики
    "results_file": "путь_к_файлу.json",  # Путь к сохраненному файлу
    "lm_eval_results": {...},  # Результаты LM Evaluation Harness
    "generation_speed_detailed": {...}  # Детальная статистика генерации
}
```

## Системные требования

- Python 3.7+
- CUDA-совместимая GPU (рекомендуется)
- Минимум 8GB RAM
- Для больших моделей рекомендуется 16GB+ RAM

## Примеры использования

Смотрите файл `example_usage.ipynb` для подробных примеров использования.

## Логирование

Все операции логируются в файл `model_evaluation.log` и выводятся в консоль.

## Поддерживаемые модели

Фреймворк поддерживает все модели, совместимые с Hugging Face Transformers:
- Qwen
- LLaMA
- GPT-2/3/4
- BLOOM
- T5
- И другие

## Важные замечания

- **Модель и токенизатор должны быть загружены заранее** - модуль не поддерживает загрузку моделей по имени
- **Обязательные параметры**: `model` и `tokenizer` должны быть переданы при инициализации
- **Эффективность**: Модуль оптимизирован для работы с уже загруженными моделями
- **Гибкость**: Можно выполнять как полную оценку, так и отдельные этапы (только скорость генерации)
