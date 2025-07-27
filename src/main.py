#!/usr/bin/env python3
"""
Скрипт для оценки моделей с логированием результатов и системных метрик
Поддерживает использование только с предзагруженными моделью и токенизатором
"""

import time
import json
import logging
import psutil
import GPUtil
from datetime import datetime
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Класс для мониторинга системных ресурсов"""
    
    @staticmethod
    def get_cpu_info():
        """Получение информации о CPU"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_avg = sum(cpu_percent) / len(cpu_percent)
        return {
            "cpu_percent_per_core": cpu_percent,
            "cpu_avg_percent": cpu_avg,
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True)
        }
    
    @staticmethod
    def get_memory_info():
        """Получение информации о памяти"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        }
    
    @staticmethod
    def get_gpu_info():
        """Получение информации о GPU"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Берем первую GPU
                return {
                    "name": gpu.name,
                    "memory_total_gb": round(gpu.memoryTotal / 1024, 2),
                    "memory_used_gb": round(gpu.memoryUsed / 1024, 2),
                    "memory_free_gb": round(gpu.memoryFree / 1024, 2),
                    "utilization_percent": gpu.load * 100,
                    "temperature": gpu.temperature
                }
        except:
            return {"error": "GPU info not available"}
        return {"error": "No GPUs found"}

class ModelEvaluator:
    def __init__(self, model, tokenizer, model_name=None):
        """
        Инициализация оценщика модели
        
        Args:
            model: Предзагруженная модель (обязательно)
            tokenizer: Предзагруженный токенизатор (обязательно)
            model_name (str, optional): Название модели для логирования и сохранения результатов
        """
        if model is None or tokenizer is None:
            raise ValueError("model и tokenizer должны быть переданы (не None)")
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name or "preloaded_model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.system_monitor = SystemMonitor()
        
        self.logger.info(f"Инициализирован оценщик для модели: {self.model_name}")
        
    def log_system_resources(self, phase=""):
        """Логирование системных ресурсов"""
        cpu_info = self.system_monitor.get_cpu_info()
        memory_info = self.system_monitor.get_memory_info()
        gpu_info = self.system_monitor.get_gpu_info()
        
        self.logger.info(f"=== Системные ресурсы {phase} ===")
        self.logger.info(f"CPU: {cpu_info['cpu_avg_percent']:.1f}% (ядер: {cpu_info['cpu_count_logical']})")
        self.logger.info(f"RAM: {memory_info['used_gb']:.1f}/{memory_info['total_gb']:.1f} GB ({memory_info['percent']:.1f}%)")
        
        if "error" not in gpu_info:
            self.logger.info(f"GPU: {gpu_info['name']}")
            self.logger.info(f"GPU VRAM: {gpu_info['memory_used_gb']:.1f}/{gpu_info['memory_total_gb']:.1f} GB")
            self.logger.info(f"GPU Util: {gpu_info['utilization_percent']:.1f}%")
        else:
            self.logger.info(f"GPU: {gpu_info['error']}")
        
        return {
            "cpu": cpu_info,
            "memory": memory_info,
            "gpu": gpu_info
        }
    
    def evaluate_model(self, tasks=["hellaswag", "mmlu", "gsm8k"], batch_size=8):
        """Оценка модели с помощью LM Evaluation Harness"""
        self.logger.info(f"Запуск оценки для задач: {tasks}")
        self.log_system_resources("(перед оценкой)")
        
        # Создаем обертку для LM Eval
        lm_obj = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size
        )
        
        start_time = time.time()
        
        # Запуск оценки
        results = evaluator.simple_evaluate(
            model=lm_obj,
            tasks=tasks,
            batch_size=batch_size,
            device=self.device,
            limit=None  # Обработать все примеры
        )
        
        eval_time = time.time() - start_time
        self.log_system_resources("(после оценки)")
        
        return results, eval_time
    
    def measure_generation_speed(self, num_samples=10):
        """Замер скорости генерации в токенах/сек"""
        self.logger.info("Замер скорости генерации...")
        self.log_system_resources("(перед генерацией)")
        
        test_prompts = [
            "Объясни, как работает искусственный интеллект.",
            "Расскажи о важности образования в современном мире.",
            "Опиши процесс фотосинтеза.",
            "Какие преимущества даёт изучение иностранных языков?",
            "Объясни концепцию машинного обучения простыми словами.",
            "Опиши, что такое квантовая физика.",
            "Расскажи о влиянии технологий на общество.",
            "Объясни, как работает блокчейн.",
            "Опиши процесс эволюции видов.",
            "Расскажи о важности экологии."
        ] * (num_samples // 10 + 1)
        
        test_prompts = test_prompts[:num_samples]
        
        total_tokens = 0
        total_time = 0
        generation_stats = []
        
        for i, prompt in enumerate(test_prompts):
            # Логируем ресурсы каждые 5 итераций
            if i % 5 == 0:
                self.log_system_resources(f"(генерация {i+1}/{num_samples})")
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            gen_time = time.time() - start_time
            
            # Подсчет токенов
            new_tokens = outputs.shape[1] - inputs.shape[1]
            tokens_per_sec = new_tokens / gen_time
            
            total_tokens += new_tokens
            total_time += gen_time
            
            stat = {
                "prompt_number": i+1,
                "prompt_length": inputs.shape[1],
                "generated_tokens": new_tokens,
                "generation_time": gen_time,
                "tokens_per_second": tokens_per_sec
            }
            generation_stats.append(stat)
            
            self.logger.info(f"Промпт {i+1}: {new_tokens} токенов за {gen_time:.2f}с = {tokens_per_sec:.2f} токенов/сек")
        
        avg_tokens_per_sec = total_tokens / total_time
        self.logger.info(f"Средняя скорость: {avg_tokens_per_sec:.2f} токенов/сек")
        self.log_system_resources("(после генерации)")
        
        return {
            "average_tokens_per_second": avg_tokens_per_sec,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "detailed_stats": generation_stats
        }
    
    def save_results(self, results, eval_time, speed_metrics, system_metrics, filename=None):
        """Сохранение результатов в JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name.replace('/', '_')}_evaluation_results_{timestamp}.json"
        
        output_data = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "system_info": system_metrics,
            "evaluation_time_seconds": eval_time,
            "lm_eval_results": results,
            "generation_speed": speed_metrics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Результаты сохранены в {filename}")
        return filename
    
    def run_full_evaluation(self, tasks=["hellaswag", "mmlu", "gsm8k"], batch_size=8, num_samples=10, save_results=True):
        """
        Полная оценка модели
        
        Args:
            tasks (list): Список задач для оценки
            batch_size (int): Размер батча
            num_samples (int): Количество образцов для измерения скорости
            save_results (bool): Сохранять ли результаты в файл
            
        Returns:
            dict: Словарь с результатами оценки
        """
        try:
            # Логируем начальные системные ресурсы
            initial_system_metrics = self.log_system_resources("(начало)")
            
            # 1. Оценка точности (если указаны задачи)
            eval_time = 0.0
            results = {}
            if tasks:
                results, eval_time = self.evaluate_model(tasks, batch_size)
            
            # 2. Замер скорости генерации
            speed_metrics = self.measure_generation_speed(num_samples)
            
            # 3. Финальные системные метрики
            final_system_metrics = self.log_system_resources("(окончание)")
            
            # 4. Сбор всех системных метрик
            system_metrics = {
                "initial": initial_system_metrics,
                "final": final_system_metrics,
                "model_load_time": 0.0,  # Модель уже загружена
                "evaluation_time": eval_time,
                "generation_time": speed_metrics["total_time"]
            }
            
            # 5. Сохранение результатов (опционально)
            result_file = None
            if save_results:
                result_file = self.save_results(results, eval_time, speed_metrics, system_metrics)
            
            # 6. Подготовка результатов для возврата
            evaluation_summary = {
                "model_name": self.model_name,
                "load_time": 0.0,  # Модель уже загружена
                "evaluation_time": eval_time,
                "generation_speed": speed_metrics['average_tokens_per_second'],
                "total_tokens_generated": speed_metrics['total_tokens'],
                "system_metrics": system_metrics,
                "results_file": result_file,
                "lm_eval_results": results,
                "generation_speed_detailed": speed_metrics
            }
            
            # 7. Вывод основных метрик
            self._print_summary(evaluation_summary)
            
            return evaluation_summary
            
        except Exception as e:
            self.logger.error(f"Ошибка при оценке: {str(e)}")
            self.logger.exception("Подробности ошибки:")
            raise
    
    def _print_summary(self, summary):
        """Вывод сводки результатов"""
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("="*60)
        print(f"Модель: {summary['model_name']}")
        print(f"Время загрузки: {summary['load_time']:.2f} сек (модель предзагружена)")
        print(f"Время оценки: {summary['evaluation_time']:.2f} сек")
        
        # Системные ресурсы
        final_mem = summary['system_metrics']["final"]["memory"]
        final_gpu = summary['system_metrics']["final"]["gpu"]
        print(f"Использование RAM: {final_mem['used_gb']:.1f}/{final_mem['total_gb']:.1f} GB ({final_mem['percent']:.1f}%)")
        if "error" not in final_gpu:
            print(f"Использование GPU VRAM: {final_gpu['memory_used_gb']:.1f}/{final_gpu['memory_total_gb']:.1f} GB")
            print(f"Загрузка GPU: {final_gpu['utilization_percent']:.1f}%")
        
        # Точность по задачам
        results = summary['lm_eval_results']
        if results and 'results' in results:
            print("\nТочность по задачам:")
            for task, metrics in results['results'].items():
                if 'acc,none' in metrics:
                    acc = metrics['acc,none']
                    stderr = metrics.get('acc_stderr,none', 0)
                    print(f"  {task}: {acc:.4f} ± {stderr:.4f}")
                elif 'exact_match,none' in metrics:
                    em = metrics['exact_match,none']
                    stderr = metrics.get('exact_match_stderr,none', 0)
                    print(f"  {task}: {em:.4f} ± {stderr:.4f}")
        
        # Скорость генерации
        print(f"\nПроизводительность:")
        print(f"  Скорость генерации: {summary['generation_speed']:.2f} токенов/сек")
        print(f"  Обработано токенов: {summary['total_tokens_generated']}")
        print(f"  Время генерации: {summary['generation_speed_detailed']['total_time']:.2f} сек")
        print(f"  Общее время: {summary['load_time'] + summary['evaluation_time'] + summary['generation_speed_detailed']['total_time']:.2f} сек")
        if summary['results_file']:
            print(f"  Результаты сохранены в: {summary['results_file']}")
        print("="*60)

# Функция для удобного использования из Jupyter Notebook
def evaluate_preloaded_model(model, tokenizer, model_name=None, tasks=["hellaswag", "mmlu", "gsm8k"], 
                           batch_size=8, num_samples=10, save_results=True):
    """
    Удобная функция для оценки предзагруженной модели из Jupyter Notebook
    
    Args:
        model: Предзагруженная модель (обязательно)
        tokenizer: Предзагруженный токенизатор (обязательно)
        model_name (str, optional): Название модели для логирования и сохранения результатов
        tasks (list): Список задач для оценки (по умолчанию: ["hellaswag", "mmlu", "gsm8k"])
        batch_size (int): Размер батча (по умолчанию: 8)
        num_samples (int): Количество образцов для измерения скорости (по умолчанию: 10)
        save_results (bool): Сохранять ли результаты в файл (по умолчанию: True)
        
    Returns:
        dict: Словарь с результатами оценки
    """
    evaluator_obj = ModelEvaluator(model=model, tokenizer=tokenizer, model_name=model_name)
    return evaluator_obj.run_full_evaluation(tasks, batch_size, num_samples, save_results)

"""
===============================================================================
                    ПРИМЕР ИСПОЛЬЗОВАНИЯ ИЗ JUPYTER NOTEBOOK
===============================================================================

ЭТАП 1: ПОДГОТОВКА И ИМПОРТ
===============================================================================

# 1.1 Настройка путей и импорт библиотек
import sys
sys.path.append('./src')  # Добавляем путь к модулю main.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from main import evaluate_preloaded_model, ModelEvaluator

# 1.2 Загрузка модели и токенизатора
model_name = "./Текстовые/Qwen3-0.6B"  # Укажите путь к вашей модели

# Токенизатор - преобразует текст в токены для модели
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Модель - основная языковая модель
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Автоматический выбор типа данных (float16/float32)
    device_map="auto"    # Автоматическое размещение на CPU/GPU
)

===============================================================================
ЭТАП 2: ОСНОВНЫЕ СПОСОБЫ ОЦЕНКИ
===============================================================================

СПОСОБ 1: БЫСТРАЯ ОЦЕНКА (рекомендуется для большинства случаев)
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Полная оценка модели одной функцией
ПРЕИМУЩЕСТВА: 
- Минимум кода
- Автоматическое выполнение всех этапов
- Готовые результаты в удобном формате

КОГДА ИСПОЛЬЗОВАТЬ:
✓ Быстрая оценка модели
✓ Сравнение нескольких моделей  
✓ Исследовательские цели
✓ Когда нужны все метрики сразу

results = evaluate_preloaded_model(
    model=model,                    # Предзагруженная модель (ОБЯЗАТЕЛЬНО)
    tokenizer=tokenizer,            # Предзагруженный токенизатор (ОБЯЗАТЕЛЬНО)
    model_name="Qwen3-0.6B",        # Название для логирования (опционально)
    tasks=["hellaswag", "gsm8k"],   # Задачи для оценки точности
    batch_size=4,                   # Размер батча (влияет на память/скорость)
    num_samples=10,                 # Образцы для измерения скорости генерации
    save_results=True               # Сохранить результаты в JSON файл
)

СПОСОБ 2: ДЕТАЛЬНЫЙ КОНТРОЛЬ (для продвинутых пользователей)
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Поэтапная оценка с полным контролем процесса
ПРЕИМУЩЕСТВА:
- Контроль каждого этапа оценки
- Доступ к промежуточным результатам
- Гибкая настройка параметров

КОГДА ИСПОЛЬЗОВАТЬ:
✓ Отладка и анализ производительности
✓ Выполнение только части оценки
✓ Интеграция в сложные пайплайны
✓ Когда нужен детальный контроль

# 2.1 Создание оценщика
evaluator = ModelEvaluator(
    model=model,                    # Предзагруженная модель
    tokenizer=tokenizer,            # Предзагруженный токенизатор
    model_name="Qwen3-0.6B"         # Название для логирования
)

# 2.2 Этап A: Измерение скорости генерации
# НАЗНАЧЕНИЕ: Определяет токенов/сек
# ПОЛЕЗНО: Сравнение производительности, оптимизация
speed_metrics = evaluator.measure_generation_speed(num_samples=5)

# 2.3 Этап B: Полная оценка с настройками
# НАЗНАЧЕНИЕ: Все этапы оценки с контролем параметров
full_results = evaluator.run_full_evaluation(
    tasks=["hellaswag"],            # Задачи для оценки
    batch_size=2,                   # Размер батча
    num_samples=5,                  # Образцы для скорости
    save_results=False              # Не сохранять в файл
)

===============================================================================
ЭТАП 3: АНАЛИЗ РЕЗУЛЬТАТОВ
===============================================================================

# 3.1 Основные метрики производительности
print(f"Скорость генерации: {results['generation_speed']:.2f} токенов/сек")
print(f"Время оценки: {results['evaluation_time']:.2f} секунд")

# 3.2 Анализ системных ресурсов
system_metrics = results['system_metrics']
print(f"Использование RAM: {system_metrics['final']['memory']['used_gb']:.1f} GB")
print(f"Использование GPU: {system_metrics['final']['gpu']['memory_used_gb']:.1f} GB")

# 3.3 Анализ точности по задачам
if results['lm_eval_results'] and 'results' in results['lm_eval_results']:
    for task, metrics in results['lm_eval_results']['results'].items():
        if 'acc,none' in metrics:
            print(f"Точность {task}: {metrics['acc,none']:.4f}")

===============================================================================
ЭТАП 4: СПЕЦИАЛЬНЫЕ СЦЕНАРИИ
===============================================================================

СЦЕНАРИЙ 1: ТОЛЬКО ИЗМЕРЕНИЕ СКОРОСТИ
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Быстрая оценка производительности без тестов точности
КОГДА ИСПОЛЬЗОВАТЬ:
✓ Быстрое сравнение скорости моделей
✓ Когда точность уже известна
✓ Оптимизация производительности
✓ Ограниченные вычислительные ресурсы

performance_results = evaluate_preloaded_model(
    model=model,
    tokenizer=tokenizer,
    tasks=[],                       # Пустой список = пропустить LM Evaluation
    num_samples=20,                 # Больше образцов для точности
    save_results=False              # Не сохранять результаты
)

СЦЕНАРИЙ 2: ТОЛЬКО ОЦЕНКА ТОЧНОСТИ
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Фокус на качестве модели без измерения производительности
КОГДА ИСПОЛЬЗОВАТЬ:
✓ Когда производительность не важна
✓ Финальная оценка качества
✓ Ограниченное время

accuracy_results = evaluator.run_full_evaluation(
    tasks=["hellaswag", "mmlu", "gsm8k"],  # Все важные задачи
    batch_size=8,                          # Оптимальный батч
    num_samples=0,                         # Пропустить измерение скорости
    save_results=True
)

СЦЕНАРИЙ 3: СРАВНЕНИЕ НЕСКОЛЬКИХ МОДЕЛЕЙ
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Систематическое сравнение производительности и качества
КОГДА ИСПОЛЬЗОВАТЬ:
✓ Выбор лучшей модели для задачи
✓ Исследование влияния размера модели
✓ Документирование экспериментов

def compare_models(model_configs):
    \"\"\"
    Сравнение нескольких моделей
    
    Args:
        model_configs: Список конфигураций моделей
        [{'name': 'model1', 'path': './path1', 'tasks': ['hellaswag']}, ...]
    
    Returns:
        dict: Результаты сравнения по моделям
    \"\"\"
    results = {}
    
    for config in model_configs:
        print(f"Оценка модели: {config['name']}")
        
        # Загрузка модели
        tokenizer = AutoTokenizer.from_pretrained(config['path'])
        model = AutoModelForCausalLM.from_pretrained(
            config['path'], torch_dtype="auto", device_map="auto"
        )
        
        # Оценка
        model_results = evaluate_preloaded_model(
            model=model,
            tokenizer=tokenizer,
            model_name=config['name'],
            tasks=config.get('tasks', ['hellaswag']),
            batch_size=config.get('batch_size', 4),
            num_samples=config.get('num_samples', 10),
            save_results=True
        )
        
        results[config['name']] = model_results
        
        # Очистка памяти
        del model, tokenizer
        torch.cuda.empty_cache()
    
    return results

# ПРИМЕР ИСПОЛЬЗОВАНИЯ СРАВНЕНИЯ:
# model_configs = [
#     {'name': 'Qwen3-0.6B', 'path': './Текстовые/Qwen3-0.6B', 'tasks': ['hellaswag']},
#     {'name': 'Qwen3-1.5B', 'path': './Текстовые/Qwen3-1.5B', 'tasks': ['hellaswag']}
# ]
# comparison_results = compare_models(model_configs)

===============================================================================
СПРАВОЧНИК ПАРАМЕТРОВ
===============================================================================

tasks (список задач):
- "hellaswag"     - здравый смысл и логика
- "mmlu"          - многозадачное понимание языка  
- "gsm8k"         - математические задачи
- "arc_easy"      - рассуждения (легкий уровень)
- "arc_challenge" - рассуждения (сложный уровень)
- "truthfulqa"    - правдивость ответов
- "winogrande"    - разрешение местоимений
- "piqa"          - физический здравый смысл

batch_size (размер батча):
- 1-2: Экономия памяти, медленная работа
- 4-8: Оптимальный баланс (рекомендуется)
- 16+: Быстрая работа, много памяти

num_samples (образцы для скорости):
- 5-10: Быстрая оценка
- 20-50: Точная оценка
- 100+: Очень точная оценка

save_results (сохранение):
- True: Сохранить в JSON файл с временной меткой
- False: Только в памяти
"""