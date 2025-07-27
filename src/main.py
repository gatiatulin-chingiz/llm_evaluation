#!/usr/bin/env python3
"""
Скрипт для оценки моделей с логированием результатов и системных метрик
Поддерживает использование только с предзагруженными моделью и токенизатором

Основные компоненты:
- SystemMonitor: Мониторинг системных ресурсов (CPU, RAM, GPU)
- ModelEvaluator: Основной класс для оценки языковых моделей
- Функции-помощники: evaluate_basic_model, evaluate_full_model

Использование:
1. Загрузите модель и токенизатор с помощью transformers
2. Вызовите evaluate_basic_model() для быстрой оценки производительности
3. Вызовите evaluate_full_model() для полной оценки с тестами точности
"""

import time
import json
import logging
import psutil
import GPUtil
import os
from datetime import datetime
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Настройка логирования
def setup_logging():
    """
    Настройка логирования для модуля.
    Создает папку results и настраивает логирование в файл и консоль.
    """
    # Создаем папку results, если её нет
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'model_evaluation.log'), encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True  # Принудительно перезаписываем настройки логирования
    )
    
    # Получаем логгер для модуля
    logger = logging.getLogger(__name__)
    logger.info("Логирование настроено успешно")
    return logger

# Инициализируем логирование при импорте модуля
logger = setup_logging()

def ensure_logging_setup():
    """
    Убеждается, что логирование настроено правильно.
    Полезно вызывать при импорте модуля в Jupyter Notebook.
    """
    global logger
    logger = setup_logging()
    logger.info("Логирование проверено и настроено")
    return logger

class SystemMonitor:
    """
    Класс для мониторинга системных ресурсов во время оценки модели.
    
    Отслеживает:
    - Использование CPU (по ядрам и общее)
    - Использование оперативной памяти (RAM)
    - Использование GPU (VRAM, загрузка, температура)
    
    Все методы статические, так как не требуют состояния объекта.
    """
    
    @staticmethod
    def get_cpu_info():
        """
        Получение детальной информации о CPU.
        
        Returns:
            dict: Словарь с информацией о CPU:
                - cpu_percent_per_core (list): Процент загрузки каждого ядра
                - cpu_avg_percent (float): Средний процент загрузки всех ядер
                - cpu_count (int): Количество физических ядер
                - cpu_count_logical (int): Количество логических ядер (с гипертрейдингом)
        """
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_avg = sum(cpu_percent) / len(cpu_percent)
        return {
            "cpu_percent_per_core": cpu_percent,  # Список процентов загрузки каждого ядра
            "cpu_avg_percent": cpu_avg,           # Средний процент загрузки всех ядер
            "cpu_count": psutil.cpu_count(),      # Количество физических ядер
            "cpu_count_logical": psutil.cpu_count(logical=True)  # Количество логических ядер
        }
    
    @staticmethod
    def get_memory_info():
        """
        Получение информации об использовании оперативной памяти.
        
        Returns:
            dict: Словарь с информацией о RAM:
                - total_gb (float): Общий объем RAM в GB
                - available_gb (float): Доступная RAM в GB
                - used_gb (float): Используемая RAM в GB
                - percent (float): Процент использования RAM
        """
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),      # Общий объем в GB
            "available_gb": round(memory.available / (1024**3), 2),  # Доступная память в GB
            "used_gb": round(memory.used / (1024**3), 2),        # Используемая память в GB
            "percent": memory.percent                            # Процент использования
        }
    
    @staticmethod
    def get_gpu_info():
        """
        Получение информации о GPU (если доступна).
        
        Returns:
            dict: Словарь с информацией о GPU:
                - name (str): Название GPU
                - memory_total_gb (float): Общий объем VRAM в GB
                - memory_used_gb (float): Используемая VRAM в GB
                - memory_free_gb (float): Свободная VRAM в GB
                - utilization_percent (float): Процент загрузки GPU
                - temperature (float): Температура GPU в градусах Цельсия
                
            Или словарь с ошибкой:
                - error (str): Описание ошибки, если GPU недоступна
        """
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Берем первую GPU (обычно основную)
                return {
                    "name": gpu.name,                                    # Название GPU
                    "memory_total_gb": round(gpu.memoryTotal / 1024, 2), # Общий VRAM в GB
                    "memory_used_gb": round(gpu.memoryUsed / 1024, 2),   # Используемый VRAM в GB
                    "memory_free_gb": round(gpu.memoryFree / 1024, 2),   # Свободный VRAM в GB
                    "utilization_percent": gpu.load * 100,               # Загрузка GPU в процентах
                    "temperature": gpu.temperature                        # Температура в градусах
                }
        except:
            return {"error": "GPU info not available"}  # Ошибка при получении информации
        return {"error": "No GPUs found"}  # GPU не найдена

class ModelEvaluator:
    """
    Основной класс для оценки языковых моделей.
    
    Предоставляет методы для:
    - Мониторинга системных ресурсов
    - Оценки точности модели на различных задачах
    - Измерения скорости генерации
    - Сохранения результатов в JSON файлы
    
    Атрибуты:
        model: Предзагруженная модель (torch.nn.Module)
        tokenizer: Предзагруженный токенизатор
        model_name (str): Название модели для логирования
        device (str): Устройство для вычислений ('cuda' или 'cpu')
        logger: Объект логирования
        system_monitor: Экземпляр SystemMonitor для мониторинга ресурсов
    """
    
    def __init__(self, model, tokenizer, model_name=None):
        """
        Инициализация оценщика модели.
        
        Args:
            model: Предзагруженная модель (обязательно)
            tokenizer: Предзагруженный токенизатор (обязательно)
            model_name (str, optional): Название модели для логирования и сохранения результатов
                                       Если не указано, используется "preloaded_model"
        
        Raises:
            ValueError: Если model или tokenizer равны None
        """
        if model is None or tokenizer is None:
            raise ValueError("model и tokenizer должны быть переданы (не None)")
        
        self.model = model                    # Предзагруженная модель
        self.tokenizer = tokenizer            # Предзагруженный токенизатор
        self.model_name = model_name or "preloaded_model"  # Название для логирования
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Устройство (GPU/CPU)
        self.system_monitor = SystemMonitor() # Мониторинг системных ресурсов
        
        # Убеждаемся, что логирование работает
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            # Если логгер не настроен, настраиваем его
            setup_logging()
            self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Инициализирован оценщик для модели: {self.model_name}")
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Токенизатор: {type(self.tokenizer).__name__}")
        self.logger.info(f"Модель: {type(self.model).__name__}")
        
    def log_system_resources(self, phase=""):
        """
        Логирование текущего состояния системных ресурсов.
        
        Args:
            phase (str): Описание фазы выполнения (например, "перед оценкой", "после генерации")
        
        Returns:
            dict: Словарь с информацией о системных ресурсах:
                - cpu: Информация о CPU
                - memory: Информация о RAM
                - gpu: Информация о GPU (если доступна)
        """
        cpu_info = self.system_monitor.get_cpu_info()
        memory_info = self.system_monitor.get_memory_info()
        gpu_info = self.system_monitor.get_gpu_info()
        
        # Логируем информацию в консоль и файл
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
            "cpu": cpu_info,      # Информация о CPU
            "memory": memory_info, # Информация о RAM
            "gpu": gpu_info        # Информация о GPU
        }
    
    def evaluate_model(self, tasks=["hellaswag", "mmlu", "gsm8k"], batch_size=8):
        """
        Оценка модели с помощью LM Evaluation Harness на стандартных задачах.
        
        Использует библиотеку lm-eval для оценки точности модели на различных задачах:
        - hellaswag: здравый смысл и логика
        - mmlu: многозадачное понимание языка
        - gsm8k: математические задачи
        
        Args:
            tasks (list): Список задач для оценки. Доступные задачи:
                - "hellaswag": здравый смысл и логика
                - "mmlu": многозадачное понимание языка
                - "gsm8k": математические задачи
                - "arc_easy": рассуждения (легкий уровень)
                - "arc_challenge": рассуждения (сложный уровень)
                - "truthfulqa": правдивость ответов
                - "winogrande": разрешение местоимений
                - "piqa": физический здравый смысл
            batch_size (int): Размер батча для обработки. Больший размер = быстрее, но больше памяти
        
        Returns:
            tuple: (results, eval_time)
                - results (dict): Результаты оценки с метриками точности
                - eval_time (float): Время выполнения оценки в секундах
        """
        self.logger.info(f"Запуск оценки для задач: {tasks}")
        self.log_system_resources("(перед оценкой)")
        
        # Создаем обертку для LM Eval - адаптер между нашей моделью и библиотекой оценки
        lm_obj = HFLM(
            pretrained=self.model,      # Наша предзагруженная модель
            tokenizer=self.tokenizer,   # Наш предзагруженный токенизатор
            batch_size=batch_size       # Размер батча для обработки
        )
        
        start_time = time.time()
        
        # Запуск оценки с помощью LM Evaluation Harness
        results = evaluator.simple_evaluate(
            model=lm_obj,           # Обертка модели
            tasks=tasks,            # Список задач для оценки
            batch_size=batch_size,  # Размер батча
            device=self.device,     # Устройство (GPU/CPU)
            limit=None              # Обработать все примеры (None = без ограничений)
        )
        
        eval_time = time.time() - start_time
        self.log_system_resources("(после оценки)")
        
        return results, eval_time
    
    def measure_generation_speed(self, num_samples=10):
        """
        Замер скорости генерации модели в токенах в секунду.
        
        Тестирует модель на наборе промптов и измеряет:
        - Время генерации для каждого промпта
        - Количество сгенерированных токенов
        - Скорость генерации (токенов/сек)
        - Сохраняет промпты и ответы модели
        
        Args:
            num_samples (int): Количество тестовых промптов для измерения скорости.
                             Больше образцов = точнее измерение, но дольше выполнение.
        
        Returns:
            dict: Словарь с метриками скорости генерации:
                - average_tokens_per_second (float): Средняя скорость генерации
                - total_tokens (int): Общее количество сгенерированных токенов
                - total_time (float): Общее время генерации в секундах
                - detailed_stats (list): Детальная статистика по каждому промпту
                - prompts_and_responses (list): Промпты и ответы модели
        """
        self.logger.info("Замер скорости генерации...")
        self.log_system_resources("(перед генерацией)")
        
        # Набор тестовых промптов для измерения скорости (математика и страхование)
        test_prompts = [
            "Опиши подробно, что такое градиентный спуск и как он работает.",
            "Игральную кость с 6 гранями бросают дважды. Найдите вероятность того, что оба раза выпало число, большее 3.",
            "Какие полисы в страховании можно считать убыточными?",
            "Объясни принцип работы франшизы в автостраховании и её влияние на стоимость полиса.",
            "Что такое математическое ожидание и как его вычислить?"
        ]
        
        # Повторяем промпты, если нужно больше образцов
        if num_samples > len(test_prompts):
            repeats = (num_samples // len(test_prompts)) + 1
            test_prompts = test_prompts * repeats
        
        test_prompts = test_prompts[:num_samples]  # Обрезаем до нужного количества (5 промптов)
        
        total_tokens = 0      # Общее количество сгенерированных токенов
        total_time = 0        # Общее время генерации
        generation_stats = [] # Детальная статистика по каждому промпту
        speeds = []           # Список скоростей для расчета среднего
        prompts_and_responses = []  # Промпты и ответы модели
        
        for i, prompt in enumerate(test_prompts):
            # Логируем ресурсы каждые 5 итераций для мониторинга
            if i % 5 == 0:
                self.log_system_resources(f"(генерация {i+1}/{num_samples})")
            
            # Кодируем промпт в токены и перемещаем на нужное устройство
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            start_time = time.time()
            
            # Генерация ответа модели
            with torch.no_grad():  # Отключаем градиенты для экономии памяти
                outputs = self.model.generate(
                    inputs,                                   # Входные токены
                    max_new_tokens=500,                       # Максимум новых токенов для генерации
                    do_sample=True,                           # Использовать сэмплирование (не жадный поиск)
                    temperature=0.7,                          # Температура для разнообразия ответов
                    pad_token_id=self.tokenizer.eos_token_id  # Токен окончания последовательности
                )
            
            gen_time = time.time() - start_time
            
            # Декодируем ответ модели
            model_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Подсчет сгенерированных токенов (исключаем входные токены)
            new_tokens = outputs.shape[1] - inputs.shape[1]
            tokens_per_sec = new_tokens / gen_time
            
            total_tokens += new_tokens
            total_time += gen_time
            speeds.append(tokens_per_sec)  # Добавляем скорость в список для среднего
            
            # Сохраняем промпт и ответ модели
            prompts_and_responses.append({
                "prompt_number": i+1,
                "prompt": prompt,
                "response": model_response,
                "prompt_tokens": inputs.shape[1],
                "response_tokens": new_tokens
            })
            
            # Сохраняем детальную статистику для каждого промпта
            stat = {
                "prompt_number": i+1,           # Номер промпта
                "prompt_length": inputs.shape[1],  # Длина входного промпта в токенах
                "generated_tokens": new_tokens,    # Количество сгенерированных токенов
                "generation_time": gen_time,       # Время генерации в секундах
                "tokens_per_second": tokens_per_sec # Скорость генерации для этого промпта
            }
            generation_stats.append(stat)
            
            self.logger.info(f"Промпт {i+1}: {new_tokens} токенов за {gen_time:.2f}с = {tokens_per_sec:.2f} токенов/сек")
        
        # Рассчитываем среднюю скорость как среднее арифметическое всех измерений
        avg_tokens_per_sec = sum(speeds) / len(speeds) if speeds else 0
        self.logger.info(f"Средняя скорость (среднее): {avg_tokens_per_sec:.2f} токенов/сек")
        self.log_system_resources("(после генерации)")
        
        return {
            "average_tokens_per_second": avg_tokens_per_sec,  # Средняя скорость генерации (среднее)
            "total_tokens": total_tokens,                     # Общее количество токенов
            "total_time": total_time,                         # Общее время генерации
            "detailed_stats": generation_stats,               # Детальная статистика по промптам
            "speed_measurements": speeds,                     # Все измерения скорости для анализа
            "prompts_and_responses": prompts_and_responses    # Промпты и ответы модели
        }
    
    def save_results(self, results, eval_time, speed_metrics, system_metrics, filename=None):
        """
        Сохранение упрощенных результатов оценки в JSON файл.
        
        Сохраняет только самые важные метрики:
        - Основная информация о модели
        - Ключевые метрики производительности
        - Результаты точности по задачам
        - Краткие системные метрики
        
        Структура выходного JSON:
        {
            "model_name": "Qwen3-0.6B",
            "timestamp": "2024-01-15T10:30:45",
            "evaluation_summary": {
                "evaluation_time_seconds": 120.5,
                "generation_speed_tokens_per_sec": 15.2,
                "total_tokens_generated": 1520,
                "total_generation_time_seconds": 100.0
            },
            "accuracy_results": {
                "hellaswag": {"accuracy": 0.7523, "stderr": 0.0123},
                "gsm8k": {"exact_match": 0.2345, "stderr": 0.0234}
            },
            "system_summary": {
                "ram_used_gb": 8.5,
                "ram_total_gb": 16.0,
                "cpu_percent": 45.2,
                "gpu_vram_used_gb": 4.2,
                "gpu_vram_total_gb": 8.0
            },
            "prompts_and_responses": [
                {
                    "prompt_number": 1,
                    "prompt": "Опиши подробно, что такое градиентный спуск и как он работает.",
                    "response": "Градиентный спуск - это алгоритм оптимизации...",
                    "prompt_tokens": 15,
                    "response_tokens": 45
                }
            ]
        }
        
        Args:
            results (dict): Результаты оценки точности от lm-eval
            eval_time (float): Время выполнения оценки в секундах
            speed_metrics (dict): Метрики скорости генерации
            system_metrics (dict): Системные метрики (CPU, RAM, GPU)
            filename (str, optional): Имя файла для сохранения.
                                    Если None, генерируется автоматически
        
        Returns:
            str: Путь к сохраненному файлу
        """
        # Создаем папку results, если её нет
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            self.logger.info(f"Создана папка {results_dir}")
        
        if filename is None:
            # Генерируем имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{self.model_name.replace('/', '_')}_evaluation_results_{timestamp}.json"
        
        # Полный путь к файлу в папке results
        filepath = os.path.join(results_dir, filename)
        
        # Извлекаем только важные результаты точности
        accuracy_results = {}
        if results and 'results' in results:
            for task, metrics in results['results'].items():
                if 'acc,none' in metrics:
                    accuracy_results[task] = {
                        'accuracy': round(metrics['acc,none'], 4),
                        'stderr': round(metrics.get('acc_stderr,none', 0), 4)
                    }
                elif 'exact_match,none' in metrics:
                    accuracy_results[task] = {
                        'exact_match': round(metrics['exact_match,none'], 4),
                        'stderr': round(metrics.get('exact_match_stderr,none', 0), 4)
                    }
        
        # Краткие системные метрики (только финальные)
        final_system = system_metrics.get('final', {})
        system_summary = {
            'ram_used_gb': round(final_system.get('memory', {}).get('used_gb', 0), 1),
            'ram_total_gb': round(final_system.get('memory', {}).get('total_gb', 0), 1),
            'cpu_percent': round(final_system.get('cpu', {}).get('cpu_avg_percent', 0), 1),
            'gpu_vram_used_gb': round(final_system.get('gpu', {}).get('memory_used_gb', 0), 1) if 'error' not in final_system.get('gpu', {}) else 0,
            'gpu_vram_total_gb': round(final_system.get('gpu', {}).get('memory_total_gb', 0), 1) if 'error' not in final_system.get('gpu', {}) else 0
        }
        
        # Упрощенная структура данных для сохранения
        output_data = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "evaluation_summary": {
                "evaluation_time_seconds": round(eval_time, 2),
                "generation_speed_tokens_per_sec": round(speed_metrics['average_tokens_per_second'], 2),
                "total_tokens_generated": speed_metrics['total_tokens'],
                "total_generation_time_seconds": round(speed_metrics['total_time'], 2)
            },
            "accuracy_results": accuracy_results,
            "system_summary": system_summary,
            "prompts_and_responses": speed_metrics.get('prompts_and_responses', [])
        }
        
        # Сохранение в JSON файл с отступами для читаемости
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Упрощенные результаты сохранены в {filepath}")
        return filepath

    def save_basic_results(self, basic_metrics, filename=None):
        """
        Сохранение упрощенных базовых результатов оценки в JSON файл.
        
        Сохраняет только самые важные метрики базовой оценки:
        - Основная информация о модели
        - Ключевые метрики производительности
        - Краткие системные метрики
        
        Структура выходного JSON:
        {
            "model_name": "Qwen3-0.6B",
            "timestamp": "2024-01-15T10:30:45",
            "evaluation_type": "basic",
            "evaluation_summary": {
                "generation_speed_tokens_per_sec": 15.2,
                "total_tokens_generated": 1520,
                "total_generation_time_seconds": 100.0
            },
            "system_summary": {
                "ram_used_gb": 8.5,
                "ram_total_gb": 16.0,
                "cpu_percent": 45.2,
                "gpu_vram_used_gb": 4.2,
                "gpu_vram_total_gb": 8.0
            },
            "prompts_and_responses": [
                {
                    "prompt_number": 1,
                    "prompt": "Опиши подробно, что такое градиентный спуск и как он работает.",
                    "response": "Градиентный спуск - это алгоритм оптимизации...",
                    "prompt_tokens": 15,
                    "response_tokens": 45
                }
            ]
        }
        
        Args:
            basic_metrics (dict): Базовые метрики оценки
            filename (str, optional): Имя файла для сохранения.
                                    Если None, генерируется автоматически
        
        Returns:
            str: Путь к сохраненному файлу
        """
        # Создаем папку results, если её нет
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            self.logger.info(f"Создана папка {results_dir}")
        
        if filename is None:
            # Генерируем имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{self.model_name.replace('/', '_')}_basic_evaluation_{timestamp}.json"
        
        # Полный путь к файлу в папке results
        filepath = os.path.join(results_dir, filename)
        
        # Краткие системные метрики (только финальные)
        final_system = basic_metrics.get('system_metrics', {}).get('final', {})
        system_summary = {
            'ram_used_gb': round(final_system.get('memory', {}).get('used_gb', 0), 1),
            'ram_total_gb': round(final_system.get('memory', {}).get('total_gb', 0), 1),
            'cpu_percent': round(final_system.get('cpu', {}).get('cpu_avg_percent', 0), 1),
            'gpu_vram_used_gb': round(final_system.get('gpu', {}).get('memory_used_gb', 0), 1) if 'error' not in final_system.get('gpu', {}) else 0,
            'gpu_vram_total_gb': round(final_system.get('gpu', {}).get('memory_total_gb', 0), 1) if 'error' not in final_system.get('gpu', {}) else 0
        }
        
        # Упрощенная структура данных для сохранения базовых результатов
        output_data = {
            "model_name": basic_metrics['model_name'],
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "basic",
            "evaluation_summary": {
                "generation_speed_tokens_per_sec": round(basic_metrics['generation_speed'], 2),
                "total_tokens_generated": basic_metrics['total_tokens_generated'],
                "total_generation_time_seconds": round(basic_metrics['generation_time'], 2)
            },
            "system_summary": system_summary,
            "prompts_and_responses": basic_metrics.get('generation_speed_detailed', {}).get('prompts_and_responses', [])
        }
        
        # Сохранение в JSON файл с отступами для читаемости
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Упрощенные базовые результаты сохранены в {filepath}")
        return filepath

    def run_basic_evaluation(self, num_samples=10, save_results=True):
        """
        Базовая оценка модели - только системные и производительные метрики.
        
        Выполняет быструю оценку модели без тестов точности:
        1. Логирует начальные системные ресурсы
        2. Измеряет скорость генерации на тестовых промптах
        3. Логирует финальные системные ресурсы
        4. Сохраняет результаты (опционально)
        
        Преимущества:
        - Быстрое выполнение (не требует дополнительных данных)
        - Минимальное использование ресурсов
        - Подходит для быстрой проверки производительности
        
        Args:
            num_samples (int): Количество образцов для измерения скорости генерации.
                             Больше образцов = точнее измерение, но дольше выполнение.
                             Рекомендуется: 10-50 для быстрой оценки, 100+ для точной.
            save_results (bool): Сохранять ли результаты в JSON файл.
                               True = сохранить, False = только в памяти
        
        Returns:
            dict: Словарь с базовыми результатами оценки:
                - model_name (str): Название модели
                - load_time (float): Время загрузки (всегда 0.0, модель предзагружена)
                - evaluation_time (float): Время оценки (всегда 0.0, нет тестов точности)
                - generation_speed (float): Средняя скорость генерации в токенах/сек
                - total_tokens_generated (int): Общее количество сгенерированных токенов
                - generation_time (float): Общее время генерации в секундах
                - total_time (float): Общее время выполнения (равно generation_time)
                - system_metrics (dict): Системные метрики (начальные и финальные)
                - generation_speed_detailed (dict): Детальная статистика скорости
                - results_file (str, optional): Путь к сохраненному файлу (если save_results=True)
        
        Raises:
            Exception: При ошибке во время оценки (логируется и перебрасывается)
        """
        try:
            self.logger.info("Запуск базовой оценки модели")
            
            # 1. Логируем начальные системные ресурсы
            initial_system_metrics = self.log_system_resources("(начало)")
            
            # 2. Замер скорости генерации на тестовых промптах
            speed_metrics = self.measure_generation_speed(num_samples)
            
            # 3. Финальные системные метрики для сравнения
            final_system_metrics = self.log_system_resources("(окончание)")
            
            # 4. Сбор всех базовых метрик в единую структуру
            basic_metrics = {
                "model_name": self.model_name,                                    # Название модели
                "load_time": 0.0,                                                # Время загрузки (модель предзагружена)
                "evaluation_time": 0.0,                                          # Время оценки (нет тестов точности)
                "generation_speed": speed_metrics['average_tokens_per_second'],  # Средняя скорость генерации
                "total_tokens_generated": speed_metrics['total_tokens'],         # Общее количество токенов
                "generation_time": speed_metrics["total_time"],                  # Время генерации
                "total_time": speed_metrics["total_time"],                       # Общее время = время генерации
                "system_metrics": {
                    "initial": initial_system_metrics,                           # Начальные системные метрики
                    "final": final_system_metrics,                               # Финальные системные метрики
                    "model_load_time": 0.0,                                      # Время загрузки модели
                    "evaluation_time": 0.0,                                      # Время оценки точности
                    "generation_time": speed_metrics["total_time"]               # Время генерации
                },
                "generation_speed_detailed": speed_metrics                       # Детальная статистика скорости
            }
            
            # 5. Сохранение результатов в файл (опционально)
            result_file = None
            if save_results:
                result_file = self.save_basic_results(basic_metrics)
                basic_metrics["results_file"] = result_file
            
            # 6. Вывод базовых метрик в консоль
            self._print_basic_summary(basic_metrics)
            
            return basic_metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка при базовой оценке: {str(e)}")
            self.logger.exception("Подробности ошибки:")
            raise

    def run_full_evaluation(self, tasks=["hellaswag", "mmlu", "gsm8k"], batch_size=8, num_samples=10, save_results=True):
        """
        Расширенная оценка модели - включает тесты точности и производительности.
        
        Выполняет полную оценку модели:
        1. Логирует начальные системные ресурсы
        2. Оценивает точность на указанных задачах (если tasks не пустой)
        3. Измеряет скорость генерации на тестовых промптах
        4. Логирует финальные системные ресурсы
        5. Сохраняет результаты (опционально)
        
        Преимущества:
        - Полная оценка качества модели
        - Включает как точность, так и производительность
        - Профессиональная оценка для исследований
        
        Args:
            tasks (list): Список задач для оценки точности. Доступные задачи:
                - "hellaswag": здравый смысл и логика
                - "mmlu": многозадачное понимание языка
                - "gsm8k": математические задачи
                - "arc_easy": рассуждения (легкий уровень)
                - "arc_challenge": рассуждения (сложный уровень)
                - "truthfulqa": правдивость ответов
                - "winogrande": разрешение местоимений
                - "piqa": физический здравый смысл
                Пустой список [] = пропустить оценку точности
            batch_size (int): Размер батча для задач оценки точности.
                            Больший размер = быстрее, но больше памяти.
                            Рекомендуется: 1-2 (экономия памяти), 4-8 (оптимально), 16+ (быстро)
            num_samples (int): Количество образцов для измерения скорости генерации.
                             Больше образцов = точнее измерение, но дольше выполнение.
                             Рекомендуется: 10-50 для быстрой оценки, 100+ для точной.
            save_results (bool): Сохранять ли результаты в JSON файл.
                               True = сохранить, False = только в памяти
        
        Returns:
            dict: Словарь с полными результатами оценки:
                - model_name (str): Название модели
                - load_time (float): Время загрузки (всегда 0.0, модель предзагружена)
                - evaluation_time (float): Время оценки точности в секундах
                - generation_speed (float): Средняя скорость генерации в токенах/сек
                - total_tokens_generated (int): Общее количество сгенерированных токенов
                - generation_time (float): Общее время генерации в секундах
                - total_time (float): Общее время выполнения (evaluation_time + generation_time)
                - system_metrics (dict): Системные метрики (начальные и финальные)
                - results_file (str, optional): Путь к сохраненному файлу (если save_results=True)
                - lm_eval_results (dict): Результаты тестов точности (если tasks не пустой)
                - generation_speed_detailed (dict): Детальная статистика скорости
        
        Raises:
            Exception: При ошибке во время оценки (логируется и перебрасывается)
        """
        try:
            self.logger.info("Запуск расширенной оценки модели")
            
            # 1. Логируем начальные системные ресурсы
            initial_system_metrics = self.log_system_resources("(начало)")
            
            # 2. Оценка точности на указанных задачах (если tasks не пустой)
            eval_time = 0.0
            results = {}
            if tasks:
                results, eval_time = self.evaluate_model(tasks, batch_size)
            
            # 3. Замер скорости генерации на тестовых промптах
            speed_metrics = self.measure_generation_speed(num_samples)
            
            # 4. Финальные системные метрики для сравнения
            final_system_metrics = self.log_system_resources("(окончание)")
            
            # 5. Сбор всех системных метрик в единую структуру
            system_metrics = {
                "initial": initial_system_metrics,                           # Начальные системные метрики
                "final": final_system_metrics,                               # Финальные системные метрики
                "model_load_time": 0.0,                                      # Время загрузки модели
                "evaluation_time": eval_time,                                # Время оценки точности
                "generation_time": speed_metrics["total_time"]               # Время генерации
            }
            
            # 6. Сохранение результатов в файл (опционально)
            result_file = None
            if save_results:
                result_file = self.save_results(results, eval_time, speed_metrics, system_metrics)
            
            # 7. Подготовка полных результатов для возврата
            evaluation_summary = {
                "model_name": self.model_name,                                    # Название модели
                "load_time": 0.0,                                                # Время загрузки (модель предзагружена)
                "evaluation_time": eval_time,                                    # Время оценки точности
                "generation_speed": speed_metrics['average_tokens_per_second'],  # Средняя скорость генерации
                "total_tokens_generated": speed_metrics['total_tokens'],         # Общее количество токенов
                "generation_time": speed_metrics["total_time"],                  # Время генерации
                "total_time": eval_time + speed_metrics["total_time"],           # Общее время выполнения
                "system_metrics": system_metrics,                                # Системные метрики
                "results_file": result_file,                                     # Путь к сохраненному файлу
                "lm_eval_results": results,                                      # Результаты тестов точности
                "generation_speed_detailed": speed_metrics                       # Детальная статистика скорости
            }
            
            # 8. Вывод полных метрик в консоль
            self._print_full_summary(evaluation_summary)
            
            return evaluation_summary
            
        except Exception as e:
            self.logger.error(f"Ошибка при расширенной оценке: {str(e)}")
            self.logger.exception("Подробности ошибки:")
            raise
    
    def _print_basic_summary(self, summary):
        """
        Вывод базовой сводки результатов в консоль.
        
        Форматирует и выводит основные метрики базовой оценки:
        - Информация о модели
        - Временные метрики
        - Системные ресурсы
        - Производительность
        - Детальная статистика по промптам
        
        Args:
            summary (dict): Словарь с базовыми результатами оценки
        """
        print("\n" + "="*60)
        print("БАЗОВЫЕ РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("="*60)
        print(f"Модель: {summary['model_name']}")
        print(f"Тип оценки: Базовая (только производительность)")
        
        # Временные метрики
        print(f"\nВРЕМЕННЫЕ МЕТРИКИ:")
        print(f"  Время загрузки: {summary['load_time']:.2f} сек (модель предзагружена)")
        print(f"  Время оценки: {summary['evaluation_time']:.2f} сек (нет тестов точности)")
        print(f"  Время генерации: {summary['generation_time']:.2f} сек")
        print(f"  Общее время: {summary['total_time']:.2f} сек")
        
        # Системные ресурсы (финальные значения)
        final_mem = summary['system_metrics']["final"]["memory"]
        final_cpu = summary['system_metrics']["final"]["cpu"]
        final_gpu = summary['system_metrics']["final"]["gpu"]
        
        print(f"\nСИСТЕМНЫЕ РЕСУРСЫ:")
        print(f"  Использование RAM: {final_mem['used_gb']:.1f}/{final_mem['total_gb']:.1f} GB ({final_mem['percent']:.1f}%)")
        print(f"  Использование CPU: {final_cpu['cpu_avg_percent']:.1f}% (ядер: {final_cpu['cpu_count_logical']})")
        if "error" not in final_gpu:
            print(f"  Использование GPU VRAM: {final_gpu['memory_used_gb']:.1f}/{final_gpu['memory_total_gb']:.1f} GB")
            print(f"  Загрузка GPU: {final_gpu['utilization_percent']:.1f}%")
        else:
            print(f"  GPU: {final_gpu['error']}")
        
        # Производительность
        print(f"\nПРОИЗВОДИТЕЛЬНОСТЬ:")
        print(f"  Скорость генерации: {summary['generation_speed']:.2f} токенов/сек")
        print(f"  Обработано токенов: {summary['total_tokens_generated']}")
        
        # Детальная статистика по промптам
        detailed_stats = summary.get('generation_speed_detailed', {}).get('detailed_stats', [])
        if detailed_stats:
            print(f"\nДЕТАЛЬНАЯ СТАТИСТИКА ПО ПРОМПТАМ:")
            
            # Вычисляем среднее время ответа
            response_times = [stat['generation_time'] for stat in detailed_stats]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            print(f"  Среднее время ответа: {avg_response_time:.2f} сек")
            print(f"  Время ответа по промптам:")
            
            for stat in detailed_stats:
                prompt_num = stat['prompt_number']
                gen_time = stat['generation_time']
                tokens = stat['generated_tokens']
                speed = stat['tokens_per_second']
                print(f"    Промпт {prompt_num}: {gen_time:.2f} сек ({tokens} токенов, {speed:.1f} токенов/сек)")
        
        if summary['results_file']:
            print(f"\nРезультаты сохранены в: {summary['results_file']}")
        print("="*60)

    def _print_full_summary(self, summary):
        """
        Вывод полной сводки результатов в консоль.
        
        Форматирует и выводит все метрики расширенной оценки:
        - Информация о модели
        - Временные метрики
        - Системные ресурсы
        - Точность по задачам (если есть)
        - Производительность
        
        Args:
            summary (dict): Словарь с полными результатами оценки
        """
        print("\n" + "="*60)
        print("ПОЛНЫЕ РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("="*60)
        print(f"Модель: {summary['model_name']}")
        print(f"Тип оценки: Расширенная (включая тесты точности)")
        
        # Временные метрики
        print(f"\nВРЕМЕННЫЕ МЕТРИКИ:")
        print(f"  Время загрузки: {summary['load_time']:.2f} сек (модель предзагружена)")
        print(f"  Время оценки: {summary['evaluation_time']:.2f} сек")
        print(f"  Время генерации: {summary['generation_time']:.2f} сек")
        print(f"  Общее время: {summary['total_time']:.2f} сек")
        
        # Системные ресурсы (финальные значения)
        final_mem = summary['system_metrics']["final"]["memory"]
        final_cpu = summary['system_metrics']["final"]["cpu"]
        final_gpu = summary['system_metrics']["final"]["gpu"]
        
        print(f"\nСИСТЕМНЫЕ РЕСУРСЫ:")
        print(f"  Использование RAM: {final_mem['used_gb']:.1f}/{final_mem['total_gb']:.1f} GB ({final_mem['percent']:.1f}%)")
        print(f"  Использование CPU: {final_cpu['cpu_avg_percent']:.1f}% (ядер: {final_cpu['cpu_count_logical']})")
        if "error" not in final_gpu:
            print(f"  Использование GPU VRAM: {final_gpu['memory_used_gb']:.1f}/{final_gpu['memory_total_gb']:.1f} GB")
            print(f"  Загрузка GPU: {final_gpu['utilization_percent']:.1f}%")
        else:
            print(f"  GPU: {final_gpu['error']}")
        
        # Точность по задачам (если есть результаты оценки)
        results = summary['lm_eval_results']
        if results and 'results' in results:
            print(f"\nТОЧНОСТЬ ПО ЗАДАЧАМ:")
            for task, metrics in results['results'].items():
                if 'acc,none' in metrics:
                    acc = metrics['acc,none']
                    stderr = metrics.get('acc_stderr,none', 0)
                    print(f"  {task}: {acc:.4f} ± {stderr:.4f}")
                elif 'exact_match,none' in metrics:
                    em = metrics['exact_match,none']
                    stderr = metrics.get('exact_match_stderr,none', 0)
                    print(f"  {task}: {em:.4f} ± {stderr:.4f}")
        
        # Производительность
        print(f"\nПРОИЗВОДИТЕЛЬНОСТЬ:")
        print(f"  Скорость генерации: {summary['generation_speed']:.2f} токенов/сек")
        print(f"  Обработано токенов: {summary['total_tokens_generated']}")
        
        if summary['results_file']:
            print(f"\nРезультаты сохранены в: {summary['results_file']}")
        print("="*60)

    def _print_summary(self, summary):
        """
        Вывод сводки результатов (устаревший метод).
        
        Перенаправляет на _print_full_summary для обратной совместимости.
        Рекомендуется использовать _print_basic_summary или _print_full_summary напрямую.
        
        Args:
            summary (dict): Словарь с результатами оценки
        """
        self._print_full_summary(summary)

# Функция для базовой оценки модели
def evaluate_basic_model(model, tokenizer, model_name=None, num_samples=10, save_results=True):
    """
    Базовая оценка модели - только системные и производительные метрики.
    
    Удобная функция-обертка для быстрой оценки модели без тестов точности.
    Создает экземпляр ModelEvaluator и запускает базовую оценку.
    
    Преимущества:
    - Минимальный код для быстрой оценки
    - Только основные метрики производительности
    - Не требует дополнительных данных
    - Подходит для быстрой проверки модели
    
    Args:
        model: Предзагруженная модель (обязательно)
        tokenizer: Предзагруженный токенизатор (обязательно)
        model_name (str, optional): Название модели для логирования и сохранения результатов.
                                   Если None, используется "preloaded_model"
        num_samples (int): Количество образцов для измерения скорости генерации.
                          Больше образцов = точнее измерение, но дольше выполнение.
                          По умолчанию: 10 (быстрая оценка)
        save_results (bool): Сохранять ли результаты в JSON файл.
                           По умолчанию: True (сохранить)
        
    Returns:
        dict: Словарь с базовыми результатами оценки (см. run_basic_evaluation)
    
    Raises:
        ValueError: Если model или tokenizer равны None
        Exception: При ошибке во время оценки
    
    Example:
        # Быстрая оценка производительности модели
        results = evaluate_basic_model(
            model=my_model,
            tokenizer=my_tokenizer,
            model_name="MyModel",
            num_samples=20
        )
        print(f"Скорость генерации: {results['generation_speed']:.2f} токенов/сек")
    """
    evaluator_obj = ModelEvaluator(model=model, tokenizer=tokenizer, model_name=model_name)
    return evaluator_obj.run_basic_evaluation(num_samples, save_results)

# Функция для расширенной оценки модели
def evaluate_full_model(model, tokenizer, model_name=None, tasks=["hellaswag", "mmlu", "gsm8k"], 
                       batch_size=8, num_samples=10, save_results=True):
    """
    Расширенная оценка модели - включает тесты точности и производительности.
    
    Удобная функция-обертка для полной оценки модели с тестами точности.
    Создает экземпляр ModelEvaluator и запускает расширенную оценку.
    
    Преимущества:
    - Полная оценка качества модели
    - Включает как точность, так и производительность
    - Профессиональная оценка для исследований
    - Автоматическое сохранение результатов
    
    Args:
        model: Предзагруженная модель (обязательно)
        tokenizer: Предзагруженный токенизатор (обязательно)
        model_name (str, optional): Название модели для логирования и сохранения результатов.
                                   Если None, используется "preloaded_model"
        tasks (list): Список задач для оценки точности. Доступные задачи:
            - "hellaswag": здравый смысл и логика
            - "mmlu": многозадачное понимание языка
            - "gsm8k": математические задачи
            - "arc_easy": рассуждения (легкий уровень)
            - "arc_challenge": рассуждения (сложный уровень)
            - "truthfulqa": правдивость ответов
            - "winogrande": разрешение местоимений
            - "piqa": физический здравый смысл
            По умолчанию: ["hellaswag", "mmlu", "gsm8k"] (основные задачи)
        batch_size (int): Размер батча для задач оценки точности.
                        Больший размер = быстрее, но больше памяти.
                        По умолчанию: 8 (оптимальный баланс)
        num_samples (int): Количество образцов для измерения скорости генерации.
                          Больше образцов = точнее измерение, но дольше выполнение.
                          По умолчанию: 10 (быстрая оценка)
        save_results (bool): Сохранять ли результаты в JSON файл.
                           По умолчанию: True (сохранить)
        
    Returns:
        dict: Словарь с полными результатами оценки (см. run_full_evaluation)
    
    Raises:
        ValueError: Если model или tokenizer равны None
        Exception: При ошибке во время оценки
    
    Example:
        # Полная оценка модели с тестами точности
        results = evaluate_full_model(
            model=my_model,
            tokenizer=my_tokenizer,
            model_name="MyModel",
            tasks=["hellaswag", "gsm8k"],
            batch_size=4,
            num_samples=20
        )
        print(f"Точность hellaswag: {results['lm_eval_results']['results']['hellaswag']['acc,none']:.4f}")
        print(f"Скорость генерации: {results['generation_speed']:.2f} токенов/сек")
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
# Добавляем путь к модулю main.py в Python path
import sys
sys.path.append('./src')  # Добавляем путь к модулю main.py

# Импортируем необходимые библиотеки
from transformers import AutoTokenizer, AutoModelForCausalLM  # Для загрузки моделей
from main import evaluate_basic_model, evaluate_full_model, ModelEvaluator, ensure_logging_setup  # Наши функции оценки

# 1.2 Убеждаемся, что логирование настроено (важно для Jupyter Notebook)
ensure_logging_setup()  # Принудительно настраиваем логирование

# 1.3 Загрузка модели и токенизатора
# Укажите путь к вашей модели (локальный путь или название с Hugging Face)
model_name = "./Текстовые/Qwen3-0.6B"  # Укажите путь к вашей модели

# Токенизатор - преобразует текст в токены для модели
# Токены - это числовые представления слов/частей слов
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Модель - основная языковая модель для генерации текста
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Автоматический выбор типа данных (float16/float32)
    device_map="auto"    # Автоматическое размещение на CPU/GPU
)

===============================================================================
ЭТАП 2: ОСНОВНЫЕ СПОСОБЫ ОЦЕНКИ
===============================================================================

СПОСОБ 1: БАЗОВАЯ ОЦЕНКА (рекомендуется для быстрой проверки)
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Быстрая оценка только системных и производительных метрик
ПРЕИМУЩЕСТВА: 
- Минимум кода
- Быстрое выполнение (не требует дополнительных данных)
- Только основные метрики производительности
- Не требует дополнительных данных

КОГДА ИСПОЛЬЗОВАТЬ:
✓ Быстрая проверка производительности модели
✓ Сравнение скорости разных моделей
✓ Отладка и тестирование системы
✓ Когда точность не важна, важна только скорость

# Простой вызов базовой оценки
results = evaluate_basic_model(
    model=model,                    # Предзагруженная модель (ОБЯЗАТЕЛЬНО)
    tokenizer=tokenizer,            # Предзагруженный токенизатор (ОБЯЗАТЕЛЬНО)
    model_name="Qwen3-0.6B",        # Название для логирования и сохранения (опционально)
    num_samples=10,                 # Образцы для измерения скорости генерации
    save_results=True               # Сохранить результаты в JSON файл
)

СПОСОБ 2: РАСШИРЕННАЯ ОЦЕНКА (включает тесты точности)
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Полная оценка модели с тестами точности на стандартных задачах
ПРЕИМУЩЕСТВА: 
- Полная оценка качества модели
- Включает как точность, так и производительность
- Профессиональная оценка для исследований
- Автоматическое сохранение результатов

КОГДА ИСПОЛЬЗОВАТЬ:
✓ Финальная оценка модели для продакшена
✓ Исследовательские цели и публикации
✓ Сравнение качества разных моделей
✓ Когда важна как точность, так и производительность

# Полная оценка с тестами точности
results = evaluate_full_model(
    model=model,                    # Предзагруженная модель (ОБЯЗАТЕЛЬНО)
    tokenizer=tokenizer,            # Предзагруженный токенизатор (ОБЯЗАТЕЛЬНО)
    model_name="Qwen3-0.6B",        # Название для логирования (опционально)
    tasks=["hellaswag", "gsm8k"],   # Задачи для оценки точности
    batch_size=4,                   # Размер батча (влияет на память/скорость)
    num_samples=10,                 # Образцы для измерения скорости генерации
    save_results=True               # Сохранить результаты в JSON файл
)

СПОСОБ 3: ДЕТАЛЬНЫЙ КОНТРОЛЬ (для продвинутых пользователей)
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Поэтапная оценка с полным контролем процесса
ПРЕИМУЩЕСТВА:
- Контроль каждого этапа оценки
- Доступ к промежуточным результатам
- Гибкая настройка параметров
- Возможность выполнения только части оценки

КОГДА ИСПОЛЬЗОВАТЬ:
✓ Отладка и анализ производительности
✓ Выполнение только части оценки
✓ Интеграция в сложные пайплайны
✓ Когда нужен детальный контроль

# 3.1 Создание оценщика с настройками
evaluator = ModelEvaluator(
    model=model,                    # Предзагруженная модель
    tokenizer=tokenizer,            # Предзагруженный токенизатор
    model_name="Qwen3-0.6B"         # Название для логирования
)

# 3.2 Этап A: Базовая оценка производительности
# НАЗНАЧЕНИЕ: Только системные и производительные метрики
# ПОЛЕЗНО: Быстрая проверка, сравнение производительности
basic_results = evaluator.run_basic_evaluation(
    num_samples=5,                  # Образцы для измерения скорости
    save_results=False              # Не сохранять в файл (только в памяти)
)

# 3.3 Этап B: Измерение скорости генерации отдельно
# НАЗНАЧЕНИЕ: Определяет токенов/сек для модели
# ПОЛЕЗНО: Сравнение производительности, оптимизация
speed_metrics = evaluator.measure_generation_speed(num_samples=5)

# 3.4 Этап C: Расширенная оценка с настройками
# НАЗНАЧЕНИЕ: Все этапы оценки с контролем параметров
# ПОЛЕЗНО: Полная оценка с кастомными настройками
full_results = evaluator.run_full_evaluation(
    tasks=["hellaswag"],            # Задачи для оценки точности
    batch_size=2,                   # Размер батча (экономия памяти)
    num_samples=5,                  # Образцы для измерения скорости
    save_results=False              # Не сохранять в файл
)

===============================================================================
ЭТАП 3: АНАЛИЗ РЕЗУЛЬТАТОВ
===============================================================================

# 3.1 Основные метрики производительности
# Скорость генерации - ключевая метрика производительности
print(f"Скорость генерации: {results['generation_speed']:.2f} токенов/сек")
print(f"Время оценки: {results['evaluation_time']:.2f} секунд")

# 3.2 Анализ системных ресурсов
# Мониторинг использования ресурсов системы
system_metrics = results['system_metrics']
print(f"Использование RAM: {system_metrics['final']['memory']['used_gb']:.1f} GB")
print(f"Использование GPU: {system_metrics['final']['gpu']['memory_used_gb']:.1f} GB")

# 3.3 Анализ точности по задачам (только для расширенной оценки)
# Результаты тестов точности на различных задачах
if results['lm_eval_results'] and 'results' in results['lm_eval_results']:
    for task, metrics in results['lm_eval_results']['results'].items():
        if 'acc,none' in metrics:
            print(f"Точность {task}: {metrics['acc,none']:.4f}")

===============================================================================
ЭТАП 4: СПЕЦИАЛЬНЫЕ СЦЕНАРИИ
===============================================================================

СЦЕНАРИЙ 1: БАЗОВАЯ ОЦЕНКА (рекомендуется для большинства случаев)
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Быстрая оценка системных и производительных метрик
КОГДА ИСПОЛЬЗОВАТЬ:
✓ Быстрая проверка модели после загрузки
✓ Сравнение производительности разных моделей
✓ Отладка и тестирование системы
✓ Когда точность не критична, важна только скорость

# Быстрая оценка производительности
basic_results = evaluate_basic_model(
    model=model,
    tokenizer=tokenizer,
    model_name="Qwen3-0.6B",        # Название модели для логирования
    num_samples=20,                 # Образцы для точного измерения скорости
    save_results=True               # Сохранить результаты в файл
)

СЦЕНАРИЙ 2: РАСШИРЕННАЯ ОЦЕНКА (включает тесты точности)
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Полная оценка модели с тестами точности на стандартных задачах
КОГДА ИСПОЛЬЗОВАТЬ:
✓ Финальная оценка модели для продакшена
✓ Исследовательские цели и публикации
✓ Сравнение качества разных моделей
✓ Когда важна как точность, так и производительность

# Полная оценка с тестами точности
full_results = evaluate_full_model(
    model=model,
    tokenizer=tokenizer,
    model_name="Qwen3-0.6B",        # Название модели
    tasks=["hellaswag", "mmlu", "gsm8k"],  # Все важные задачи для оценки
    batch_size=8,                          # Оптимальный размер батча
    num_samples=20,                        # Образцы для точного измерения скорости
    save_results=True                      # Сохранить результаты
)

СЦЕНАРИЙ 3: СРАВНЕНИЕ НЕСКОЛЬКИХ МОДЕЛЕЙ
------------------------------------------------------------------------------
НАЗНАЧЕНИЕ: Систематическое сравнение производительности и качества моделей
КОГДА ИСПОЛЬЗОВАТЬ:
✓ Выбор лучшей модели для конкретной задачи
✓ Исследование влияния размера модели на производительность
✓ Документирование экспериментов
✓ Сравнение разных архитектур моделей

# ПРИМЕР ИСПОЛЬЗОВАНИЯ СРАВНЕНИЯ МОДЕЛЕЙ:
# Конфигурация моделей для сравнения
# model_configs = [
#     {'name': 'Qwen3-0.6B', 'path': './Текстовые/Qwen3-0.6B'},
#     {'name': 'Qwen3-1.5B', 'path': './Текстовые/Qwen3-1.5B'}
# ]
# 
# # Базовое сравнение (быстрое, только производительность)
# basic_comparison = compare_models_basic(model_configs)
# 
# # Полное сравнение (медленное, но точное, включая точность)
# full_comparison = compare_models_full([
#     {'name': 'Qwen3-0.6B', 'path': './Текстовые/Qwen3-0.6B', 'tasks': ['hellaswag']},
#     {'name': 'Qwen3-1.5B', 'path': './Текстовые/Qwen3-1.5B', 'tasks': ['hellaswag']}
# ])

===============================================================================
СПРАВОЧНИК ПАРАМЕТРОВ
===============================================================================

evaluate_basic_model() - Базовая оценка:
- model: Предзагруженная модель (ОБЯЗАТЕЛЬНО)
- tokenizer: Предзагруженный токенизатор (ОБЯЗАТЕЛЬНО)
- model_name: Название модели (опционально)
- num_samples: Образцы для измерения скорости (по умолчанию: 10)
- save_results: Сохранять результаты (по умолчанию: True)

evaluate_full_model() - Расширенная оценка:
- model: Предзагруженная модель (ОБЯЗАТЕЛЬНО)
- tokenizer: Предзагруженный токенизатор (ОБЯЗАТЕЛЬНО)
- model_name: Название модели (опционально)
- tasks: Список задач для оценки точности (по умолчанию: ["hellaswag", "mmlu", "gsm8k"])
- batch_size: Размер батча (по умолчанию: 8)
- num_samples: Образцы для измерения скорости (по умолчанию: 10)
- save_results: Сохранять результаты (по умолчанию: True)

Доступные задачи (только для расширенной оценки):
- "hellaswag"     - здравый смысл и логика
- "mmlu"          - многозадачное понимание языка  
- "gsm8k"         - математические задачи
- "arc_easy"      - рассуждения (легкий уровень)
- "arc_challenge" - рассуждения (сложный уровень)
- "truthfulqa"    - правдивость ответов
- "winogrande"    - разрешение местоимений
- "piqa"          - физический здравый смысл

Рекомендации по параметрам:
- batch_size: 1-2 (экономия памяти), 4-8 (оптимально), 16+ (быстро)
- num_samples: 5-10 (быстро), 20-50 (точно), 100+ (очень точно)

===============================================================================
БАЗОВЫЕ МЕТРИКИ (всегда измеряются)
===============================================================================

1. Время загрузки: 0.0 сек (модель предзагружена)
2. Время оценки: 0.0 сек (базовая оценка) / N сек (расширенная оценка)
3. Использование RAM: текущее/общее GB (процент)
4. Использование CPU: средний процент (количество ядер)
5. Использование GPU VRAM: текущее/общее GB
6. Загрузка GPU: процент использования
7. Скорость генерации: токенов/сек
8. Обработано токенов: общее количество
9. Время генерации: секунды
10. Общее время: сумма всех времен
"""

# Функции для сравнения моделей (вынесены из документации для корректного синтаксиса)

def compare_models_basic(model_configs):
    """
    Базовое сравнение нескольких моделей (только производительность).
    
    Функция для систематического сравнения производительности моделей.
    Загружает каждую модель, выполняет базовую оценку и возвращает результаты.
    
    Args:
        model_configs: Список конфигураций моделей
        [{'name': 'model1', 'path': './path1'}, ...]
    
    Returns:
        dict: Результаты сравнения по моделям
    """
    results = {}
    
    for config in model_configs:
        print(f"Базовая оценка модели: {config['name']}")
        
        # Загрузка модели и токенизатора
        tokenizer = AutoTokenizer.from_pretrained(config['path'])
        model = AutoModelForCausalLM.from_pretrained(
            config['path'], torch_dtype="auto", device_map="auto"
        )
        
        # Базовая оценка производительности
        model_results = evaluate_basic_model(
            model=model,
            tokenizer=tokenizer,
            model_name=config['name'],
            num_samples=config.get('num_samples', 20),  # Количество образцов
            save_results=True                           # Сохранить результаты
        )
        
        results[config['name']] = model_results
        
        # Очистка памяти для следующей модели
        del model, tokenizer
        torch.cuda.empty_cache()  # Очистка GPU памяти
    
    return results

def compare_models_full(model_configs):
    """
    Полное сравнение нескольких моделей (включая точность).
    
    Функция для полного сравнения моделей с тестами точности.
    Более медленная, но дает полную картину качества моделей.
    
    Args:
        model_configs: Список конфигураций моделей
        [{'name': 'model1', 'path': './path1', 'tasks': ['hellaswag']}, ...]
    
    Returns:
        dict: Результаты сравнения по моделям
    """
    results = {}
    
    for config in model_configs:
        print(f"Полная оценка модели: {config['name']}")
        
        # Загрузка модели и токенизатора
        tokenizer = AutoTokenizer.from_pretrained(config['path'])
        model = AutoModelForCausalLM.from_pretrained(
            config['path'], torch_dtype="auto", device_map="auto"
        )
        
        # Полная оценка с тестами точности
        model_results = evaluate_full_model(
            model=model,
            tokenizer=tokenizer,
            model_name=config['name'],
            tasks=config.get('tasks', ['hellaswag']),    # Задачи для оценки
            batch_size=config.get('batch_size', 4),      # Размер батча
            num_samples=config.get('num_samples', 10),   # Образцы для скорости
            save_results=True                            # Сохранить результаты
        )
        
        results[config['name']] = model_results
        
        # Очистка памяти для следующей модели
        del model, tokenizer
        torch.cuda.empty_cache()  # Очистка GPU памяти
    
    return results