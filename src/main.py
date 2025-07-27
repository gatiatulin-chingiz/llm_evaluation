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
def setup_logging(model_name=None):
    """Настройка логирования для модуля."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Создаем уникальное имя лог-файла для каждой модели
    if model_name:
        log_filename = f"{model_name.replace('/', '_')}_evaluation.log"
    else:
        log_filename = 'model_evaluation.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(results_dir, log_filename), encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class SystemMonitor:
    """
    Класс для мониторинга системных ресурсов (CPU, RAM, GPU).
    
    Предоставляет статические методы для получения информации о:
    - Загрузке процессора (по ядрам и общая)
    - Использовании оперативной памяти
    - Состоянии графического процессора
    
    Все методы возвращают словари с метриками в удобном для анализа формате.
    """
    
    @staticmethod
    def get_cpu_info():
        """
        Получение детальной информации о загрузке процессора.
        
        Измеряет загрузку каждого ядра процессора и вычисляет среднее значение.
        Использует интервал в 1 секунду для получения актуальных данных.
        
        Returns:
            dict: Словарь с информацией о CPU:
                - cpu_percent_per_core (list): Загрузка каждого ядра в процентах
                - cpu_avg_percent (float): Средняя загрузка всех ядер
                - cpu_count (int): Количество физических ядер
                - cpu_count_logical (int): Количество логических ядер
                
        Example:
            >>> cpu_info = SystemMonitor.get_cpu_info()
            >>> print(f"Средняя загрузка: {cpu_info['cpu_avg_percent']:.1f}%")
        """
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
        """
        Получение информации об использовании оперативной памяти.
        
        Измеряет общий объем, доступную, используемую память и процент использования.
        Все значения памяти возвращаются в гигабайтах для удобства.
        
        Returns:
            dict: Словарь с информацией о RAM:
                - total_gb (float): Общий объем памяти в ГБ
                - available_gb (float): Доступная память в ГБ
                - used_gb (float): Используемая память в ГБ
                - percent (float): Процент использования памяти
                
        Example:
            >>> mem_info = SystemMonitor.get_memory_info()
            >>> print(f"Использовано: {mem_info['used_gb']:.1f} ГБ из {mem_info['total_gb']:.1f} ГБ")
        """
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        }
    
    @staticmethod
    def get_gpu_info():
        """
        Получение информации о состоянии графического процессора.
        
        Измеряет память GPU, загрузку и температуру. Если GPU недоступен
        или произошла ошибка, возвращает словарь с информацией об ошибке.
        
        Returns:
            dict: Словарь с информацией о GPU:
                - name (str): Название GPU
                - memory_total_gb (float): Общий объем видеопамяти в ГБ
                - memory_used_gb (float): Используемая видеопамять в ГБ
                - memory_free_gb (float): Свободная видеопамять в ГБ
                - utilization_percent (float): Загрузка GPU в процентах
                - temperature (float): Температура GPU в градусах Цельсия
                
            Или словарь с ошибкой:
                - error (str): Описание ошибки
                
        Example:
            >>> gpu_info = SystemMonitor.get_gpu_info()
            >>> if 'error' not in gpu_info:
            ...     print(f"GPU: {gpu_info['name']}, Память: {gpu_info['memory_used_gb']:.1f}/{gpu_info['memory_total_gb']:.1f} ГБ")
        """
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
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
    """
    Основной класс для оценки языковых моделей.
    
    Предоставляет комплексные методы для оценки производительности и точности
    предобученных языковых моделей. Включает мониторинг системных ресурсов,
    измерение скорости генерации и оценку точности на стандартных тестах.
    
    Основные возможности:
    - Мониторинг системных ресурсов (CPU, RAM, GPU) до и после оценки
    - Измерение скорости генерации на 5 специализированных промптах
    - Оценка точности на стандартных задачах (hellaswag, mmlu, gsm8k)
    - Сохранение результатов в структурированном JSON формате
    - Детальная статистика по каждому промпту
    
    Attributes:
        model: Предобученная модель Hugging Face
        tokenizer: Токенизатор для модели
        model_name (str): Название модели для идентификации
        device (str): Устройство для вычислений ('cuda' или 'cpu')
        logger: Логгер для записи процесса оценки
        
    Example:
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> evaluator = ModelEvaluator(model, tokenizer, "GPT-2")
        >>> results = evaluator.run_basic_evaluation()
    """
    
    def __init__(self, model, tokenizer, model_name=None):
        """
        Инициализация оценщика модели.
        
        Создает экземпляр ModelEvaluator с предзагруженной моделью и токенизатором.
        Автоматически определяет доступное устройство (CUDA/CPU) и настраивает
        мониторинг системных ресурсов.
        
        Args:
            model: Предобученная модель Hugging Face (AutoModelForCausalLM или аналогичная)
            tokenizer: Токенизатор для модели (AutoTokenizer или аналогичный)
            model_name (str, optional): Название модели для идентификации в логах и файлах.
                                       Если не указано, используется "preloaded_model"
        
        Raises:
            ValueError: Если model или tokenizer равны None
            
        Example:
            >>> from transformers import AutoTokenizer, AutoModelForCausalLM
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> evaluator = ModelEvaluator(model, tokenizer, "GPT-2")
        """
        if model is None or tokenizer is None:
            raise ValueError("model и tokenizer должны быть переданы (не None)")
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name or "preloaded_model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_monitor = SystemMonitor()
        
        # Создаем уникальный логгер для этой модели
        self.logger = setup_logging(self.model_name)
        
        self.logger.info(f"Инициализирован оценщик для модели: {self.model_name}")
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Токенизатор: {type(self.tokenizer).__name__}")
        self.logger.info(f"Модель: {type(self.model).__name__}")
        
    def log_system_resources(self, phase=""):
        """
        Логирование текущего состояния системных ресурсов.
        
        Собирает информацию о загрузке CPU, использовании RAM и состоянии GPU,
        записывает её в лог и возвращает структурированные данные для анализа.
        
        Args:
            phase (str, optional): Описательная метка фазы (например, "(начало)", "(после генерации)").
                                 Используется для идентификации момента измерения в логах.
        
        Returns:
            dict: Словарь с информацией о системных ресурсах:
                - cpu (dict): Информация о процессоре (загрузка, количество ядер)
                - memory (dict): Информация об оперативной памяти (использование, общий объем)
                - gpu (dict): Информация о графическом процессоре (память, загрузка)
                
        Example:
            >>> evaluator = ModelEvaluator(model, tokenizer)
            >>> resources = evaluator.log_system_resources("(перед оценкой)")
            >>> print(f"CPU загрузка: {resources['cpu']['cpu_avg_percent']:.1f}%")
        """
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
        
        return {"cpu": cpu_info, "memory": memory_info, "gpu": gpu_info}
    
    def evaluate_model(self, tasks=["hellaswag", "mmlu", "gsm8k"], batch_size=8):
        """
        Оценка точности модели на стандартных задачах с помощью LM Evaluation Harness.
        
        Выполняет оценку модели на указанных задачах, используя библиотеку lm_eval.
        Измеряет время выполнения и логирует системные ресурсы до и после оценки.
        
        Args:
            tasks (list, optional): Список задач для оценки. По умолчанию:
                - "hellaswag": Тест на понимание контекста и здравый смысл
                - "mmlu": Massive Multitask Language Understanding
                - "gsm8k": Математические задачи
            batch_size (int, optional): Размер батча для обработки. По умолчанию 8.
        
        Returns:
            tuple: (results, eval_time)
                - results (dict): Результаты оценки с метриками точности по каждой задаче
                - eval_time (float): Время выполнения оценки в секундах
                
        Note:
            Требует установленной библиотеки lm_eval и соответствующих датасетов.
            Первый запуск может занять время на загрузку данных.
            
        Example:
            >>> evaluator = ModelEvaluator(model, tokenizer)
            >>> results, time = evaluator.evaluate_model(["hellaswag", "gsm8k"])
            >>> print(f"Hellaswag accuracy: {results['results']['hellaswag']['acc,none']:.3f}")
        """
        self.logger.info(f"Запуск оценки для задач: {tasks}")
        self.log_system_resources("(перед оценкой)")
        
        lm_obj = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size
        )
        
        start_time = time.time()
        results = evaluator.simple_evaluate(
            model=lm_obj,
            tasks=tasks,
            batch_size=batch_size,
            device=self.device,
            limit=None
        )
        
        eval_time = time.time() - start_time
        self.log_system_resources("(после оценки)")
        
        return results, eval_time
    
    def measure_generation_speed(self):
        """
        Измерение скорости генерации модели на специализированных промптах.
        
        Выполняет генерацию ответов на 5 предопределенных промптов, связанных с
        математикой и страхованием. Измеряет время генерации, количество токенов
        и вычисляет скорость генерации для каждого промпта и в среднем.
        
        Промпты включают:
        - Градиентный спуск (машинное обучение)
        - Вероятность (математика)
        - Убыточные полисы (страхование)
        - Франшиза в автостраховании (страхование)
        - Математическое ожидание (математика)
        
        Returns:
            dict: Словарь с результатами измерения скорости:
                - average_tokens_per_second (float): Средняя скорость генерации (токенов/сек)
                - total_tokens (int): Общее количество сгенерированных токенов
                - total_time (float): Общее время генерации в секундах
                - detailed_stats (list): Детальная статистика по каждому промпту
                - speed_measurements (list): Список скоростей для каждого промпта
                - prompts_and_responses (list): Промпты и ответы модели
                
        Note:
            Использует параметры генерации, оптимизированные для подавления
            "thinking" вывода и улучшения качества ответов.
            
        Example:
            >>> evaluator = ModelEvaluator(model, tokenizer)
            >>> speed_metrics = evaluator.measure_generation_speed()
            >>> print(f"Средняя скорость: {speed_metrics['average_tokens_per_second']:.2f} токенов/сек")
        """
        self.logger.info("Замер скорости генерации...")
        self.log_system_resources("(перед генерацией)")
        
        test_prompts = [
            "Объясни, что такое градиентный спуск и как он работает в машинном обучении.",
            "Вычисли вероятность того, что при двух бросках игральной кости оба раза выпадет число больше 3.",
            "Какие страховые полисы считаются убыточными для страховой компании?",
            "Как работает франшиза в автостраховании и как она влияет на стоимость полиса?",
            "Объясни понятие математического ожидания и покажи, как его вычислять."
        ]
        
        self.logger.info(f"Используется {len(test_prompts)} уникальных промптов для оценки")
        
        total_tokens = 0
        total_time = 0
        generation_stats = []
        speeds = []
        prompts_and_responses = []
        
        for i, prompt in enumerate(test_prompts):
            if i % 5 == 0:
                self.log_system_resources(f"(генерация {i+1}/5)")
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.3,  # Снижена температура для более детерминированных ответов
                    top_p=0.9,  # Добавлен top_p для лучшего контроля
                    top_k=50,   # Добавлен top_k для ограничения выбора токенов
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Увеличен штраф за повторения
                    no_repeat_ngram_size=5,  # Увеличен размер n-грамм
                    early_stopping=True,
                    use_cache=True,
                    return_dict_in_generate=False,
                    output_scores=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    eos_token_id=self.tokenizer.eos_token_id,  # Явно указан токен конца
                    bos_token_id=self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None
                )
            
            gen_time = time.time() - start_time
            model_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Убираем повторение промпта из ответа
            if model_response.startswith(prompt):
                model_response = model_response[len(prompt):].strip()
            
            # Убираем лишние маркеры и форматирование
            model_response = model_response.replace("### Ответ:", "").replace("###", "").strip()
            model_response = model_response.replace("**Предварительные условия:**", "").strip()
            model_response = model_response.replace("**Ответ должен быть в формате текста.**", "").strip()
            
            # Ограничиваем длину ответа, если он слишком длинный
            if len(model_response) > 2000:
                # Ищем естественную точку остановки (конец предложения)
                sentences = model_response.split('.')
                if len(sentences) > 3:
                    model_response = '. '.join(sentences[:3]) + '.'
                else:
                    model_response = model_response[:2000] + "..."
            
            new_tokens = outputs.shape[1] - inputs.shape[1]
            tokens_per_sec = new_tokens / gen_time
            
            total_tokens += new_tokens
            total_time += gen_time
            speeds.append(tokens_per_sec)
            
            prompts_and_responses.append({
                "prompt_number": i+1,
                "prompt": prompt,
                "response": model_response,
                "prompt_tokens": inputs.shape[1],
                "response_tokens": new_tokens
            })
            
            stat = {
                "prompt_number": i+1,
                "prompt_length": inputs.shape[1],
                "generated_tokens": new_tokens,
                "generation_time": gen_time,
                "tokens_per_second": tokens_per_sec
            }
            generation_stats.append(stat)
            
            self.logger.info(f"Промпт {i+1}: {new_tokens} токенов за {gen_time:.2f}с = {tokens_per_sec:.2f} токенов/сек")
        
        avg_tokens_per_sec = sum(speeds) / len(speeds) if speeds else 0
        self.logger.info(f"Средняя скорость (среднее): {avg_tokens_per_sec:.2f} токенов/сек")
        self.log_system_resources("(после генерации)")
        
        return {
            "avg_tokens_per_second": avg_tokens_per_sec,
            "sum_total_tokens": total_tokens,
            "sum_total_time": total_time,
            "detailed_stats": generation_stats,
            "speed_measurements": speeds,
            "prompts_and_responses": prompts_and_responses
        }
    
    def save_results(self, results, eval_time, speed_metrics, system_metrics, filename=None):
        """
        Сохранение полных результатов оценки в структурированный JSON файл.
        
        Создает JSON файл с результатами расширенной оценки, включая метрики точности,
        производительности и системные ресурсы. Файл сохраняется в папку 'results'
        с автоматически генерируемым именем на основе названия модели и даты.
        
        Args:
            results (dict): Результаты оценки точности от lm_eval
            eval_time (float): Время выполнения оценки в секундах
            speed_metrics (dict): Метрики скорости генерации
            system_metrics (dict): Системные метрики (начальные и финальные)
            filename (str, optional): Пользовательское имя файла. Если не указано,
                                    генерируется автоматически в формате:
                                    "{model_name}_evaluation_results_{YYYYMMDD}.json"
        
        Returns:
            str: Путь к сохраненному файлу
            
        Note:
            Создает папку 'results' автоматически, если она не существует.
            Извлекает только ключевые метрики точности (accuracy/exact_match)
            для упрощения структуры JSON.
            
        Example:
            >>> evaluator = ModelEvaluator(model, tokenizer, "MyModel")
            >>> filepath = evaluator.save_results(results, 120.5, speed_metrics, system_metrics)
            >>> print(f"Результаты сохранены в: {filepath}")
        """
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            self.logger.info(f"Создана папка {results_dir}")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{self.model_name.replace('/', '_')}_evaluation_results_{timestamp}.json"
        
        filepath = os.path.join(results_dir, filename)
        
        accuracy_results = {}
        if results and 'results' in results:
            for task, metrics in results['results'].items():
                if 'acc,none' in metrics:
                    accuracy_results[task] = {
                        'avg_accuracy': {
                            'value': round(metrics['acc,none'], 4),
                            'description': f'Средняя точность модели на задаче {task} (доля правильных ответов)'
                        },
                        'accuracy_stderr': {
                            'value': round(metrics.get('acc_stderr,none', 0), 4),
                            'description': f'Стандартная ошибка точности на задаче {task}'
                        }
                    }
                elif 'exact_match,none' in metrics:
                    accuracy_results[task] = {
                        'avg_exact_match': {
                            'value': round(metrics['exact_match,none'], 4),
                            'description': f'Средняя точность точного совпадения на задаче {task}'
                        },
                        'exact_match_stderr': {
                            'value': round(metrics.get('exact_match_stderr,none', 0), 4),
                            'description': f'Стандартная ошибка точного совпадения на задаче {task}'
                        }
                    }
        
        final_system = system_metrics.get('final', {})
        system_summary = {
            'ram_used_gb': {
                'value': round(final_system.get('memory', {}).get('used_gb', 0), 1),
                'description': 'Используемая оперативная память в гигабайтах'
            },
            'ram_total_gb': {
                'value': round(final_system.get('memory', {}).get('total_gb', 0), 1),
                'description': 'Общий объем оперативной памяти в гигабайтах'
            },
            'cpu_percent': {
                'value': round(final_system.get('cpu', {}).get('cpu_avg_percent', 0), 1),
                'description': 'Средняя загрузка процессора в процентах'
            },
            'gpu_vram_used_gb': {
                'value': round(final_system.get('gpu', {}).get('memory_used_gb', 0), 1) if 'error' not in final_system.get('gpu', {}) else 0,
                'description': 'Используемая видеопамять GPU в гигабайтах'
            },
            'gpu_vram_total_gb': {
                'value': round(final_system.get('gpu', {}).get('memory_total_gb', 0), 1) if 'error' not in final_system.get('gpu', {}) else 0,
                'description': 'Общий объем видеопамяти GPU в гигабайтах'
            }
        }
        
        # Добавляем реальную скорость для каждого промпта
        prompts_with_speed = []
        for i, prompt_data in enumerate(speed_metrics.get('prompts_and_responses', [])):
            prompt_speed = speed_metrics['speed_measurements'][i] if i < len(speed_metrics['speed_measurements']) else 0
            prompt_data_with_speed = {
                **prompt_data,
                "real_speed_tokens_per_sec": {
                    "value": round(prompt_speed, 2),
                    "description": f"Реальная скорость генерации для промпта {prompt_data['prompt_number']} в токенах в секунду"
                },
                "prompt_tokens": {
                    "value": prompt_data["prompt_tokens"],
                    "description": f"Количество токенов во входном промпте {prompt_data['prompt_number']}"
                },
                "response_tokens": {
                    "value": prompt_data["response_tokens"],
                    "description": f"Количество токенов в ответе модели на промпт {prompt_data['prompt_number']}"
                }
            }
            prompts_with_speed.append(prompt_data_with_speed)
        
        output_data = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "evaluation_summary": {
                "evaluation_time_seconds": {
                    "value": round(eval_time, 2),
                    "description": "Время выполнения оценки точности модели на стандартных задачах в секундах"
                },
                "avg_generation_speed_tokens_per_sec": {
                    "value": round(speed_metrics['avg_tokens_per_second'], 2),
                    "description": "Средняя скорость генерации текста в токенах в секунду по всем промптам"
                },
                "sum_total_tokens_generated": {
                    "value": speed_metrics['sum_total_tokens'],
                    "description": "Общее количество сгенерированных токенов по всем промптам"
                },
                "sum_total_generation_time_seconds": {
                    "value": round(speed_metrics['sum_total_time'], 2),
                    "description": "Общее время генерации текста по всем промптам в секундах"
                }
            },
            "accuracy_results": accuracy_results,
            "system_summary": system_summary,
            "prompts_and_responses": prompts_with_speed
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Результаты сохранены в {filepath}")
        return filepath

    def save_basic_results_to_txt(self, basic_metrics, filename=None):
        """
        Сохранение базовых результатов оценки в текстовый файл.
        
        Создает txt файл с результатами базовой оценки, содержащий точно такой же
        вывод, как в консоли. Файл сохраняется в папку 'results' с автоматически
        генерируемым именем.
        
        Args:
            basic_metrics (dict): Словарь с базовыми метриками оценки
            filename (str, optional): Пользовательское имя файла. Если не указано,
                                    генерируется автоматически в формате:
                                    "{model_name}_basic_evaluation_{YYYYMMDD}.txt"
        
        Returns:
            str: Путь к сохраненному файлу
            
        Note:
            Создает папку 'results' автоматически, если она не существует.
            Содержимое файла идентично выводу в консоли.
            
        Example:
            >>> evaluator = ModelEvaluator(model, tokenizer, "MyModel")
            >>> basic_metrics = evaluator.run_basic_evaluation(save_results=False)
            >>> filepath = evaluator.save_basic_results_to_txt(basic_metrics)
            >>> print(f"Базовые результаты сохранены в: {filepath}")
        """
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            self.logger.info(f"Создана папка {results_dir}")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{self.model_name.replace('/', '_')}_basic_evaluation_{timestamp}.txt"
        
        filepath = os.path.join(results_dir, filename)
        
        # Перенаправляем вывод в файл
        import io
        import sys
        
        # Сохраняем оригинальный stdout
        original_stdout = sys.stdout
        
        # Создаем StringIO для захвата вывода
        output_buffer = io.StringIO()
        sys.stdout = output_buffer
        
        # Вызываем метод вывода в консоль
        self._print_basic_summary(basic_metrics)
        
        # Получаем захваченный вывод
        captured_output = output_buffer.getvalue()
        
        # Восстанавливаем оригинальный stdout
        sys.stdout = original_stdout
        
        # Записываем в файл
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(captured_output)
        
        self.logger.info(f"Базовые результаты сохранены в {filepath}")
        return filepath

    def run_basic_evaluation(self, save_results=True):
        """
        Выполнение базовой оценки модели - только производительность.
        
        Проводит быструю оценку модели, измеряя скорость генерации на 5 специализированных
        промптах и собирая системные метрики. Не включает тесты точности, что делает
        оценку значительно быстрее полной версии.
        
        Args:
            save_results (bool, optional): Сохранять ли результаты в JSON файл.
                                         По умолчанию True.
        
        Returns:
            dict: Словарь с результатами базовой оценки:
                - model_name (str): Название модели
                - load_time (float): Время загрузки (0.0 для предзагруженных моделей)
                - evaluation_time (float): Время оценки (0.0 для базовой оценки)
                - generation_speed (float): Средняя скорость генерации в токенах/сек
                - total_tokens_generated (int): Общее количество сгенерированных токенов
                - generation_time (float): Общее время генерации в секундах
                - total_time (float): Общее время выполнения
                - system_metrics (dict): Системные метрики (начальные и финальные)
                - generation_speed_detailed (dict): Детальные метрики скорости
                - results_file (str, optional): Путь к сохраненному файлу (если save_results=True)
                
        Raises:
            Exception: При ошибках во время оценки (логируется детально)
            
        Note:
            Автоматически выводит сводку результатов в консоль.
            Если save_results=True, создает txt файл в папке 'results'.
            
        Example:
            >>> evaluator = ModelEvaluator(model, tokenizer, "GPT-2")
            >>> results = evaluator.run_basic_evaluation()
            >>> print(f"Скорость генерации: {results['generation_speed']:.2f} токенов/сек")
        """
        try:
            self.logger.info("Запуск базовой оценки модели")
            
            initial_system_metrics = self.log_system_resources("(начало)")
            speed_metrics = self.measure_generation_speed()
            final_system_metrics = self.log_system_resources("(окончание)")
            
            basic_metrics = {
                "model_name": self.model_name,
                "generation_speed": speed_metrics['avg_tokens_per_second'],
                "total_tokens_generated": speed_metrics['sum_total_tokens'],
                "generation_time": speed_metrics["sum_total_time"],
                "system_metrics": {
                    "initial": initial_system_metrics,
                    "final": final_system_metrics
                },
                "generation_speed_detailed": speed_metrics
            }
            
            result_file = None
            if save_results:
                result_file = self.save_basic_results_to_txt(basic_metrics)
                basic_metrics["results_file"] = result_file
            
            self._print_basic_summary(basic_metrics)
            return basic_metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка при базовой оценке: {str(e)}")
            self.logger.exception("Подробности ошибки:")
            raise

    def run_full_evaluation(self, tasks=["hellaswag", "mmlu", "gsm8k"], batch_size=8, save_results=True):
        """
        Выполнение полной оценки модели - включает тесты точности и производительности.
        
        Проводит комплексную оценку модели, включающую:
        1. Оценку точности на стандартных задачах (hellaswag, mmlu, gsm8k)
        2. Измерение скорости генерации на специализированных промптах
        3. Мониторинг системных ресурсов на всех этапах
        
        Args:
            tasks (list, optional): Список задач для оценки точности. По умолчанию:
                - "hellaswag": Тест на понимание контекста и здравый смысл
                - "mmlu": Massive Multitask Language Understanding
                - "gsm8k": Математические задачи
            batch_size (int, optional): Размер батча для оценки точности. По умолчанию 8.
            save_results (bool, optional): Сохранять ли результаты в JSON файл.
                                         По умолчанию True.
        
        Returns:
            dict: Словарь с результатами полной оценки:
                - model_name (str): Название модели
                - load_time (float): Время загрузки (0.0 для предзагруженных моделей)
                - evaluation_time (float): Время оценки точности в секундах
                - generation_speed (float): Средняя скорость генерации в токенах/сек
                - total_tokens_generated (int): Общее количество сгенерированных токенов
                - generation_time (float): Время генерации в секундах
                - total_time (float): Общее время выполнения (оценка + генерация)
                - system_metrics (dict): Системные метрики (начальные и финальные)
                - results_file (str, optional): Путь к сохраненному файлу
                - lm_eval_results (dict): Полные результаты lm_eval
                - generation_speed_detailed (dict): Детальные метрики скорости
                
        Raises:
            Exception: При ошибках во время оценки (логируется детально)
            
        Note:
            Автоматически выводит сводку результатов в консоль.
            Если save_results=True, создает txt файл в папке 'results'.
            Может занять значительное время в зависимости от количества задач.
            
        Example:
            >>> evaluator = ModelEvaluator(model, tokenizer, "GPT-2")
            >>> results = evaluator.run_full_evaluation(tasks=["hellaswag"])
            >>> print(f"Точность Hellaswag: {results['lm_eval_results']['results']['hellaswag']['acc,none']:.3f}")
        """
        try:
            self.logger.info("Запуск расширенной оценки модели")
            
            initial_system_metrics = self.log_system_resources("(начало)")
            
            eval_time = 0.0
            results = {}
            if tasks:
                results, eval_time = self.evaluate_model(tasks, batch_size)
            
            speed_metrics = self.measure_generation_speed()
            final_system_metrics = self.log_system_resources("(окончание)")
            
            system_metrics = {
                "initial": initial_system_metrics,
                "final": final_system_metrics,
                "model_load_time": 0.0,
                "evaluation_time": eval_time,
                "generation_time": speed_metrics["sum_total_time"]
            }
            
            result_file = None
            if save_results:
                result_file = self.save_results(results, eval_time, speed_metrics, system_metrics)
            
            evaluation_summary = {
                "model_name": self.model_name,
                "load_time": 0.0,
                "evaluation_time": eval_time,
                "generation_speed": speed_metrics['avg_tokens_per_second'],
                "total_tokens_generated": speed_metrics['sum_total_tokens'],
                "generation_time": speed_metrics["sum_total_time"],
                "total_time": eval_time + speed_metrics["sum_total_time"],
                "system_metrics": system_metrics,
                "results_file": result_file,
                "lm_eval_results": results,
                "generation_speed_detailed": speed_metrics
            }
            
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
        print(f"  Время генерации: {summary['generation_time']:.2f} сек")
        
        # Системные ресурсы (финальные значения)
        final_mem = summary['system_metrics']["final"]["memory"]
        final_cpu = summary['system_metrics']["final"]["cpu"]
        final_gpu = summary['system_metrics']["final"]["gpu"]
        
        print(f"\nСИСТЕМНЫЕ РЕСУРСЫ:")
        print(f"  Использование RAM: {final_mem['used_gb']:.1f}/{final_mem['total_gb']:.1f} GB")
        print(f"  Использование CPU: {final_cpu['cpu_avg_percent']:.1f}%")
        if "error" not in final_gpu:
            print(f"  Использование GPU VRAM: {final_gpu['memory_used_gb']:.1f}/{final_gpu['memory_total_gb']:.1f} GB")
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
def evaluate_basic_model(model, tokenizer, model_name=None, save_results=True):
    """
    Удобная функция для базовой оценки модели - только производительность.
    
    Создает экземпляр ModelEvaluator и выполняет базовую оценку модели,
    измеряя скорость генерации на специализированных промптах без тестов точности.
    
    Args:
        model: Предобученная модель Hugging Face (AutoModelForCausalLM или аналогичная)
        tokenizer: Токенизатор для модели (AutoTokenizer или аналогичный)
        model_name (str, optional): Название модели для идентификации.
                                   Если не указано, используется "preloaded_model"
        save_results (bool, optional): Сохранять ли результаты в JSON файл.
                                      По умолчанию True.
    
    Returns:
        dict: Словарь с результатами базовой оценки (см. ModelEvaluator.run_basic_evaluation)
        
    Note:
        Это удобная функция-обертка для быстрого запуска базовой оценки
        без необходимости создавать экземпляр ModelEvaluator вручную.
        
    Example:
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> results = evaluate_basic_model(model, tokenizer, "GPT-2")
        >>> print(f"Скорость генерации: {results['generation_speed']:.2f} токенов/сек")
    """
    evaluator_obj = ModelEvaluator(model=model, tokenizer=tokenizer, model_name=model_name)
    return evaluator_obj.run_basic_evaluation(save_results)

# Функция для расширенной оценки модели
def evaluate_full_model(model, tokenizer, model_name=None, tasks=["hellaswag", "mmlu", "gsm8k"], 
                       batch_size=8, save_results=True):
    """
    Удобная функция для полной оценки модели - включает тесты точности и производительности.
    
    Создает экземпляр ModelEvaluator и выполняет комплексную оценку модели,
    включающую тесты точности на стандартных задачах и измерение скорости генерации.
    
    Args:
        model: Предобученная модель Hugging Face (AutoModelForCausalLM или аналогичная)
        tokenizer: Токенизатор для модели (AutoTokenizer или аналогичный)
        model_name (str, optional): Название модели для идентификации.
                                   Если не указано, используется "preloaded_model"
        tasks (list, optional): Список задач для оценки точности. По умолчанию:
            - "hellaswag": Тест на понимание контекста и здравый смысл
            - "mmlu": Massive Multitask Language Understanding
            - "gsm8k": Математические задачи
        batch_size (int, optional): Размер батча для оценки точности. По умолчанию 8.
        save_results (bool, optional): Сохранять ли результаты в JSON файл.
                                      По умолчанию True.
    
    Returns:
        dict: Словарь с результатами полной оценки (см. ModelEvaluator.run_full_evaluation)
        
    Note:
        Это удобная функция-обертка для быстрого запуска полной оценки
        без необходимости создавать экземпляр ModelEvaluator вручную.
        Может занять значительное время в зависимости от количества задач.
        
    Example:
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> results = evaluate_full_model(model, tokenizer, "GPT-2", tasks=["hellaswag"])
        >>> print(f"Точность Hellaswag: {results['lm_eval_results']['results']['hellaswag']['acc,none']:.3f}")
    """
    evaluator_obj = ModelEvaluator(model=model, tokenizer=tokenizer, model_name=model_name)
    return evaluator_obj.run_full_evaluation(tasks, batch_size, save_results)

# Функции для сравнения моделей

def compare_models_basic(model_configs):
    """
    Базовое сравнение нескольких моделей - только производительность.
    
    Загружает и оценивает несколько моделей по производительности, используя
    базовую оценку (без тестов точности). Автоматически освобождает память
    после каждой модели для эффективного использования ресурсов.
    
    Args:
        model_configs (list): Список конфигураций моделей. Каждая конфигурация должна содержать:
            - name (str): Название модели для идентификации
            - path (str): Путь к модели (локальный или Hugging Face Hub)
            - num_samples (int, optional): Количество сэмплов (игнорируется, всегда 5)
            - tasks (list, optional): Список задач (игнорируется в базовой оценке)
            - batch_size (int, optional): Размер батча (игнорируется в базовой оценке)
    
    Returns:
        dict: Словарь с результатами базовой оценки для каждой модели:
            {model_name: results_dict, ...}
            
    Note:
        - Автоматически создает txt файлы для каждой модели в папке 'results'
        - Освобождает GPU память после каждой модели
        - Использует автоматическое определение типа данных и устройства
        - Выводит прогресс в консоль
        
    Example:
        >>> model_configs = [
        ...     {"name": "GPT-2", "path": "gpt2"},
        ...     {"name": "GPT-2 Medium", "path": "gpt2-medium"}
        ... ]
        >>> results = compare_models_basic(model_configs)
        >>> for name, result in results.items():
        ...     print(f"{name}: {result['generation_speed']:.2f} токенов/сек")
    """
    results = {}
    
    for config in model_configs:
        print(f"Базовая оценка модели: {config['name']}")
        
        tokenizer = AutoTokenizer.from_pretrained(config['path'])
        model = AutoModelForCausalLM.from_pretrained(
            config['path'], torch_dtype="auto", device_map="auto"
        )
        
        model_results = evaluate_basic_model(
            model=model,
            tokenizer=tokenizer,
            model_name=config['name'],
            save_results=True
        )
        
        results[config['name']] = model_results
        
        del model, tokenizer
        torch.cuda.empty_cache()
    
    return results

def compare_models_full(model_configs):
    """
    Полное сравнение нескольких моделей - включает тесты точности и производительности.
    
    Загружает и оценивает несколько моделей комплексно, включая тесты точности
    на стандартных задачах и измерение скорости генерации. Автоматически освобождает
    память после каждой модели для эффективного использования ресурсов.
    
    Args:
        model_configs (list): Список конфигураций моделей. Каждая конфигурация должна содержать:
            - name (str): Название модели для идентификации
            - path (str): Путь к модели (локальный или Hugging Face Hub)
            - tasks (list, optional): Список задач для оценки точности. По умолчанию ['hellaswag']
            - batch_size (int, optional): Размер батча для оценки точности. По умолчанию 4
            - num_samples (int, optional): Количество сэмплов (игнорируется, всегда 5)
    
    Returns:
        dict: Словарь с результатами полной оценки для каждой модели:
            {model_name: results_dict, ...}
            
    Note:
        - Автоматически создает txt файлы для каждой модели в папке 'results'
        - Освобождает GPU память после каждой модели
        - Использует автоматическое определение типа данных и устройства
        - Выводит прогресс в консоль
        - Может занять значительное время в зависимости от количества задач
        
    Example:
        >>> model_configs = [
        ...     {"name": "GPT-2", "path": "gpt2", "tasks": ["hellaswag"]},
        ...     {"name": "GPT-2 Medium", "path": "gpt2-medium", "tasks": ["hellaswag", "gsm8k"]}
        ... ]
        >>> results = compare_models_full(model_configs)
        >>> for name, result in results.items():
        ...     print(f"{name}: {result['generation_speed']:.2f} токенов/сек")
        ...     if 'hellaswag' in result['lm_eval_results']['results']:
        ...         acc = result['lm_eval_results']['results']['hellaswag']['acc,none']
        ...         print(f"  Hellaswag accuracy: {acc:.3f}")
    """
    results = {}
    
    for config in model_configs:
        print(f"Полная оценка модели: {config['name']}")
        
        tokenizer = AutoTokenizer.from_pretrained(config['path'])
        model = AutoModelForCausalLM.from_pretrained(
            config['path'], torch_dtype="auto", device_map="auto"
        )
        
        model_results = evaluate_full_model(
            model=model,
            tokenizer=tokenizer,
            model_name=config['name'],
            tasks=config.get('tasks', ['hellaswag']),
            batch_size=config.get('batch_size', 4),
            save_results=True
        )
        
        results[config['name']] = model_results
        
        del model, tokenizer
        torch.cuda.empty_cache()
    
    return results

# Пример использования:
"""
# Импорт
from transformers import AutoTokenizer, AutoModelForCausalLM
from main import evaluate_basic_model, evaluate_full_model

# Загрузка модели
model_name = "./path/to/model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Базовая оценка (только производительность)
results = evaluate_basic_model(model, tokenizer, "MyModel")

# Полная оценка (включая точность)
results = evaluate_full_model(model, tokenizer, "MyModel", tasks=["hellaswag", "gsm8k"])
"""