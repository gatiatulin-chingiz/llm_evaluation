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
    """Настройка логирования для модуля."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'model_evaluation.log'), encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class SystemMonitor:
    """Мониторинг системных ресурсов (CPU, RAM, GPU)."""
    
    @staticmethod
    def get_cpu_info():
        """Получение информации о CPU."""
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
        """Получение информации о RAM."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        }
    
    @staticmethod
    def get_gpu_info():
        """Получение информации о GPU."""
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
        """Инициализация оценщика модели."""
        if model is None or tokenizer is None:
            raise ValueError("model и tokenizer должны быть переданы (не None)")
        
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name or "preloaded_model"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_monitor = SystemMonitor()
        self.logger = logger
        
        self.logger.info(f"Инициализирован оценщик для модели: {self.model_name}")
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Токенизатор: {type(self.tokenizer).__name__}")
        self.logger.info(f"Модель: {type(self.model).__name__}")
        
    def log_system_resources(self, phase=""):
        """Логирование системных ресурсов."""
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
        """Оценка модели с помощью LM Evaluation Harness."""
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
    
    def measure_generation_speed(self, num_samples=10):
        """Замер скорости генерации модели (5 промптов)."""
        self.logger.info("Замер скорости генерации...")
        self.log_system_resources("(перед генерацией)")
        
        test_prompts = [
            "Опиши подробно, что такое градиентный спуск и как он работает.",
            "Игральную кость с 6 гранями бросают дважды. Найдите вероятность того, что оба раза выпало число, большее 3.",
            "Какие полисы в страховании можно считать убыточными?",
            "Объясни принцип работы франшизы в автостраховании и её влияние на стоимость полиса.",
            "Что такое математическое ожидание и как его вычислить?"
        ]
        
        test_prompts = test_prompts[:5]
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
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    use_cache=True,
                    return_dict_in_generate=False,
                    output_scores=False,
                    output_hidden_states=False,
                    output_attentions=False
                )
            
            gen_time = time.time() - start_time
            model_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
            "average_tokens_per_second": avg_tokens_per_sec,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "detailed_stats": generation_stats,
            "speed_measurements": speeds,
            "prompts_and_responses": prompts_and_responses
        }
    
    def save_results(self, results, eval_time, speed_metrics, system_metrics, filename=None):
        """Сохранение результатов оценки в JSON файл."""
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
                        'accuracy': round(metrics['acc,none'], 4),
                        'stderr': round(metrics.get('acc_stderr,none', 0), 4)
                    }
                elif 'exact_match,none' in metrics:
                    accuracy_results[task] = {
                        'exact_match': round(metrics['exact_match,none'], 4),
                        'stderr': round(metrics.get('exact_match_stderr,none', 0), 4)
                    }
        
        final_system = system_metrics.get('final', {})
        system_summary = {
            'ram_used_gb': round(final_system.get('memory', {}).get('used_gb', 0), 1),
            'ram_total_gb': round(final_system.get('memory', {}).get('total_gb', 0), 1),
            'cpu_percent': round(final_system.get('cpu', {}).get('cpu_avg_percent', 0), 1),
            'gpu_vram_used_gb': round(final_system.get('gpu', {}).get('memory_used_gb', 0), 1) if 'error' not in final_system.get('gpu', {}) else 0,
            'gpu_vram_total_gb': round(final_system.get('gpu', {}).get('memory_total_gb', 0), 1) if 'error' not in final_system.get('gpu', {}) else 0
        }
        
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
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Результаты сохранены в {filepath}")
        return filepath

    def save_basic_results(self, basic_metrics, filename=None):
        """Сохранение базовых результатов оценки в JSON файл."""
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            self.logger.info(f"Создана папка {results_dir}")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{self.model_name.replace('/', '_')}_basic_evaluation_{timestamp}.json"
        
        filepath = os.path.join(results_dir, filename)
        
        final_system = basic_metrics.get('system_metrics', {}).get('final', {})
        system_summary = {
            'ram_used_gb': round(final_system.get('memory', {}).get('used_gb', 0), 1),
            'ram_total_gb': round(final_system.get('memory', {}).get('total_gb', 0), 1),
            'cpu_percent': round(final_system.get('cpu', {}).get('cpu_avg_percent', 0), 1),
            'gpu_vram_used_gb': round(final_system.get('gpu', {}).get('memory_used_gb', 0), 1) if 'error' not in final_system.get('gpu', {}) else 0,
            'gpu_vram_total_gb': round(final_system.get('gpu', {}).get('memory_total_gb', 0), 1) if 'error' not in final_system.get('gpu', {}) else 0
        }
        
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
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Базовые результаты сохранены в {filepath}")
        return filepath

    def run_basic_evaluation(self, num_samples=10, save_results=True):
        """Базовая оценка модели - только производительность."""
        try:
            self.logger.info("Запуск базовой оценки модели")
            
            initial_system_metrics = self.log_system_resources("(начало)")
            speed_metrics = self.measure_generation_speed(num_samples)
            final_system_metrics = self.log_system_resources("(окончание)")
            
            basic_metrics = {
                "model_name": self.model_name,
                "load_time": 0.0,
                "evaluation_time": 0.0,
                "generation_speed": speed_metrics['average_tokens_per_second'],
                "total_tokens_generated": speed_metrics['total_tokens'],
                "generation_time": speed_metrics["total_time"],
                "total_time": speed_metrics["total_time"],
                "system_metrics": {
                    "initial": initial_system_metrics,
                    "final": final_system_metrics,
                    "model_load_time": 0.0,
                    "evaluation_time": 0.0,
                    "generation_time": speed_metrics["total_time"]
                },
                "generation_speed_detailed": speed_metrics
            }
            
            result_file = None
            if save_results:
                result_file = self.save_basic_results(basic_metrics)
                basic_metrics["results_file"] = result_file
            
            self._print_basic_summary(basic_metrics)
            return basic_metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка при базовой оценке: {str(e)}")
            self.logger.exception("Подробности ошибки:")
            raise

    def run_full_evaluation(self, tasks=["hellaswag", "mmlu", "gsm8k"], batch_size=8, num_samples=10, save_results=True):
        """Расширенная оценка модели - включает тесты точности."""
        try:
            self.logger.info("Запуск расширенной оценки модели")
            
            initial_system_metrics = self.log_system_resources("(начало)")
            
            eval_time = 0.0
            results = {}
            if tasks:
                results, eval_time = self.evaluate_model(tasks, batch_size)
            
            speed_metrics = self.measure_generation_speed(num_samples)
            final_system_metrics = self.log_system_resources("(окончание)")
            
            system_metrics = {
                "initial": initial_system_metrics,
                "final": final_system_metrics,
                "model_load_time": 0.0,
                "evaluation_time": eval_time,
                "generation_time": speed_metrics["total_time"]
            }
            
            result_file = None
            if save_results:
                result_file = self.save_results(results, eval_time, speed_metrics, system_metrics)
            
            evaluation_summary = {
                "model_name": self.model_name,
                "load_time": 0.0,
                "evaluation_time": eval_time,
                "generation_speed": speed_metrics['average_tokens_per_second'],
                "total_tokens_generated": speed_metrics['total_tokens'],
                "generation_time": speed_metrics["total_time"],
                "total_time": eval_time + speed_metrics["total_time"],
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
    """Базовая оценка модели - только производительность."""
    evaluator_obj = ModelEvaluator(model=model, tokenizer=tokenizer, model_name=model_name)
    return evaluator_obj.run_basic_evaluation(num_samples, save_results)

# Функция для расширенной оценки модели
def evaluate_full_model(model, tokenizer, model_name=None, tasks=["hellaswag", "mmlu", "gsm8k"], 
                       batch_size=8, num_samples=10, save_results=True):
    """Расширенная оценка модели - включает тесты точности."""
    evaluator_obj = ModelEvaluator(model=model, tokenizer=tokenizer, model_name=model_name)
    return evaluator_obj.run_full_evaluation(tasks, batch_size, num_samples, save_results)

# Функции для сравнения моделей

def compare_models_basic(model_configs):
    """Базовое сравнение нескольких моделей (только производительность)."""
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
            num_samples=config.get('num_samples', 20),
            save_results=True
        )
        
        results[config['name']] = model_results
        
        del model, tokenizer
        torch.cuda.empty_cache()
    
    return results

def compare_models_full(model_configs):
    """Полное сравнение нескольких моделей (включая точность)."""
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
            num_samples=config.get('num_samples', 10),
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