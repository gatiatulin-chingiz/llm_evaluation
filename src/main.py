#!/usr/bin/env python3
"""
Скрипт для оценки Qwen на Tesla V100 с логированием результатов и системных метрик
"""

import time
import json
import logging
import psutil
import GPUtil
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qwen_evaluation.log'),
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

class QwenEvaluator:
    def __init__(self, model_name="Qwen/Qwen1.5-7B-Chat"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.system_monitor = SystemMonitor()
        
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
    
    def load_model(self):
        """Загрузка модели с оптимизациями для V100"""
        start_time = time.time()
        
        self.logger.info(f"Загрузка модели {self.model_name}...")
        self.log_system_resources("(до загрузки модели)")
        
        # Оптимизации для Tesla V100 (32GB VRAM)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Используем float16 для экономии памяти
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        load_time = time.time() - start_time
        self.logger.info(f"Модель загружена за {load_time:.2f} секунд")
        self.log_system_resources("(после загрузки модели)")
        
        # Информация о памяти PyTorch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            self.logger.info(f"PyTorch GPU: {gpu_name}")
            self.logger.info(f"PyTorch выделено VRAM: {memory_allocated:.2f} GB")
            self.logger.info(f"PyTorch зарезервировано VRAM: {memory_reserved:.2f} GB")
        
        return load_time
    
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
            filename = f"qwen_evaluation_results_{timestamp}.json"
        
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

def main():
    # Параметры для Tesla V100 (32GB)
    MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
    BATCH_SIZE = 8  # Оптимальный размер батча для V100
    TASKS = ["hellaswag", "mmlu", "gsm8k", "arc_easy"]
    
    evaluator_obj = QwenEvaluator(MODEL_NAME)
    
    try:
        # Логируем начальные системные ресурсы
        initial_system_metrics = evaluator_obj.log_system_resources("(начало)")
        
        # 1. Загрузка модели
        load_time = evaluator_obj.load_model()
        
        # 2. Оценка точности
        results, eval_time = evaluator_obj.evaluate_model(TASKS, BATCH_SIZE)
        
        # 3. Замер скорости генерации
        speed_metrics = evaluator_obj.measure_generation_speed(num_samples=10)
        
        # 4. Финальные системные метрики
        final_system_metrics = evaluator_obj.log_system_resources("(окончание)")
        
        # 5. Сбор всех системных метрик
        system_metrics = {
            "initial": initial_system_metrics,
            "final": final_system_metrics,
            "model_load_time": load_time,
            "evaluation_time": eval_time,
            "generation_time": speed_metrics["total_time"]
        }
        
        # 6. Сохранение результатов
        result_file = evaluator_obj.save_results(results, eval_time, speed_metrics, system_metrics)
        
        # 7. Вывод основных метрик
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
        print("="*60)
        print(f"Модель: {MODEL_NAME}")
        print(f"Время загрузки: {load_time:.2f} сек")
        print(f"Время оценки: {eval_time:.2f} сек")
        
        # Системные ресурсы
        final_mem = final_system_metrics["memory"]
        final_gpu = final_system_metrics["gpu"]
        print(f"Использование RAM: {final_mem['used_gb']:.1f}/{final_mem['total_gb']:.1f} GB ({final_mem['percent']:.1f}%)")
        if "error" not in final_gpu:
            print(f"Использование GPU VRAM: {final_gpu['memory_used_gb']:.1f}/{final_gpu['memory_total_gb']:.1f} GB")
            print(f"Загрузка GPU: {final_gpu['utilization_percent']:.1f}%")
        
        # Точность по задачам
        if 'results' in results:
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
        print(f"  Скорость генерации: {speed_metrics['average_tokens_per_second']:.2f} токенов/сек")
        print(f"  Обработано токенов: {speed_metrics['total_tokens']}")
        print(f"  Время генерации: {speed_metrics['total_time']:.2f} сек")
        print(f"  Общее время: {load_time + eval_time + speed_metrics['total_time']:.2f} сек")
        print(f"  Результаты сохранены в: {result_file}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Ошибка при оценке: {str(e)}")
        logger.exception("Подробности ошибки:")
        raise

if __name__ == "__main__":
    main()