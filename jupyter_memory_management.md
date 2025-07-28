# Управление памятью GPU в Jupyter Notebook

## Автоматическая очистка памяти

Код уже настроен для автоматической очистки памяти после каждой модели в функциях `compare_models_basic()` и `compare_models_full()`.

## Ручная очистка памяти

### 1. Импорт функции очистки
```python
from src.main import force_clear_gpu_memory
```

### 2. Очистка после каждой модели
```python
# После работы с моделью
del model, tokenizer
force_clear_gpu_memory("название_модели")
```

### 3. Полная очистка памяти
```python
# Очистка всей GPU памяти
force_clear_gpu_memory()
```

## Альтернативные способы очистки

### 1. Сброс ядра Jupyter Notebook
- **Kernel → Restart** - перезапускает ядро и очищает всю память
- **Kernel → Restart & Clear Output** - перезапускает ядро и очищает вывод

### 2. Магические команды Jupyter
```python
# Очистка всех переменных
%reset -f

# Очистка вывода
%clear

# Очистка кэша
%cache_clear
```

### 3. Принудительная очистка PyTorch
```python
import torch
import gc

# Удаление всех тензоров
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            del obj
    except:
        pass

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

## Рекомендуемый workflow

### Для тестирования нескольких моделей:
```python
from src.main import compare_models_basic, force_clear_gpu_memory

# Конфигурация моделей
model_configs = [
    {"name": "GPT-2", "path": "gpt2"},
    {"name": "GPT-2 Medium", "path": "gpt2-medium"},
    {"name": "GPT-2 Large", "path": "gpt2-large"}
]

# Автоматическое тестирование с очисткой памяти
results = compare_models_basic(model_configs)

# Дополнительная очистка в конце
force_clear_gpu_memory()
```

### Для ручного тестирования:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.main import evaluate_basic_model, force_clear_gpu_memory

# Модель 1
tokenizer1 = AutoTokenizer.from_pretrained("gpt2")
model1 = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype="auto", device_map="auto")
results1 = evaluate_basic_model(model1, tokenizer1, "GPT-2")

# Очистка памяти
del model1, tokenizer1
force_clear_gpu_memory("GPT-2")

# Модель 2
tokenizer2 = AutoTokenizer.from_pretrained("gpt2-medium")
model2 = AutoModelForCausalLM.from_pretrained("gpt2-medium", torch_dtype="auto", device_map="auto")
results2 = evaluate_basic_model(model2, tokenizer2, "GPT-2 Medium")

# Очистка памяти
del model2, tokenizer2
force_clear_gpu_memory("GPT-2 Medium")
```

## Мониторинг памяти

### Проверка текущего использования GPU:
```python
import torch

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU память: {allocated:.2f} GB (выделено) / {reserved:.2f} GB (зарезервировано)")
```

### Информация о GPU:
```python
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Общая память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

## Советы

1. **Всегда очищайте память** после работы с моделью
2. **Используйте автоматические функции** `compare_models_basic()` и `compare_models_full()`
3. **Мониторьте память** перед загрузкой новой модели
4. **Перезапускайте ядро** если память не освобождается
5. **Используйте меньшие модели** если память ограничена 