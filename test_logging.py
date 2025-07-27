#!/usr/bin/env python3
"""
Тестовый скрипт для проверки логирования
"""

import sys
import os

# Добавляем путь к модулю
sys.path.append('./src')

# Импортируем функции
from main import ensure_logging_setup, logger

def test_logging():
    """Тестирует работу логирования"""
    print("Тестирование логирования...")
    
    # Принудительно настраиваем логирование
    logger = ensure_logging_setup()
    
    # Тестируем различные уровни логирования
    logger.debug("Это debug сообщение")
    logger.info("Это info сообщение")
    logger.warning("Это warning сообщение")
    logger.error("Это error сообщение")
    
    # Проверяем, что файл создался
    log_file = os.path.join("results", "model_evaluation.log")
    if os.path.exists(log_file):
        print(f"✅ Лог-файл создан: {log_file}")
        
        # Читаем содержимое файла
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"📄 Размер файла: {len(content)} символов")
            if content:
                print("✅ Логирование работает! Содержимое файла:")
                print("-" * 50)
                print(content)
                print("-" * 50)
            else:
                print("❌ Файл пустой!")
    else:
        print(f"❌ Лог-файл не найден: {log_file}")

if __name__ == "__main__":
    test_logging() 