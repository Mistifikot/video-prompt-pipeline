"""Проверка и исправление .env файла"""
import os
from pathlib import Path
from dotenv import load_dotenv

env_file = Path(__file__).parent / '.env'
print(f"Проверка .env файла: {env_file.absolute()}")

if env_file.exists():
    print("OK - Файл .env найден")
    load_dotenv(env_file)
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if openai_key:
        print(f"OK - OPENAI_API_KEY загружен (начинается с: {openai_key[:10]}...)")
    else:
        print("ОШИБКА - OPENAI_API_KEY НЕ ЗАГРУЖЕН!")
        print("Проверьте содержимое .env файла")

    if gemini_key:
        print(f"OK - GEMINI_API_KEY загружен")
    else:
        print("INFO - GEMINI_API_KEY не найден (опционально)")
else:
    print("ОШИБКА - Файл .env НЕ НАЙДЕН!")
    print(f"Создаю новый .env файл...")

    # Создаем .env с шаблоном (без реальных ключей!)
    env_content = """OPENAI_API_KEY=your-openai-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
USE_GEMINI=false
PORT=8000
"""
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_content)
    print(f"OK - .env файл создан: {env_file}")

    # Проверяем еще раз
    load_dotenv(env_file)
    if os.getenv("OPENAI_API_KEY"):
        print("OK - Ключи успешно записаны и загружены!")

