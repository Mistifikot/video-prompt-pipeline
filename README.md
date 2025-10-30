# Video Prompt Pipeline

Автоматическая генерация промптов для AI-видео систем из визуальных референсов.

## 🚀 Быстрый старт

**Один файл для запуска всего:**
```bash
START.bat
```

Этот файл:
- Проверяет .env
- Устанавливает зависимости (если нужно)
- Останавливает старый сервер
- Запускает новый сервер
- Открывает оба интерфейса в браузере

## 📋 Функционал

### 1. Анализ видео/изображений
- Загрузите видео или изображение
- Получите детальный анализ
- Сгенерируйте промпт для Veo 3, Sora 2, Seedream

**Интерфейс:** `web_interface.html` или `http://localhost:8000/ui`

### 2. Veo 3.1 - Генерация видео по образцам
- Загрузите 3 референсных изображения
- Опционально: видео-референс
- Автоматический анализ → промпт → генерация видео

**Интерфейс:** `veo31_interface.html` или `http://localhost:8000/veo31`

## 📁 Структура проекта

```
video-prompt-pipeline/
├── START.bat                    # Единый запуск всего
├── fix_env.py                   # Проверка .env
├── api_server.py                # Главный сервер
├── pipeline.py                  # Основной пайплайн
├── agent1_visual_analyzer.py    # Анализатор
├── agent2_prompt_generator.py   # Генератор промптов
├── veo31_client.py              # Клиент Veo 3.1 API
├── veo31_generator.py           # Veo 3.1 workflow
├── web_interface.html           # Интерфейс анализа
├── veo31_interface.html          # Интерфейс Veo 3.1
├── .env                         # API ключи (не коммитить!)
├── requirements.txt              # Зависимости
└── README.md                     # Документация
```

## ⚙️ Настройка

1. Создайте `.env` файл:
```
OPENAI_API_KEY=ваш-ключ
GEMINI_API_KEY=ваш-ключ
USE_GEMINI=false
PORT=8000
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Запустите:
```bash
START.bat
```

## 🌐 API Endpoints

### Анализ:
- `POST /process` - Полный анализ + промпт
- `POST /analyze` - Только анализ
- `POST /generate` - Только генерация промпта

### Veo 3.1:
- `POST /veo31/generate` - Загрузка файлов
- `POST /veo31/generate-from-urls` - Через URL
- `GET /veo31/status/{task_id}` - Статус генерации

## 📝 Использование

### Вариант 1: Веб-интерфейс
Запустите `START.bat` и используйте открывшиеся интерфейсы.

### Вариант 2: API
```bash
curl -X POST "http://localhost:8000/process" \
  -F "url=https://example.com/video.mp4" \
  -F "platform=veo3" \
  -F "use_case=product_video"
```

## 🔧 Troubleshooting

**Сервер не запускается:**
- Проверьте .env файл: `python fix_env.py`
- Убедитесь что порт 8000 свободен

**Ошибка "API key не найден":**
- Проверьте .env файл существует
- Перезапустите сервер

**Veo 3.1 не работает:**
- Убедитесь что `GEMINI_API_KEY` установлен
- Проверьте endpoint в `veo31_client.py` (может потребоваться обновление)

## 📚 Документация

- `VEO31_README.md` - Документация по Veo 3.1
- `README.md` - Эта документация

## ⚠️ Важно

- `.env` файл содержит API ключи - не коммитьте в git
- Veo 3.1 API endpoint может потребовать обновления при получении официальной документации
