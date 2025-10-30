# Veo 3.1 - Генерация видео по образцам

## Описание

Полный workflow для генерации видео в Google Veo 3.1 на основе референсных изображений и видео.

## Функционал

### 1. Анализ референсов
- Анализ 3 референсных изображений (минимум 1)
- Опциональный анализ референсного видео для понимания стиля движения
- Автоматическое объединение анализов

### 2. Генерация промпта
- Создание промпта на основе анализа
- Оптимизация под Veo 3.1
- Возможность добавить дополнительные инструкции

### 3. Генерация видео
- Отправка запроса в Veo 3.1 API
- Поддержка референсных изображений
- Поддержка видео-референса
- Настройка параметров (длительность, соотношение сторон, качество)

## Использование

### Вариант 1: Веб-интерфейс

1. Запустите сервер: `start_server.bat`
2. Откройте интерфейс: `start_veo31_ui.bat`
   Или вручную: `veo31_interface.html`
3. Загрузите:
   - 3 изображения (минимум 1)
   - Опционально: видео-референс
4. Настройте параметры
5. Нажмите "Сгенерировать видео"

### Вариант 2: Через API

#### Загрузка файлов:
```bash
curl -X POST "http://localhost:8000/veo31/generate" \
  -F "reference_image1=@image1.jpg" \
  -F "reference_image2=@image2.jpg" \
  -F "reference_image3=@image3.jpg" \
  -F "video_reference=@reference.mp4" \
  -F "use_case=product_video" \
  -F "duration_seconds=10" \
  -F "aspect_ratio=16:9" \
  -F "quality=high"
```

#### Через URL:
```bash
curl -X POST "http://localhost:8000/veo31/generate-from-urls" \
  -F "reference_image_urls=https://example.com/img1.jpg,https://example.com/img2.jpg,https://example.com/img3.jpg" \
  -F "video_reference_url=https://example.com/ref.mp4" \
  -F "use_case=product_video" \
  -F "duration_seconds=10"
```

### Вариант 3: Python код

```python
from veo31_generator import Veo31VideoGenerator

generator = Veo31VideoGenerator()

result = generator.generate_from_references(
    reference_images=[
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
        "https://example.com/image3.jpg"
    ],
    video_reference="https://example.com/reference.mp4",  # Опционально
    use_case="product_video",
    duration_seconds=10,
    aspect_ratio="16:9",
    quality="high"
)

print(result["step2_prompt"])  # Сгенерированный промпт
print(result["step3_video_generation"]["video_url"])  # URL видео
```

## Параметры

### Референсы
- `reference_images`: Список из 1-3 изображений (URL, путь или bytes)
- `video_reference`: Опциональное видео для анализа стиля движения

### Генерация
- `duration_seconds`: Длительность видео (5-60 секунд)
- `aspect_ratio`: Соотношение сторон ("16:9", "9:16", "1:1")
- `quality`: Качество ("high", "standard")
- `use_case`: Тип задачи (product_video, hero_image, gemstone_closeup, luxury_brand, general)
- `additional_prompt`: Дополнительные инструкции к промпту

## Workflow

```
1. Загрузка референсов
   ↓
2. Анализ изображений + видео (если есть)
   ↓
3. Генерация промпта на основе анализа
   ↓
4. Отправка в Veo 3.1 API с референсами и промптом
   ↓
5. Получение видео
```

## Проверка статуса

Если видео генерируется асинхронно:
```bash
curl "http://localhost:8000/veo31/status/{task_id}"
```

Или в Python:
```python
generator.check_status(task_id)
```

## Важно

⚠️ **Примечание о Veo 3.1 API:**
Точный endpoint и структура запросов могут отличаться от реализованного.
Текущая реализация использует структуру на основе Gemini API.
При получении официальной документации Veo 3.1 API может потребоваться обновление:
- `veo31_client.py` - изменить endpoint и формат запросов
- Возможно потребуется использование Vertex AI вместо Generative AI API

## Требования

- `GEMINI_API_KEY` или `GOOGLE_API_KEY` в .env файле (для Veo 3.1)
- `OPENAI_API_KEY` (для анализа изображений)
- Интернет-соединение для работы с API

## Troubleshooting

**Ошибка: "Veo 3.1 генератор не инициализирован"**
- Проверьте наличие `GEMINI_API_KEY` в .env файле
- Перезапустите сервер

**Ошибка: "API endpoint не найден"**
- Возможно, Veo 3.1 API использует другой endpoint
- Проверьте официальную документацию Google
- Возможно требуется использовать Vertex AI

**Видео генерируется долго**
- Генерация видео может занять несколько минут
- Используйте проверку статуса через `/veo31/status/{task_id}`

