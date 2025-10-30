# N8N Workflow Setup Guide

## Быстрая установка для Video Prompt Pipeline

### Шаг 1: Регистрация и доступ

1. Зарегистрируйтесь на [n8n.cloud](https://n8n.cloud) (облачная версия) или установите self-hosted:
   ```bash
   npm install n8n -g
   n8n start
   ```

### Шаг 2: Импорт workflow

1. В N8N создайте новый workflow
2. Импортируйте `n8n_workflow.json`:
   - Нажмите на меню (три точки) → Import from File
   - Выберите `n8n_workflow.json`

### Шаг 3: Настройка API ключей

1. Перейдите в **Settings** → **Credentials**
2. Создайте credential для OpenAI:
   - Название: "OpenAI API"
   - API Key: ваш ключ от OpenAI

### Шаг 4: Настройка узлов

#### Узел "Webhook Trigger"
- Метод: POST
- Path: `video-prompt`
- Response Mode: Last Node

#### Узел "Download Media"
- Метод: GET
- URL: `={{ $json.body.video_url }}`
- Response Format: File

#### Узел "Agent 1: Visual Analyzer"
- Model: `gpt-4o`
- Temperature: `0.3`
- Max Tokens: `2000`
- В attachments добавьте binary data из предыдущего узла

#### Узел "Agent 2: Prompt Generator"
- Model: `gpt-4o`
- Temperature: `0.7`
- Max Tokens: `1000`
- System prompt настроен автоматически

### Шаг 5: Активация и тестирование

1. Сохраните workflow
2. Активируйте его (переключатель в правом верхнем углу)
3. Скопируйте Production Webhook URL

### Шаг 6: Тестирование

```bash
curl -X POST "YOUR_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/jewelry.jpg",
    "platform": "veo3",
    "use_case": "product_video"
  }'
```

## Альтернатива: Использование FastAPI сервера

Если не хотите использовать N8N напрямую, можно использовать наш FastAPI сервер:

1. Запустите сервер:
   ```bash
   cd video-prompt-pipeline
   python api_server.py
   ```

2. В N8N создайте HTTP Request узел:
   - Method: POST
   - URL: `http://localhost:8000/process`
   - Body: Form Data
     - `url`: `={{ $json.body.video_url }}`
     - `platform`: `={{ $json.body.platform }}`
     - `use_case`: `={{ $json.body.use_case }}`

## Примеры payload

### Изображение для Veo 3
```json
{
  "video_url": "https://example.com/jewelry.jpg",
  "platform": "veo3",
  "use_case": "product_video"
}
```

### Видео для Sora 2
```json
{
  "video_url": "https://example.com/jewelry-showcase.mp4",
  "platform": "sora2",
  "use_case": "luxury_brand"
}
```

## Troubleshooting

**Ошибка: "API key не найден"**
- Проверьте, что credential создан и правильно подключен к узлам

**Ошибка: "Не удалось загрузить медиа"**
- Проверьте доступность URL
- Убедитесь, что URL доступен публично

**Тайм-аут при анализе видео**
- Увеличьте timeout в узле Download Media
- Для больших видео используйте Gemini (требует дополнительной настройки)

