# Dify Workflow Setup Guide

## Быстрая установка для Video Prompt Pipeline

### Шаг 1: Регистрация

1. Зарегистрируйтесь на [dify.ai](https://dify.ai)
2. Войдите в свой аккаунт

### Шаг 2: Настройка API ключей

1. Перейдите в **Settings** → **Model Provider**
2. Добавьте OpenAI API Key:
   - Provider: OpenAI
   - API Key: ваш ключ от OpenAI

3. (Опционально) Добавьте Google Gemini:
   - Provider: Google
   - API Key: ваш ключ от Gemini

### Шаг 3: Создание Workflow

1. Перейдите в **Workflows** → **Create Workflow**
2. Выберите тип: **Chatflow**
3. Используйте импорт YAML:

   - Нажмите на меню (три точки) → **Import from YAML**
   - Загрузите `dify_workflow.yaml`

   ИЛИ создайте вручную:

#### Узел Start (Inputs)
- Добавьте переменные:
  - `video_url` (text-input, required)
  - `platform` (select: veo3, sora2, seedream)
  - `use_case` (select: product_video, hero_image, gemstone_closeup, luxury_brand, general)

#### Узел HTTP Request (Download Media)
- Type: HTTP Request
- Method: GET
- URL: `{{start.video_url}}`
- Response Format: File/Binary

#### Узел LLM (Agent 1: Visual Analyzer)
- Model: OpenAI GPT-4o
- Temperature: 0.3
- Max Tokens: 2000
- Prompt:
  ```
  Analyze this image/video in extreme detail as a professional jewelry photographer.

  [Upload file from HTTP Request node]

  Extract and describe:
  1. Main objects
  2. Materials
  3. Lighting
  4. Camera work
  5. Composition
  6. Background and surfaces
  7. Motion and dynamics
  8. Style and mood

  Output as JSON with these fields: main_objects, materials, lighting, camera_work, composition, background_surfaces, motion_dynamics, style_mood
  ```

#### Узел LLM (Agent 2: Prompt Generator)
- Model: OpenAI GPT-4o
- Temperature: 0.7
- Max Tokens: 1000
- Prompt:
  ```
  You are an expert prompt engineer for AI video generators.

  Platform: {{start.platform}}
  Use case: {{start.use_case}}

  Create an optimized prompt based on this scene analysis:
  {{analyze-node.text}}

  Rules for {{start.platform}}:
  - [Platform-specific rules]

  Return ONLY the final prompt in English, ready to use.
  ```

#### Узел Answer
- Output: `{{generate-node.text}}`
- Metadata: Include platform and use_case

### Шаг 4: Публикация и API

1. Сохраните workflow
2. Нажмите **Publish**
3. Перейдите в **API Access**
4. Создайте API Key

### Шаг 5: Тестирование

#### Через API:
```bash
curl -X POST 'https://api.dify.ai/v1/chat-messages' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": {
      "video_url": "https://example.com/jewelry.jpg",
      "platform": "veo3",
      "use_case": "product_video"
    },
    "user": "user-123"
  }'
```

#### Через UI:
1. Откройте workflow
2. Заполните inputs:
   - Video URL
   - Platform
   - Use Case
3. Нажмите **Run**

## Альтернатива: Использование Python API сервера

Если хотите использовать наш FastAPI сервер с Dify:

1. Запустите сервер:
   ```bash
   cd video-prompt-pipeline
   python api_server.py
   ```

2. В Dify создайте HTTP Request узел:
   - URL: `http://your-server:8000/process`
   - Method: POST
   - Body: Form Data
     - `url`: `{{start.video_url}}`
     - `platform`: `{{start.platform}}`
     - `use_case`: `{{start.use_case}}`

## Особенности Dify

- **Визуальный редактор**: Легко настраивать workflow через UI
- **Versioning**: Сохранение версий workflow
- **Analytics**: Отслеживание использования
- **Team Collaboration**: Работа в команде

## Troubleshooting

**Ошибка: "File upload failed"**
- Убедитесь, что HTTP Request node правильно скачивает файл
- Проверьте доступность URL

**Ошибка: "Model provider not configured"**
- Проверьте настройки API ключей в Settings

**Тайм-аут при анализе**
- Увеличьте timeout для HTTP Request узла
- Для видео используйте Gemini (лучше работает с видео)

