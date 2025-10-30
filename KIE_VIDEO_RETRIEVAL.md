# Получение видео из Kie.ai после генерации

## Что было сделано

Обновлена система для автоматического получения готового видео из Kie.ai API после завершения генерации.

## Изменения

### 1. `kie_api_client.py`
- ✅ Исправлен endpoint для проверки статуса: `https://api.kie.ai/v1/video/task/{task_id}`
- ✅ Добавлен парсинг ответа для извлечения `video_url` из разных форматов ответа
- ✅ Метод `check_task_status()` теперь возвращает структурированный ответ с `video_url` если видео готово

### 2. `veo31_generator.py`
- ✅ Добавлен метод `check_kie_status()` для проверки статуса Kie.ai
- ✅ Обновлен метод `check_status()` с поддержкой параметра `provider`

### 3. `api_server.py`
- ✅ Обновлен endpoint `/veo31/kie-status/{task_id}` для возврата `video_url` в структурированном формате

### 4. `veo31_interface.html`
- ✅ Улучшена логика автоматической проверки статуса
- ✅ Добавлена поддержка получения `video_url` из обновленного формата ответа
- ✅ Автоматическое отображение видео плеера когда видео готово
- ✅ Улучшена обработка различных статусов (`processing`, `completed`, `failed`)

## Использование

### Через API

#### 1. Запуск генерации
```bash
curl -X POST "http://localhost:8000/veo31/generate-from-urls" \
  -F "reference_image_urls=https://images.unsplash.com/photo-1656543802898-41c8c46683a7" \
  -F "prefer_kie_api=true" \
  -F "use_case=product_video" \
  -F "duration_seconds=5" \
  -F "aspect_ratio=16:9"
```

Ответ будет содержать `task_id`:
```json
{
  "step3_video_generation": {
    "provider": "kie.ai",
    "status": "submitted",
    "task_id": "88513be422a2f971886273b32867fd6f"
  }
}
```

#### 2. Проверка статуса и получение видео
```bash
curl "http://localhost:8000/veo31/kie-status/88513be422a2f971886273b32867fd6f"
```

Ответ когда видео готово:
```json
{
  "provider": "kie.ai",
  "task_id": "88513be422a2f971886273b32867fd6f",
  "status": "completed",
  "video_url": "https://..."
}
```

### Через веб-интерфейс

1. Откройте `veo31_interface.html` или `http://localhost:8000/veo31`
2. Загрузите изображения или укажите URL
3. Выберите "Использовать Kie.ai API"
4. Нажмите "Сгенерировать видео"
5. Интерфейс автоматически будет проверять статус каждые 10 секунд
6. Когда видео готово, оно автоматически отобразится в плеере

## Формат ответа от Kie.ai API

Согласно документации, endpoint `/v1/video/task/{task_id}` возвращает:

```json
{
  "status": "completed",
  "result": {
    "video_url": "https://tempfile.aiquickdraw.com/v/8b95fc92b0a5e9fff2b13256dad8b135_1761820876.mp4"
  }
}
```

Наш код поддерживает также альтернативные форматы:
- `status.video_url` (прямо в корне)
- `status.data.video_url` (через data)
- `status.result.videoUrl` (camelCase)
- `status.result.video` (просто "video")
- `status.url` (если это видео ссылка)
- Автоматический поиск ссылок на `tempfile.aiquickdraw.com` через regex

**Формат ссылки на видео:**
Видео от Kie.ai обычно возвращается в формате:
```
https://tempfile.aiquickdraw.com/v/{hash}_{timestamp}.mp4
```

Эти ссылки доступны напрямую и могут быть использованы в `<video>` тегах или для скачивания.

## Автоматическая проверка статуса

Веб-интерфейс автоматически проверяет статус:
- **Каждые 10 секунд** для Kie.ai
- **Каждые 5 секунд** для Google Veo

Проверка прекращается когда:
- Видео готово (`status: "completed"` и есть `video_url`)
- Произошла ошибка (`status: "failed"` или `status: "error"`)

## Troubleshooting

### Видео не отображается
1. Проверьте что `task_id` правильный
2. Убедитесь что генерация завершена на платформе Kie.ai
3. Проверьте консоль браузера на ошибки
4. Проверьте логи сервера на наличие `[OK] Видео готово!`

### Endpoint не работает
Если endpoint `/v1/video/task/{task_id}` не работает:
- Проверьте что API ключ правильный
- Проверьте что base_url правильный: `https://api.kie.ai`
- Проверьте документацию Kie.ai на изменения в API

### Статус показывает "unknown"
Это означает что endpoint не найден. Система продолжит проверку, но вы можете:
- Проверить статус вручную на https://app.kie.ai
- Использовать task_id из ответа для ручной проверки

## Пример успешного workflow

1. **Генерация запущена:**
   ```json
   {
     "step3_video_generation": {
       "provider": "kie.ai",
       "status": "submitted",
       "task_id": "abc123"
     }
   }
   ```

2. **Проверка статуса (видео еще генерируется):**
   ```json
   {
     "status": "processing",
     "task_id": "abc123"
   }
   ```

3. **Видео готово:**
   ```json
   {
     "status": "completed",
     "task_id": "abc123",
     "video_url": "https://cdn.kie.ai/videos/abc123.mp4"
   }
   ```

4. **В интерфейсе автоматически:**
   - Останавливается проверка статуса
   - Отображается видео плеер
   - Появляются кнопки "Скачать" и "Копировать ссылку"

