"""Клиент для работы с Kie.ai Veo 3.1 API"""

import os
from typing import List, Optional, Dict, Any

import requests


class KieVeoClient:
    """Обертка над REST API https://api.kie.ai/api/v1/veo/generate"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("KIE_API_KEY")
        if not self.api_key:
            raise ValueError("Kie.ai API key не найден. Установите KIE_API_KEY в .env")

        self.base_url = (base_url or os.getenv("KIE_API_BASE_URL") or "https://api.kie.ai").rstrip("/")

    def generate_video(
        self,
        prompt: str,
        image_urls: Optional[List[str]] = None,
        model: str = "veo3_fast",
        aspect_ratio: str = "16:9",
        generation_type: Optional[str] = None,
        enable_translation: bool = True,
        callback_url: Optional[str] = None,
        seeds: Optional[int] = None,
        watermark: Optional[str] = None
    ) -> Dict[str, Any]:
        """Создает задачу генерации видео через Kie.ai"""

        # Очищаем и ограничиваем промпт
        # Убираем двойные кавычки в начале и конце, если они есть
        cleaned_prompt = prompt.strip()
        if cleaned_prompt.startswith('"') and cleaned_prompt.endswith('"'):
            cleaned_prompt = cleaned_prompt[1:-1]

        # Ограничиваем длину промпта (примерно 1000 символов для безопасности)
        # Kie.ai может иметь ограничения на длину промпта
        MAX_PROMPT_LENGTH = 800

        if len(cleaned_prompt) > MAX_PROMPT_LENGTH:
            print(f"[WARNING] Промпт слишком длинный ({len(cleaned_prompt)} символов), обрезаем до {MAX_PROMPT_LENGTH}")
            # Обрезаем до последнего предложения в пределах лимита
            truncated = cleaned_prompt[:MAX_PROMPT_LENGTH]
            last_period = truncated.rfind('.')
            if last_period > MAX_PROMPT_LENGTH * 0.7:  # Если точка находится в разумных пределах
                cleaned_prompt = truncated[:last_period + 1]
            else:
                cleaned_prompt = truncated

        payload: Dict[str, Any] = {
            "prompt": cleaned_prompt,
            "model": model,
            "aspectRatio": aspect_ratio,
            "enableTranslation": enable_translation,
        }

        if image_urls:
            payload["imageUrls"] = image_urls

        if generation_type:
            payload["generationType"] = generation_type
        elif image_urls:
            # У Kie для imageUrls требуется режим REFERENCE_2_VIDEO
            payload["generationType"] = "REFERENCE_2_VIDEO"

        if callback_url:
            payload["callBackUrl"] = callback_url

        if seeds is not None:
            payload["seeds"] = seeds

        if watermark:
            payload["watermark"] = watermark

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        url = f"{self.base_url}/api/v1/veo/generate"

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            # Улучшенная обработка ошибок для отладки
            error_detail = f"HTTP {response.status_code}"
            try:
                error_json = response.json()
                error_detail += f": {error_json}"
            except:
                error_detail += f": {response.text[:500]}"

            print(f"[ERROR] Kie.ai API ошибка: {error_detail}")
            print(f"[DEBUG] Payload отправлен (prompt length: {len(payload.get('prompt', ''))}): {payload}")

            # Бросаем исключение с деталями
            raise requests.HTTPError(f"Kie.ai API error: {error_detail}")
        except requests.RequestException as e:
            print(f"[ERROR] Kie.ai запрос не удался: {e}")
            raise

    def check_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Проверяет статус задачи генерации видео по Task ID и возвращает видео URL если готово

        Согласно документации Kie.ai, есть два способа получить результат:
        1. Callback URL (рекомендуется) - система автоматически отправляет результат
        2. Polling через "Get Veo3.1 Video Details" endpoint (точный URL не указан в документации)

        Если endpoint для проверки статуса недоступен, пробуем проверить доступность видео напрямую.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Согласно документации, endpoint может быть:
        # - /api/v1/veo/detail?taskId={task_id}
        # - /api/v1/veo/detail/{task_id}
        # Но точный формат не указан, пробуем разные варианты
        endpoints = [
            # Вариант 1: Query параметр (согласно документации "Get Veo3.1 Video Details")
            f"{self.base_url}/api/v1/veo/detail?taskId={task_id}",
            # Вариант 2: Path параметр
            f"{self.base_url}/api/v1/veo/detail/{task_id}",
            # Вариант 3: Без /api
            f"{self.base_url}/v1/veo/detail?taskId={task_id}",
            f"{self.base_url}/v1/veo/detail/{task_id}",
            # Вариант 4: Альтернативные варианты (старые, на случай если API изменился)
            f"{self.base_url}/api/v1/veo/task/{task_id}",
            f"{self.base_url}/v1/video/task/{task_id}",
            f"{self.base_url}/api/v1/video/task/{task_id}",
        ]

        last_error = None
        last_response = None
        tried_endpoints = []

        for url in endpoints:
            tried_endpoints.append(url)
            try:
                print(f"[i] Пробую endpoint: {url}")
                response = requests.get(url, headers=headers, timeout=30)
                last_response = response

                print(f"[i] Ответ: HTTP {response.status_code}")

                if response.status_code == 200:
                    print(f"[OK] Status endpoint работает: {url}")
                    data = response.json()

                    # Логируем полный ответ для отладки (первые 500 символов)
                    print(f"[DEBUG] Полный ответ API: {str(data)[:500]}")

                    result = {
                        "status": data.get("status", "unknown"),
                        "task_id": task_id,
                        "raw_response": data
                    }

                    # Извлекаем video_url из разных возможных мест в ответе
                    video_url = None
                    if isinstance(data, dict):
                        # Формат из callback (согласно документации):
                        # {
                        #   "code": 200,
                        #   "data": {
                        #     "taskId": "...",
                        #     "info": {
                        #       "resultUrls": ["https://tempfile.aiquickdraw.com/v/..."],
                        #       "originUrls": [...],
                        #       "resolution": "1080p"
                        #     }
                        #   }
                        # }

                        # Вариант 1: data.data.info.resultUrls (стандартный формат callback)
                        if "data" in data and isinstance(data["data"], dict):
                            info = data["data"].get("info", {})
                            if isinstance(info, dict):
                                result_urls = info.get("resultUrls", [])
                                if isinstance(result_urls, list) and len(result_urls) > 0:
                                    video_url = result_urls[0]

                        # Вариант 2: data.data.resultUrls (без info)
                        if not video_url and "data" in data and isinstance(data["data"], dict):
                            result_urls = data["data"].get("resultUrls", [])
                            if isinstance(result_urls, list) and len(result_urls) > 0:
                                video_url = result_urls[0]

                        # Вариант 3: data.result.video_url
                        if not video_url and "result" in data:
                            result_obj = data["result"]
                            if isinstance(result_obj, dict):
                                video_url = result_obj.get("video_url") or result_obj.get("videoUrl") or result_obj.get("video")
                            elif isinstance(result_obj, str) and result_obj.startswith("http"):
                                video_url = result_obj

                        # Вариант 4: data.video_url (прямо в корне)
                        if not video_url:
                            video_url = data.get("video_url") or data.get("videoUrl") or data.get("video")

                        # Вариант 5: Ищем любую ссылку на tempfile.aiquickdraw.com через regex
                        if not video_url:
                            import re
                            json_str = str(data)
                            video_patterns = [
                                r'(https?://tempfile\.aiquickdraw\.com/[^\s"\']+\.mp4)',
                                r'(https?://[^\s"\']+\.mp4)',
                            ]
                            for pattern in video_patterns:
                                matches = re.findall(pattern, json_str)
                                if matches:
                                    video_url = matches[0]
                                    print(f"[i] Найдена ссылка на видео через regex: {video_url}")
                                    break

                    if video_url:
                        result["video_url"] = video_url
                        result["status"] = "completed"
                        print(f"[OK] Видео готово! URL: {video_url}")
                    else:
                        # Проверяем code в ответе
                        code = data.get("code")
                        if code == 200:
                            result["status"] = "completed"
                            print(f"[i] Статус код 200, но video_url не найден в ответе")
                        elif code:
                            result["status"] = "failed" if code >= 400 else "processing"

                    return result
                elif response.status_code == 404:
                    print(f"[i] Endpoint {url} вернул 404, пробую следующий...")
                    continue
                elif response.status_code == 401:
                    error_text = response.text[:200]
                    print(f"[!] Endpoint {url} вернул 401 (Unauthorized): {error_text}")
                    continue
                else:
                    error_text = response.text[:500]
                    print(f"[!] Endpoint {url} вернул {response.status_code}: {error_text}")
                    # Если не 404, возможно endpoint существует но формат неправильный
                    continue
            except requests.RequestException as e:
                last_error = e
                print(f"[!] Ошибка запроса к {url}: {e}")
                continue

        # Если все endpoints вернули 404, пробуем альтернативный подход:
        # Проверяем доступность видео напрямую через известный формат URL
        print(f"[i] Все endpoints вернули 404, пробую проверить доступность видео напрямую...")

        # Формат URL: https://tempfile.aiquickdraw.com/v/{task_id}_{timestamp}.mp4
        # Пробуем проверить доступность с текущим timestamp и несколькими вариантами вокруг
        import time
        current_timestamp = int(time.time())

        # Пробуем несколько вариантов timestamp:
        # - Текущее время и несколько минут назад (видео могло быть сгенерировано недавно)
        # - Проверяем по 5-минутным интервалам за последние 2 часа
        test_timestamps = []
        for offset in range(0, 7200, 300):  # Каждые 5 минут за последние 2 часа
            test_timestamps.append(current_timestamp - offset)

        # Также добавляем более точную проверку последних 10 минут (каждую минуту)
        for offset in range(0, 600, 60):
            test_timestamps.append(current_timestamp - offset)

        # Убираем дубликаты и сортируем
        test_timestamps = sorted(set(test_timestamps))

        for test_timestamp in test_timestamps:
            test_url = f"https://tempfile.aiquickdraw.com/v/{task_id}_{test_timestamp}.mp4"

            try:
                head_response = requests.head(test_url, timeout=3, allow_redirects=True)
                if head_response.status_code == 200:
                    print(f"[OK] Видео найдено напрямую: {test_url}")
                    return {
                        "status": "completed",
                        "task_id": task_id,
                        "video_url": test_url,
                        "method": "direct_check"
                    }
            except:
                continue

        # Если не нашли через прямой доступ, возвращаем ошибку
        error_messages = []
        if last_response:
            error_messages.append(f"Последний ответ: HTTP {last_response.status_code}")
            if last_response.text:
                error_messages.append(f"Ответ сервера: {last_response.text[:300]}")
        if last_error:
            error_messages.append(f"Ошибка запроса: {last_error}")

        error_detail = ". ".join(error_messages) if error_messages else "Все endpoints вернули 404"
        error_detail += f". Пробовал endpoints: {len(tried_endpoints)} вариантов. Прямая проверка видео также не дала результата."

        # Не бросаем исключение, возвращаем статус "processing" чтобы интерфейс продолжал проверку
        return {
            "status": "processing",
            "task_id": task_id,
            "error": error_detail,
            "note": "Endpoint для проверки статуса недоступен. Рекомендуется использовать callback URL или проверять вручную на https://app.kie.ai"
        }


