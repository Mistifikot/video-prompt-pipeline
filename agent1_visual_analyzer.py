"""
Агент 1: Визуальный смысловой анализатор
Анализирует изображения/видео и извлекает детальное описание сцены
"""

import os
import json
import base64
import requests
from typing import Dict, Optional, Union, Tuple, List
from pathlib import Path
import mimetypes
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


class VisualAnalyzer:
    """Анализатор визуального контента для извлечения структурированного описания сцены"""

    def __init__(self, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.preferred_model = "gpt-4o"  # GPT-4o поддерживает vision и видео

    def _get_analysis_prompt(self) -> str:
        """Возвращает универсальный промпт для анализа медиа"""
        return """
ROLE

You are a senior Director of Photography (DP), camera operator, and color supervisor. Analyze the provided media input (single image, sequence of frames, or video) exactly as a professional film crew would. Do not invent details—mark unknowns as null and report confidence.

OBJECTIVE

1) Extract the most complete description of scene, optics, camera motion, lighting, composition, backgrounds, textures, and post effects.

2) If input is a video, reconstruct the camera path over time (keyframes, durations, motion types, stabilization, micro-jitters).

3) Produce a precise transfer-prompt and parameter set so a video generator can replicate the camera move and lighting on a NEW subject.

OUTPUT FORMAT

Return two blocks: (A) Human-readable report and (B) Machine-readable JSON (strictly following the schema below), then (C) a Transfer-Prompt for the generator.

- Camera motion terminology: pan, tilt, dolly, truck, pedestal, crane(jib), arc/orbit, roll, zoom, handheld, steadicam, gimbal, drone.
- Lighting roles: key, fill, rim/back, practical, ambient, bounce.
- Color temperature in Kelvin.
- If possible, estimate fps, exposure and shutter-angle from motion blur.
- Use units: meters, millimeters, degrees, seconds, Kelvin.
- Keep wording factual and concise. No flowery language.

WHAT TO EXTRACT

1) Media metadata
- media_type: image | video | frames
- duration_s (if video), fps (estimated or exact), resolution_px (W×H), aspect_ratio
- playback_speed_anomalies: speed-ramp / slow-motion / time-lapse / none

2) Subjects & materials
- subjects: object classes, shape, scale relative to frame
- materials/textures: metal/wood/glass/fabric/skin/other; glossy/satin/matte; micro-relief/facets
- dominant_colors (hex), reflectivity/translucency

3) Composition & framing
- shot_size: extreme close-up / close-up / medium / wide / extreme wide
- techniques: rule of thirds, symmetry, leading lines, negative space, perspective/parallax
- depth_of_field: shallow/medium/deep; bokeh character (disc shape, cat's-eye, onion-ring)

4) Optics & focus
- estimated focal_length_mm and FOV (fov_deg)
- aperture (T or f), focus_distance_m, focus pulls (with timecodes)
- distortion (barrel/pincushion/mustache), vignetting, chromatic aberration

5) Camera motion (video/frames)
- motion types: pan/tilt/dolly/truck/pedestal/crane/arc/roll/zoom/handheld/steadicam/gimbal/drone
- trajectory character: linear/arc/orbit/spiral; direction vs subject
- speed & acceleration (m/s or deg/s), easing curves (linear/easeIn/easeOut/easeInOut/bezier)
- micro-jitter: amplitude (px or deg), frequency (Hz), stabilization (optical/digital/none), crop%
- global 2D motion model (translation/rotation/scale) if derivable

6) Lighting & environment
- lighting rig: list sources (key/fill/rim/practical/ambient/bounce)
  For each source: position (x,y,z relative to subject/camera), distance (m), size (m),
  modifier (softbox/umbrella/grid/flag/diffusion/gobo/none), softness, relative intensity,
  color_temp_K, tint (G/M), incidence angle (deg), notable reflections/speculars
- environment: interior/exterior, time of day, weather, dominant reflectors/windows/neon, reflections/caustics

7) Color, tone & post
- grading/look: contrast (low/mid/high), saturation (low/mid/high),
  hue shifts in shadows/mids/highlights, overall gamma curve
- white_balance_K, LUT/ACES-like workflow (if evident), grain/noise, halation/bloom,
  sharpening, compression artifacts, other VFX/SFX

8) Editing & rhythm (if cuts exist)
- transitions: cut/dissolve/whip/match/speed-ramp/freeze/overlay
- cut durations, sync to beats/accents, viewer guidance intent

9) Audio (if present)
- rhythm/beats/SFX, sync to keyframes or motion

10) Ambiguities & confidence
- list all assumptions and per-section confidence scores in [0..1]

=== OUTPUT (A): Human-Readable Report ===
- One-paragraph scene synopsis.
- Bullet details for sections 1–9 with timecodes where applicable. Short, precise sentences.

=== OUTPUT (B): Machine-Readable JSON (STRICT SCHEMA) ===
{
  "media": {
    "type": "image|video|frames",
    "duration_s": null | number,
    "fps": null | number,
    "resolution_px": {"w": number, "h": number},
    "aspect_ratio": "string",
    "playback_speed_anomalies": ["none" | "speed_ramp" | "slow_motion" | "time_lapse"]
  },
  "subjects": [
    {
      "name": "string",
      "class": "string",
      "approx_scale_in_frame": "XS|S|M|L|XL",
      "materials": [{"type":"metal|glass|gem|wood|fabric|skin|other","finish":"glossy|satin|matte","notes":"string"}],
      "dominant_colors_hex": ["#RRGGBB", "..."]
    }
  ],
  "composition": {
    "shot_size": "ECU|CU|MCU|MS|MLS|WS|EWS",
    "framing": {"rule_of_thirds": true, "symmetry": false, "leading_lines": true, "negative_space":"low|mid|high"},
    "depth_of_field": "shallow|medium|deep",
    "bokeh_notes": "string"
  },
  "optics": {
    "focal_length_mm": null | number,
    "fov_deg": null | number,
    "aperture_T": null | number,
    "focus_distance_m": null | number,
    "focus_pulls": [{"t_s": number, "to_m": number}],
    "distortion": "none|barrel|pincushion|mustache",
    "vignetting": "none|low|mid|high",
    "chromatic_aberration": "none|low|mid|high"
  },
  "camera_motion": {
    "support": "tripod|handheld|steadicam|gimbal|dolly|crane|drone|mixed",
    "global_model_2d": {"translation_px_per_s":[number, number], "rotation_deg_per_s": number, "scale_change_per_s": number},
    "stabilization": {"type":"none|optical|digital","crop_pct": null | number},
    "micro_jitter": {"amplitude_px": null | number, "frequency_hz": null | number},
    "keyframes": [
      {
        "t_s": number,
        "type": ["pan","tilt","dolly","truck","pedestal","crane","arc","roll","zoom"],
        "position_m": {"x": null | number, "y": null | number, "z": null | number},
        "rotation_euler_deg": {"yaw": number, "pitch": number, "roll": number},
        "distance_to_subject_m": null | number,
        "focal_length_mm": null | number,
        "fov_deg": null | number,
        "focus_distance_m": null | number,
        "ease": "linear|easeIn|easeOut|easeInOut|bezier",
        "duration_s": null | number,
        "notes": "string"
      }
    ]
  },
  "lighting": {
    "environment": {"type":"interior|exterior","time_of_day":"string","weather":"string","reflections":"string"},
    "lights": [
      {
        "role":"key|fill|rim|back|practical|ambient|bounce",
        "position_m": {"x": null | number, "y": null | number, "z": null | number},
        "distance_m": null | number,
        "size_m": null | number,
        "modifier":"none|softbox|umbrella|grid|flag|diffusion|gobo",
        "softness":"hard|semi|soft",
        "intensity_rel": "low|mid|high",
        "color_temp_K": null | number,
        "tint":"string|null",
        "angle_deg": null | number,
        "notes":"string"
      }
    ]
  },
  "color_and_post": {
    "white_balance_K": null | number,
    "look": {"contrast":"low|mid|high","saturation":"low|mid|high","shadows_hue":"string","mids_hue":"string","highlights_hue":"string"},
    "grain_noise":"none|low|mid|high",
    "halation_bloom":"none|low|mid|high",
    "sharpness":"soft|neutral|crisp",
    "compression_artifacts":"none|low|mid|high",
    "vfx_sfx":"string"
  },
  "editing": {
    "cuts":[{"at_s": number, "type":"cut|dissolve|whip|match|freeze|speed_ramp"}],
    "music_sync":{"beats_s":[number], "notes":"string"}
  },
  "uncertainty": [{"section":"string","assumption":"string","confidence": number}],
  "confidence_overall": number
}

=== OUTPUT (C): Transfer-Prompt for the Generator ===
1) Write a concise scene description for a NEW subject, keeping:
   — the same camera motion (per keyframes),
   — the same lighting rig (roles, approximate positions, softness, color temps),
   — the same frame parameters (fps, duration, aspect, focal/FOV, DOF),
   — the same edit tempo/beat sync (if present).

2) Provide a machine block:
{
  "apply_to_subject": "{{NEW_SUBJECT}}",
  "duration_s": <from media.duration_s>,
  "fps": <from media.fps>,
  "camera_path": <camera_motion.keyframes>,
  "camera_support": <camera_motion.support>,
  "stabilization": <camera_motion.stabilization>,
  "lens_mm": <optics.focal_length_mm>,
  "aperture_T": <optics.aperture_T>,
  "focus_distance_m": <optics.focus_distance_m>,
  "lighting_rig": <lighting.lights>,
  "color_look": <color_and_post.look>,
  "white_balance_K": <color_and_post.white_balance_K>,
  "notes": "replicate motion/lighting exactly; scale distances if subject size differs"
}

3) If critical parameters are unknown, set null and briefly explain assumptions.

QUALITY CHECKS (before returning):
- Every keyframe has t_s and at least one of: rotation_euler_deg, position_m, distance_to_subject_m, focal_length_mm/FOV.
- Handheld implies non-zero micro_jitter.
- At least one light marked as key (or explain dominant source).
- Color temperature provided (K) or null with reason.
- Every major section includes a confidence score [0..1].

TONE
Factual, concise, reproducible. No stylistic flourishes. Use standard terminology.
"""

    def _download_media(self, url: str, max_retries: int = 3) -> Tuple[bytes, str]:
        """
        Скачивает медиа по URL и возвращает данные + MIME type
        Использует заголовки браузера и retry логику для обхода блокировок
        """
        from urllib.parse import urlparse

        # Формируем правильный Referer из домена URL
        parsed_url = urlparse(url)
        referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"

        # Разные User-Agent для попыток
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

        # Retry логика с разными подходами (БЕЗ Sec-Fetch заголовков!)
        last_error = None
        for attempt in range(max_retries):
            try:
                # Пробуем разные наборы заголовков
                if attempt == 0:
                    # Первая попытка: минимальные заголовки без Referer
                    headers = {
                        'User-Agent': user_agents[0],
                        'Accept': '*/*'
                    }
                elif attempt == 1:
                    # Вторая попытка: с Referer
                    headers = {
                        'User-Agent': user_agents[1],
                        'Accept': '*/*',
                        'Referer': referer
                    }
                else:
                    # Третья попытка: полный набор но без Sec-Fetch
                    headers = {
                        'User-Agent': user_agents[2],
                        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8,video/*,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Referer': referer,
                        'Connection': 'keep-alive'
                    }

                response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                response.raise_for_status()
                content_type = response.headers.get('content-type', '')
                return response.content, content_type
            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response.status_code == 403 and attempt < max_retries - 1:
                    # Для 403 пробуем следующий подход
                    continue
                # Для других ошибок HTTP сразу пробрасываем
                if e.response.status_code != 403:
                    raise
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    continue
                raise

        # Если все попытки не удались, пробрасываем последнюю ошибку
        if last_error:
            raise last_error

    def _is_video(self, content_type: str, file_path: Optional[str] = None) -> bool:
        """Определяет, является ли медиа видео"""
        video_mimes = ['video/', 'application/x-mpegURL']
        video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v']
        if content_type:
            if any(mime in content_type for mime in video_mimes):
                return True
        if file_path:
            path_str = str(file_path).lower()
            if any(path_str.endswith(ext) for ext in video_extensions):
                return True
            mime, _ = mimetypes.guess_type(file_path)
            if mime and 'video' in mime:
                return True
        return False

    def _extract_video_frames(self, video_data: bytes, max_frames: int = 8) -> List[bytes]:
        """Извлекает ключевые кадры из видео для анализа"""
        try:
            # Сохраняем временный файл
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_file.write(video_data)
                tmp_path = tmp_file.name

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            # Выбираем кадры равномерно по длительности
            frames_to_extract = min(max_frames, max(1, int(total_frames * (max_frames / total_frames))))
            frame_interval = max(1, total_frames // max_frames)

            extracted_frames = []
            frame_indices = []

            for i in range(0, total_frames, frame_interval):
                if len(extracted_frames) >= max_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # Конвертируем BGR в RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Кодируем в JPEG
                    img = Image.fromarray(frame_rgb)
                    buffer = BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    extracted_frames.append(buffer.getvalue())
                    frame_indices.append(i)

            cap.release()

            # Удаляем временный файл
            try:
                os.unlink(tmp_path)
            except:
                pass

            return extracted_frames

        except Exception as e:
            print(f"Ошибка извлечения кадров: {e}")
            return []

    def _encode_media_to_base64(self, media_data: bytes) -> str:
        """Кодирует медиа в base64"""
        return base64.b64encode(media_data).decode('utf-8')

    def _analyze_with_openai(self, media_data: bytes, content_type: str, is_video: bool) -> Dict:
        """Анализ через OpenAI GPT-4o (поддерживает изображения и видео)"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key не найден")

        analysis_prompt = self._get_analysis_prompt()

        # Для видео извлекаем несколько кадров
        if is_video:
            frames = self._extract_video_frames(media_data, max_frames=8)
            if not frames:
                raise ValueError("Не удалось извлечь кадры из видео")

            # Анализируем несколько кадров для понимания динамики
            content_parts = [{
                "type": "text",
                "text": analysis_prompt
            }]

            for frame in frames[:4]:  # Используем первые 4 кадра для анализа
                base64_frame = self._encode_media_to_base64(frame)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_frame}"
                    }
                })
        else:
            base64_image = self._encode_media_to_base64(media_data)

            # Определяем формат для OpenAI
            if 'png' in content_type or 'image/png' in content_type:
                media_format = "png"
            elif 'jpeg' in content_type or 'jpg' in content_type:
                media_format = "jpeg"
            elif 'gif' in content_type:
                media_format = "gif"
            elif 'webp' in content_type:
                media_format = "webp"
            else:
                media_format = "jpeg"

            content_parts = [{
                "type": "text",
                "text": analysis_prompt
            }, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{media_format};base64,{base64_image}"
                }
            }]

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.preferred_model,
            "messages": [
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            "max_tokens": 4000,  # Увеличено для детального анализа с камерой, освещением и т.д.
            "temperature": 0.3
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120  # Увеличен timeout для детального анализа
        )
        response.raise_for_status()

        result = response.json()
        analysis_text = result['choices'][0]['message']['content']

        # Попытка извлечь JSON из ответа
        try:
            # Ищем JSON в тексте
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                analysis_json = json.loads(analysis_text[json_start:json_end])
            else:
                # Если JSON не найден, создаем структурированный объект из текста
                analysis_json = {"raw_analysis": analysis_text}
        except json.JSONDecodeError:
            # Fallback: создаем структурированный ответ
            analysis_json = {
                "raw_analysis": analysis_text,
                "format": "text_response"
            }

        return analysis_json

    def _analyze_with_gemini(self, media_data: bytes, content_type: str, is_video: bool) -> Dict:
        """Анализ через Google Gemini (лучше работает с видео)"""
        if not self.gemini_api_key:
            raise ValueError("Gemini API key не найден")

        # Определяем MIME type
        if is_video:
            if 'mp4' in content_type.lower():
                mime_type = "video/mp4"
            elif 'webm' in content_type.lower():
                mime_type = "video/webm"
            elif 'mov' in content_type.lower():
                mime_type = "video/quicktime"
            else:
                mime_type = "video/mp4"

            # Для видео используем Gemini File API (поддерживает видео напрямую)
            return self._analyze_video_with_gemini_file_api(media_data, mime_type)
        else:
            # Для изображений используем inline base64
            mime_type = content_type or "image/jpeg"
            base64_media = self._encode_media_to_base64(media_data)

            # Используем тот же универсальный промпт что и для OpenAI
            analysis_prompt = self._get_analysis_prompt()

            headers = {
                "Content-Type": "application/json"
            }

            payload = {
                "contents": [{
                    "parts": [
                        {
                            "text": analysis_prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_media
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 4000  # Увеличено для детального анализа
                }
            }

            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.gemini_api_key}",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            analysis_text = result['candidates'][0]['content']['parts'][0]['text']

            # Извлекаем JSON из ответа
            try:
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    analysis_json = json.loads(analysis_text[json_start:json_end])
                else:
                    analysis_json = {"raw_analysis": analysis_text}
            except json.JSONDecodeError:
                analysis_json = {
                    "raw_analysis": analysis_text,
                    "format": "text_response"
                }

            return analysis_json

    def _analyze_video_with_gemini_file_api(self, video_data: bytes, mime_type: str) -> Dict:
        """Загружает видео через Gemini File API и анализирует"""
        # Шаг 1: Загружаем файл через File API
        upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.gemini_api_key}"

        headers = {
            'X-Goog-Upload-Protocol': 'multipart'
        }

        files = {
            'metadata': ('metadata', json.dumps({
                "file": {
                    "display_name": "video_analysis"
                }
            }), 'application/json; charset=UTF-8'),
            'file': ('video_analysis.mp4', video_data, mime_type)
        }

        response = requests.post(upload_url, headers=headers, files=files, timeout=120)
        if response.status_code >= 400:
            raise ValueError(f"Gemini File API error {response.status_code}: {response.text}")
        file_metadata = response.json()
        file_uri = file_metadata.get('file', {}).get('uri')

        if not file_uri:
            raise ValueError("Не удалось загрузить видео в Gemini File API")

        # Шаг 2: Ждем пока файл обработается (опционально, можно пропустить для быстрых запросов)
        import time
        time.sleep(2)  # Небольшая задержка для обработки файла

        # Шаг 3: Анализируем видео через generateContent API
        analysis_prompt = self._get_analysis_prompt()

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": analysis_prompt
                    },
                    {
                        "file_data": {
                            "mime_type": mime_type,
                            "file_uri": file_uri
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2000
            }
        }

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.gemini_api_key}",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        analysis_text = result['candidates'][0]['content']['parts'][0]['text']

        # Удаляем файл после анализа (опционально)
        try:
            delete_url = f"https://generativelanguage.googleapis.com/v1beta/{file_uri}?key={self.gemini_api_key}"
            requests.delete(delete_url, timeout=10)
        except:
            pass  # Игнорируем ошибки удаления

        # Извлекаем JSON из ответа
        try:
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                analysis_json = json.loads(analysis_text[json_start:json_end])
            else:
                analysis_json = {"raw_analysis": analysis_text}
        except json.JSONDecodeError:
            analysis_json = {
                "raw_analysis": analysis_text,
                "format": "text_response"
            }

        return analysis_json

    def analyze(
        self,
        media_source: Union[str, Path, bytes],
        content_type: Optional[str] = None,
        use_gemini: bool = False
    ) -> Dict:
        """
        Анализирует медиа и возвращает структурированное описание

        Args:
            media_source: URL, путь к файлу или bytes медиа
            content_type: MIME type (опционально)
            use_gemini: Использовать Gemini вместо OpenAI

        Returns:
            Dict со структурированным описанием сцены
        """
        # Определяем тип источника и загружаем данные
        if isinstance(media_source, bytes):
            media_data = media_source
            is_video = self._is_video(content_type or "")
        elif isinstance(media_source, (str, Path)):
            source_str = str(media_source)
            # Проверяем, URL это или путь к файлу
            if source_str.startswith(('http://', 'https://')):
                media_data, detected_content_type = self._download_media(source_str)
                content_type = content_type or detected_content_type
            else:
                # Локальный файл
                file_path = Path(source_str)
                if not file_path.exists():
                    raise FileNotFoundError(f"Файл не найден: {file_path}")
                media_data = file_path.read_bytes()
                content_type = content_type or mimetypes.guess_type(str(file_path))[0]

            is_video = self._is_video(content_type or "", str(media_source))
        else:
            raise ValueError(f"Неподдерживаемый тип источника: {type(media_source)}")

        # Выбираем модель для анализа
        analysis_result = None
        model_used = None
        if use_gemini and self.gemini_api_key:
            try:
                analysis_result = self._analyze_with_gemini(media_data, content_type or "", is_video)
                model_used = "gemini-1.5-pro"
            except Exception as gemini_error:
                print(f"[!] Gemini анализ не удался: {gemini_error}")
                if self.openai_api_key:
                    print("[i] Переключаюсь на OpenAI для анализа")
                    analysis_result = self._analyze_with_openai(media_data, content_type or "", is_video)
                    model_used = self.preferred_model
                else:
                    raise
        if analysis_result is None:
            if self.openai_api_key:
                analysis_result = self._analyze_with_openai(media_data, content_type or "", is_video)
                model_used = self.preferred_model
            else:
                raise ValueError("Не найден API key для анализа (нужен OpenAI или Gemini)")

        # Добавляем метаданные
        analysis_result['metadata'] = {
            'is_video': is_video,
            'content_type': content_type,
            'model_used': model_used or ("gemini-1.5-pro" if (use_gemini and self.gemini_api_key) else self.preferred_model)
        }

        return analysis_result


if __name__ == "__main__":
    # Тестовый пример
    analyzer = VisualAnalyzer()

    # Пример использования
    # result = analyzer.analyze("https://example.com/jewelry.jpg")
    # print(json.dumps(result, indent=2, ensure_ascii=False))

