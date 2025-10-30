"""
Агент 1: Визуальный смысловой анализатор
Анализирует изображения/видео и извлекает детальное описание сцены
"""

import os
import json
import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Any

import mimetypes

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image


class VisualAnalyzer:
    """Анализатор визуального контента для извлечения структурированного описания сцены"""

    def __init__(self, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.preferred_model = "gpt-4o"  # GPT-4o поддерживает vision и видео
        self.logger = logging.getLogger(self.__class__.__name__)
        self._http = self._build_retry_session()

    def _build_retry_session(self) -> requests.Session:
        """Создает requests.Session с повторными попытками для нестабильных сетей"""
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=1.5,
            status_forcelist=(408, 409, 425, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _post_with_retries(
        self,
        url: str,
        *,
        max_attempts: int = 4,
        backoff_factor: float = 2.0,
        raise_for_status: bool = True,
        **kwargs,
    ) -> requests.Response:
        """Делает POST запрос с дополнительной экспоненциальной паузой между попытками"""
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._http.post(url, **kwargs)
                if raise_for_status:
                    response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                last_error = exc
                self.logger.warning(
                    "POST %s failed on attempt %s/%s: %s", url, attempt, max_attempts, exc
                )
                if attempt >= max_attempts:
                    raise
                sleep_seconds = backoff_factor ** (attempt - 1)
                time.sleep(sleep_seconds)
        # Если сюда дошли — бросаем последнюю ошибку
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected empty retry loop state")

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

                response = self._http.get(url, headers=headers, timeout=30, allow_redirects=True)
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

    def _resize_image(self, image: Image.Image, max_dim: int = 1280) -> Image.Image:
        """Масштабирует изображение так, чтобы длинная сторона не превышала max_dim."""
        if max(image.size) <= max_dim:
            return image
        resized = image.copy()
        resized.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return resized

    def _encode_image(self, image: Image.Image, quality: int = 85) -> Tuple[bytes, str]:
        """Конвертирует изображение в JPEG и возвращает байты вместе с MIME типом."""
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue(), "image/jpeg"

    def _prepare_image_bytes(self, image_bytes: bytes, max_dim: int = 1280) -> Tuple[bytes, str]:
        """Готовит изображение для отправки в модель: ресайз и перекодирование."""
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                processed = self._resize_image(img, max_dim=max_dim)
                return self._encode_image(processed)
        except Exception as exc:
            self.logger.debug("Image preprocessing failed, using original bytes: %s", exc)
            return image_bytes, "image/jpeg"

    def _extract_video_frames(
        self,
        video_data: bytes,
        max_frames: int = 8,
    ) -> Tuple[List[bytes], Dict[str, Any]]:
        """Извлекает ключевые кадры и базовые метаданные из видео"""
        try:
            # Сохраняем временный файл
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_file.write(video_data)
                tmp_path = tmp_file.name

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return [], {}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = float(total_frames / fps) if fps and fps > 0 else None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames_to_extract = max_frames if total_frames <= 0 else min(max_frames, total_frames)
            frame_interval = max(1, (total_frames // frames_to_extract) if total_frames else 1)

            sampled_indices: List[int] = []
            extracted_frames: List[bytes] = []
            resized_gray_frames: List[np.ndarray] = []
            brightness_values: List[float] = []

            scale_factor = 1.0
            if width > 0:
                scale_factor = min(1.0, 320.0 / float(width))

            current_index = 0
            while current_index < total_frames and len(extracted_frames) < frames_to_extract:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
                ret, frame = cap.read()
                if not ret:
                    current_index += frame_interval
                    continue

                sampled_indices.append(current_index)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                processed = self._resize_image(img)
                frame_bytes, _ = self._encode_image(processed)
                extracted_frames.append(frame_bytes)

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if scale_factor < 1.0:
                    gray_frame = cv2.resize(
                        gray_frame,
                        (0, 0),
                        fx=scale_factor,
                        fy=scale_factor,
                        interpolation=cv2.INTER_AREA,
                    )
                resized_gray_frames.append(gray_frame)
                brightness_values.append(float(np.mean(gray_frame)))

                current_index += frame_interval

            cap.release()

            # Удаляем временный файл
            try:
                os.unlink(tmp_path)
            except:
                pass

            metadata: Dict[str, Any] = {
                "frame_count": total_frames if total_frames > 0 else None,
                "fps": float(fps) if fps and fps > 0 else None,
                "duration_s": round(duration, 3) if duration else None,
                "resolution_px": {"w": width, "h": height} if width and height else None,
                "sampled_frames": len(extracted_frames),
                "sample_indices": sampled_indices,
            }

            if brightness_values:
                metadata["luma_stats"] = {
                    "avg": round(float(np.mean(brightness_values)), 3),
                    "min": round(float(np.min(brightness_values)), 3),
                    "max": round(float(np.max(brightness_values)), 3),
                    "std": round(float(np.std(brightness_values)), 3),
                }

            motion_metadata = self._estimate_video_motion(resized_gray_frames, metadata.get("fps"))
            if motion_metadata:
                metadata["auto_motion"] = motion_metadata

            return extracted_frames, metadata

        except Exception as e:
            self.logger.error(f"Ошибка извлечения кадров: {e}")
            return [], {}

    def _estimate_video_motion(
        self,
        gray_frames: List[np.ndarray],
        fps: Optional[float],
    ) -> Dict[str, Any]:
        """Пытается оценить движение камеры и объекта на основе оптического потока"""
        if len(gray_frames) < 2:
            return {}

        flow_magnitudes: List[float] = []
        translation_magnitudes: List[float] = []
        rotation_scores: List[float] = []
        direction_vectors: List[Tuple[float, float]] = []

        for prev_gray, next_gray in zip(gray_frames[:-1], gray_frames[1:]):
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    next_gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=25,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
            except cv2.error:
                continue

            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            mean_x = float(np.mean(flow_x))
            mean_y = float(np.mean(flow_y))
            translation_vector = (mean_x, mean_y)
            direction_vectors.append(translation_vector)
            translation_magnitudes.append(float(np.hypot(mean_x, mean_y)))
            flow_magnitudes.append(float(np.mean(np.hypot(flow_x, flow_y))))

            h, w = prev_gray.shape
            ys, xs = np.mgrid[0:h, 0:w]
            rel_x = (xs - (w / 2.0)) / max(w, 1)
            rel_y = (ys - (h / 2.0)) / max(h, 1)
            rotation = float(np.mean((rel_x * flow_y) - (rel_y * flow_x)))
            rotation_scores.append(rotation)

        if not flow_magnitudes:
            return {}

        avg_flow = float(np.mean(flow_magnitudes))
        median_translation = float(np.median(translation_magnitudes)) if translation_magnitudes else 0.0
        median_rotation = float(np.median(rotation_scores)) if rotation_scores else 0.0
        rotation_strength = float(np.median(np.abs(rotation_scores))) if rotation_scores else 0.0

        translation_threshold = 0.08
        translation_high_threshold = 0.18
        rotation_threshold = 0.015

        if median_translation < translation_threshold and rotation_strength > rotation_threshold:
            dominant_motion = "static_camera_subject_rotation"
        elif median_translation >= translation_threshold:
            dominant_motion = "camera_motion"
        else:
            dominant_motion = "mostly_static"

        camera_support_guess = "tripod"
        if dominant_motion == "camera_motion":
            camera_support_guess = "stabilized"
            if median_translation > translation_high_threshold:
                camera_support_guess = "handheld"

        subject_motion = "none"
        if dominant_motion == "static_camera_subject_rotation":
            subject_motion = "axial_rotation"
        elif rotation_strength > rotation_threshold * 1.5:
            subject_motion = "combined"

        avg_direction_x = float(np.mean([vec[0] for vec in direction_vectors])) if direction_vectors else 0.0
        avg_direction_y = float(np.mean([vec[1] for vec in direction_vectors])) if direction_vectors else 0.0
        avg_direction_deg = None
        if avg_direction_x or avg_direction_y:
            avg_direction_deg = round(float(np.degrees(np.arctan2(avg_direction_y, avg_direction_x))), 2)

        motion_summary: Dict[str, Any] = {
            "dominant_motion": dominant_motion,
            "average_flow_px": round(avg_flow, 4),
            "median_translation_px": round(median_translation, 4),
            "median_rotation_score": round(median_rotation, 5),
            "rotation_strength": round(rotation_strength, 5),
            "camera_support_guess": camera_support_guess,
            "subject_motion": subject_motion,
            "sample_pairs": len(flow_magnitudes),
            "confidence": round(min(1.0, len(flow_magnitudes) / 4.0), 3),
        }

        if avg_direction_deg is not None:
            motion_summary["camera_translation_direction_deg"] = avg_direction_deg

        if fps:
            motion_summary["pair_interval_s"] = round(float(1.0 / fps), 4)

        if rotation_strength > rotation_threshold:
            motion_summary["rotation_direction"] = "counter_clockwise" if median_rotation > 0 else "clockwise"

        return motion_summary

    def _format_video_insights(self, metadata: Dict[str, Any]) -> str:
        """Формирует текстовый блок с объективными метриками видео для подсказки модели"""
        if not metadata:
            return ""

        lines: List[str] = []
        fps = metadata.get("fps")
        duration = metadata.get("duration_s")
        resolution = metadata.get("resolution_px")

        if fps:
            lines.append(f"- Estimated FPS: {fps}")
        if duration:
            lines.append(f"- Duration: {duration} sec")
        if resolution:
            lines.append(
                f"- Native resolution: {resolution.get('w')}x{resolution.get('h')}"
            )

        luma_stats = metadata.get("luma_stats")
        if isinstance(luma_stats, dict) and luma_stats:
            lines.append(
                "- Average scene luminance (0-255): "
                f"avg={luma_stats.get('avg')}, min={luma_stats.get('min')}, max={luma_stats.get('max')}"
            )

        auto_motion = metadata.get("auto_motion")
        if isinstance(auto_motion, dict) and auto_motion:
            dominant = auto_motion.get("dominant_motion")
            if dominant:
                lines.append(f"- Dominant motion classification: {dominant}")
            subject_motion = auto_motion.get("subject_motion")
            if subject_motion and subject_motion != "none":
                lines.append(f"- Subject motion heuristic: {subject_motion}")
            support = auto_motion.get("camera_support_guess")
            if support:
                lines.append(f"- Likely camera support: {support}")
            rotation_dir = auto_motion.get("rotation_direction")
            if rotation_dir:
                lines.append(f"- Rotation direction: {rotation_dir}")
            translation_mag = auto_motion.get("median_translation_px")
            if translation_mag is not None:
                lines.append(f"- Median camera translation flow: {translation_mag} px")
            avg_flow = auto_motion.get("average_flow_px")
            if avg_flow is not None:
                lines.append(f"- Average optical-flow magnitude: {avg_flow} px")

        if not lines:
            return ""

        lines.append(
            "- Integrate these metrics into camera_motion and lighting notes. If the camera is static but the subject rotates, spell it out."
        )

        return "VIDEO STRUCTURAL INSIGHTS (objective measurements):\n" + "\n".join(lines)

    def _encode_media_to_base64(self, media_data: bytes) -> str:
        """Кодирует медиа в base64"""
        return base64.b64encode(media_data).decode('utf-8')

    @staticmethod
    def _extract_json_from_text(analysis_text: str) -> Dict:
        """Пытается извлечь JSON из ответа модели."""
        if not analysis_text:
            return {"raw_analysis": "", "format": "empty_response"}

        import re

        # Ищем блоки формата ```json ... ```
        code_block_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
        for match in code_block_pattern.findall(analysis_text):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        json_start = analysis_text.find('{')
        json_end = analysis_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            candidate = analysis_text[json_start:json_end]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        return {
            "raw_analysis": analysis_text,
            "format": "text_response"
        }

    def _analyze_with_openai(self, media_data: bytes, content_type: str, is_video: bool) -> Dict:
        """Анализ через OpenAI GPT-4o (поддерживает изображения и видео)"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key не найден")

        analysis_prompt = self._get_analysis_prompt()
        video_metadata: Dict[str, Any] = {}

        # Для видео извлекаем несколько кадров
        if is_video:
            frames, video_metadata = self._extract_video_frames(media_data, max_frames=8)
            if not frames:
                raise ValueError("Не удалось извлечь кадры из видео")

            insights_block = self._format_video_insights(video_metadata)
            prompt_text = analysis_prompt
            if insights_block:
                prompt_text = (
                    f"{analysis_prompt}\n\n{insights_block}\n"
                    "Ensure the JSON camera_motion keyframes and transfer prompt strictly follow these measured cues."
                )

            # Анализируем несколько кадров для понимания динамики
            content_parts = [{
                "type": "text",
                "text": prompt_text
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
            processed_bytes, detected_type = self._prepare_image_bytes(media_data)
            base64_image = self._encode_media_to_base64(processed_bytes)
            if detected_type:
                content_type = detected_type

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
                media_format = (detected_type.split('/')[-1] if detected_type else "jpeg")

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

        response = self._post_with_retries(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,  # Увеличен timeout для детального анализа
        )
        result = response.json()
        analysis_text = result['choices'][0]['message']['content']

        analysis_json = self._extract_json_from_text(analysis_text)
        if "raw_analysis" not in analysis_json:
            analysis_json["raw_analysis"] = analysis_text

        if video_metadata:
            analysis_json.setdefault("metadata", {})
            analysis_json["metadata"]["auto_video_insights"] = video_metadata

            # Дополнительно пробрасываем эвристику движения, если модель не заполнила блок camera_motion
            auto_motion = video_metadata.get("auto_motion")
            if auto_motion:
                if not analysis_json.get("camera_motion"):
                    analysis_json["camera_motion"] = {
                        "support": auto_motion.get("camera_support_guess"),
                        "dominant_motion": auto_motion.get("dominant_motion"),
                        "notes": "Auto-computed fallback; please validate",
                    }
                # КРИТИЧНО: Добавляем движение объекта, если оно есть
                subject_motion = auto_motion.get("subject_motion")
                if subject_motion and subject_motion != "none":
                    # Добавляем в camera_motion для доступности генератору промптов
                    if isinstance(analysis_json.get("camera_motion"), dict):
                        analysis_json["camera_motion"]["subject_motion"] = subject_motion
                        analysis_json["camera_motion"]["subject_motion_notes"] = (
                            "Object movement detected: " + subject_motion +
                            ". Ensure prompt includes micro-movements or subject rotation."
                        )
                    # Также добавляем на верхний уровень для легкого доступа
                    analysis_json["subject_motion_detected"] = subject_motion

            media_block = analysis_json.setdefault("media", {})
            if video_metadata.get("fps") and not media_block.get("fps"):
                media_block["fps"] = video_metadata["fps"]
            if video_metadata.get("duration_s") and not media_block.get("duration_s"):
                media_block["duration_s"] = video_metadata["duration_s"]
            if video_metadata.get("resolution_px") and not media_block.get("resolution_px"):
                media_block["resolution_px"] = video_metadata["resolution_px"]
            if not media_block.get("type"):
                media_block["type"] = "video"

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
            processed_bytes, detected_type = self._prepare_image_bytes(media_data)
            base64_media = self._encode_media_to_base64(processed_bytes)
            mime_type = detected_type or content_type or "image/jpeg"

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

            response = self._post_with_retries(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.gemini_api_key}",
                headers=headers,
                json=payload,
                timeout=60,
            )
            result = response.json()
            analysis_text = result['candidates'][0]['content']['parts'][0]['text']

            analysis_json = self._extract_json_from_text(analysis_text)
            if "raw_analysis" not in analysis_json:
                analysis_json["raw_analysis"] = analysis_text

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

        response = self._post_with_retries(
            upload_url,
            headers=headers,
            files=files,
            timeout=120,
            raise_for_status=False,
        )
        if response.status_code >= 400:
            raise ValueError(f"Gemini File API error {response.status_code}: {response.text}")
        file_metadata = response.json()
        file_uri = file_metadata.get('file', {}).get('uri')

        if not file_uri:
            raise ValueError("Не удалось загрузить видео в Gemini File API")

        # Шаг 2: Ждем пока файл обработается (опционально, можно пропустить для быстрых запросов)
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

        response = self._post_with_retries(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.gemini_api_key}",
            headers=headers,
            json=payload,
            timeout=120,
        )
        result = response.json()
        analysis_text = result['candidates'][0]['content']['parts'][0]['text']

        # Удаляем файл после анализа (опционально)
        try:
            delete_url = f"https://generativelanguage.googleapis.com/v1beta/{file_uri}?key={self.gemini_api_key}"
            self._http.delete(delete_url, timeout=10)
        except:
            pass  # Игнорируем ошибки удаления

        analysis_json = self._extract_json_from_text(analysis_text)
        if "raw_analysis" not in analysis_json:
            analysis_json["raw_analysis"] = analysis_text

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

