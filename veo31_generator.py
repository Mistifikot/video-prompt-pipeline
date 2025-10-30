"""
Полный workflow для генерации видео в Veo 3.1:
1. Анализ референсного видео (если есть)
2. Анализ референсных изображений (3 шт)
3. Генерация промпта
4. Генерация видео в Veo 3.1
"""

import os
from typing import List, Dict, Optional, Union
from pathlib import Path
from collections import defaultdict
import json
import mimetypes
import requests

# Импорт логирования
try:
    from logger_utils import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item:
            continue
        if isinstance(item, (dict, list)):
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        else:
            key = item if isinstance(item, (str, int, float, bool)) else repr(item)
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered

from pipeline import VideoPromptPipeline
from veo31_client import Veo31Client
from kie_api_client import KieVeoClient
from media_uploader import build_image_uploader


class Veo31VideoGenerator:
    """Полный пайплайн для генерации видео в Veo 3.1 с референсами"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        perplexity_api_key: Optional[str] = None,
        auto_perplexity_polish: Optional[bool] = None,
        kie_api_key: Optional[str] = None,
        prefer_kie_default: Optional[bool] = None
    ):
        """
        Инициализация генератора

        Args:
            openai_api_key: API ключ OpenAI для анализа
            gemini_api_key: API ключ Google для Veo 3.1
        """
        resolved_gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        self.pipeline = VideoPromptPipeline(
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            gemini_api_key=resolved_gemini_key,
            use_gemini_for_analysis=False,
            perplexity_api_key=perplexity_api_key or os.getenv("PPLX_API_KEY"),
            auto_perplexity_polish=auto_perplexity_polish
        )

        self.veo_client: Optional[Veo31Client] = None
        if resolved_gemini_key:
            try:
                self.veo_client = Veo31Client(api_key=resolved_gemini_key)
                print("[OK] Veo 3.1 клиент (Google) инициализирован")
            except Exception as gemini_error:
                print(f"[!] Не удалось инициализировать Veo 3.1 клиента: {gemini_error}")
        else:
            print("[i] GEMINI_API_KEY не найден, пропускаю Veo 3.1 (Google) клиента")

        self.kie_client: Optional[KieVeoClient] = None
        self.use_kie_default = prefer_kie_default
        if self.use_kie_default is None:
            self.use_kie_default = os.getenv("USE_KIE_API", "false").strip().lower() in {"1", "true", "yes", "on"}

        kie_key = kie_api_key or os.getenv("KIE_API_KEY")
        if kie_key:
            try:
                self.kie_client = KieVeoClient(api_key=kie_key)
                print("[OK] Kie.ai клиент инициализирован")
            except Exception as kie_error:
                print(f"[!] Не удалось инициализировать Kie.ai клиента: {kie_error}")

        self.image_uploader = build_image_uploader() if kie_key else None

    def analyze_and_generate_prompt(
        self,
        reference_images: List[Union[str, Path, bytes, Dict]],
        video_reference: Optional[Union[str, Path, bytes, Dict]] = None,
        platform: str = "veo3",
        use_case: str = "product_video",
        polish_with_perplexity: Optional[bool] = None
    ) -> Dict:
        """
        Только анализ референсов и генерация промпта (без генерации видео)
        Логика: сначала видео, потом изображения

        Args:
            reference_images: Список референсных изображений
            video_reference: Опциональное референсное видео
            platform: Платформа (veo3)
            use_case: Тип задачи
            polish_with_perplexity: Полировать промпт через Perplexity

        Returns:
            Dict с анализом и промптом
        """
        if not reference_images:
            raise ValueError("Необходимо минимум одно референсное изображение")

        combined_analysis = {}
        video_analysis = None
        video_meta = None

        # ШАГ 1: Анализ видео (если есть) - ПЕРВЫМ
        if video_reference:
            try:
                logger.info("Шаг 1.1: Анализ видео-референса через OpenAI...")
                resolved_video = self._resolve_media_input(video_reference, "reference_video")
                video_source = resolved_video.get("source")

                if video_source is None:
                    logger.warning("Не удалось получить источник видео-референса")
                else:
                    video_analysis = self.pipeline.analyzer.analyze(
                        media_source=video_source,
                        content_type=resolved_video.get("content_type"),
                        use_gemini=False  # Всегда используем OpenAI
                    )
                    video_meta = {
                        "label": resolved_video.get("label"),
                        "content_type": resolved_video.get("content_type")
                    }
                    metadata_model = video_analysis.get("metadata", {}).get("model_used")
                    video_meta["model_used"] = metadata_model
                    logger.info(f"Видео-референс проанализирован успешно через {metadata_model}")
            except Exception as video_error:
                logger.error(f"Ошибка при анализе видео-референса: {video_error}", exc_info=True)
                video_analysis = {"error": str(video_error)}
                video_meta = {"error": str(video_error)}

        # ШАГ 2: Анализ изображений - ВТОРЫМ
        logger.info("Шаг 1.2: Анализ изображений через OpenAI...")
        image_analysis_payloads = []
        prepared_images_for_generation: List[Dict] = []

        for idx, image_entry in enumerate(reference_images, start=1):
            try:
                resolved = self._resolve_media_input(image_entry, f"reference_image_{idx}")
                source = resolved.get("source")

                if source is None:
                    logger.warning(f"Изображение {idx}: нет данных")
                    image_analysis_payloads.append({
                        "analysis": {"error": "No image data provided"},
                        "meta": resolved
                    })
                    continue

                content_type = resolved.get("content_type") or "image/jpeg"
                logger.info(f"Анализ изображения {idx}: {resolved.get('label')}")

                analysis = self.pipeline.analyzer.analyze(
                    media_source=source,
                    content_type=content_type,
                    use_gemini=False  # Всегда используем OpenAI
                )

                image_analysis_payloads.append({
                    "analysis": analysis,
                    "meta": {
                        "label": resolved.get("label"),
                        "content_type": content_type
                    }
                })

                prepared_images_for_generation.append({
                    "data": source if isinstance(source, bytes) else None,
                    "url": source if isinstance(source, str) else None,
                    "path": source if isinstance(source, Path) else None,
                    "content_type": content_type,
                    "filename": resolved.get("label")
                })

            except Exception as image_error:
                logger.error(f"Ошибка анализа изображения {idx}: {image_error}", exc_info=True)
                image_analysis_payloads.append({
                    "analysis": {"error": str(image_error)},
                    "meta": {"label": f"reference_image_{idx}", "error": True}
                })

        # Агрегируем результаты анализа изображений
        image_aggregate = self._aggregate_image_analyses(image_analysis_payloads)
        combined_analysis = dict(image_aggregate)

        # Объединяем с анализом видео если есть
        if video_analysis:
            combined_analysis = self._merge_analyses(image_aggregate, video_analysis, video_meta)
            combined_analysis["video_analysis"] = video_analysis
        if video_meta:
            combined_analysis.setdefault("video_reference_info", video_meta)

        combined_analysis["image_analysis_summary"] = image_aggregate.get("reference_summary")
        combined_analysis["image_reference_notes"] = image_aggregate.get("reference_notes")
        combined_analysis["image_sources"] = image_aggregate.get("sources")

        # ШАГ 3: Генерация промпта
        logger.info("Шаг 2: Генерация промпта...")

        # Проверяем, есть ли структурированные данные для генерации промпта
        has_main_objects = bool(combined_analysis.get("main_objects"))
        has_materials = bool(combined_analysis.get("materials"))
        has_lighting = bool(combined_analysis.get("lighting"))
        has_raw_analysis = bool(combined_analysis.get("reference_notes") or combined_analysis.get("image_analysis_summary"))

        if not (has_main_objects or has_materials or has_lighting):
            # Если нет структурированных данных, пытаемся использовать текстовые данные
            if has_raw_analysis:
                logger.warning("Нет структурированных данных анализа, используем текстовое описание")
                combined_analysis["raw_analysis"] = combined_analysis.get("image_analysis_summary", "") or " ".join(combined_analysis.get("reference_notes", []))
            else:
                logger.error("Анализ не вернул данных для генерации промпта!")
                return {
                    "step1_analysis": combined_analysis,
                    "step2_prompt": "Ошибка: Анализ изображений не вернул структурированных данных. Проверьте доступность OpenAI API и формат изображений.",
                    "prompt": "Ошибка: Анализ изображений не вернул структурированных данных. Проверьте доступность OpenAI API и формат изображений.",
                    "ready_for_editing": True
                }

        try:
            prompt_result = self.pipeline.generator.generate(
                scene_description=combined_analysis,
                platform=platform,
                use_case=use_case,
                polish_with_perplexity=polish_with_perplexity
            )

            generated_prompt = prompt_result["prompt"]
            logger.info("Промпт сгенерирован успешно")
        except Exception as prompt_error:
            logger.error(f"Ошибка генерации промпта: {prompt_error}", exc_info=True)
            generated_prompt = f"Ошибка генерации промпта: {str(prompt_error)}"

        return {
            "step1_analysis": combined_analysis,
            "step2_prompt": generated_prompt,
            "prompt": generated_prompt,  # Дублируем для совместимости
            "ready_for_editing": True
        }

    def generate_from_references(
        self,
        reference_images: List[Union[str, Path, bytes, Dict]],
        video_reference: Optional[Union[str, Path, bytes, Dict]] = None,
        platform: str = "veo3",
        use_case: str = "product_video",
        duration_seconds: int = 5,
        aspect_ratio: str = "16:9",
        quality: str = "high",
        additional_prompt: Optional[str] = None,
        polish_with_perplexity: Optional[bool] = None,
        prefer_kie_api: Optional[bool] = None
    ) -> Dict:
        """
        Полный цикл: анализ → промпт → генерация видео

        Args:
            reference_images: Список из 3 референсных изображений
            video_reference: Опциональное референсное видео для анализа стиля
            platform: Платформа (veo3)
            use_case: Тип задачи
            duration_seconds: Длительность видео
            aspect_ratio: Соотношение сторон
            quality: Качество (для Google Veo). Для Kie.ai преобразуется в model:
                "high" -> "veo3", иначе -> "veo3_fast"
            additional_prompt: Дополнительные инструкции к промпту
            prefer_kie_api: Принудительно использовать Kie.ai API (если доступен)

        Returns:
            Dict с результатом генерации
        """
        results = {
            "step1_analysis": None,
            "step2_prompt": None,
            "step3_video_generation": None
        }

        # Шаг 1: Анализ референсов
        logger.info("Шаг 1: Анализ референсных материалов...")

        if not reference_images:
            raise ValueError("Необходимо минимум одно референсное изображение")

        image_analysis_payloads = []
        prepared_images_for_generation: List[Dict] = []

        for idx, image_entry in enumerate(reference_images, start=1):
            resolved = self._resolve_media_input(image_entry, f"reference_image_{idx}")
            source = resolved.get("source")
            if source is None:
                image_analysis_payloads.append({"analysis": {"error": "No image data provided"}, "meta": resolved})
                continue

            content_type = resolved.get("content_type") or "image/jpeg"
            try:
                logger.debug(f"Анализ изображения {idx}: {resolved.get('label')}")
                analysis = self.pipeline.analyzer.analyze(
                    media_source=source,
                    content_type=content_type,
                    use_gemini=False
                )
            except Exception as image_error:
                logger.error(f"Ошибка анализа изображения {idx}: {image_error}", exc_info=True)
                analysis = {"error": str(image_error)}

            image_analysis_payloads.append({
                "analysis": analysis,
                "meta": {
                    "label": resolved.get("label"),
                    "content_type": content_type
                }
            })

            prepared_images_for_generation.append({
                "data": source if isinstance(source, bytes) else None,
                "url": source if isinstance(source, str) else None,
                "path": source if isinstance(source, Path) else None,
                "content_type": content_type,
                "filename": resolved.get("label")
            })

        image_aggregate = self._aggregate_image_analyses(image_analysis_payloads)

        combined_analysis = dict(image_aggregate)
        video_analysis = None
        video_meta = None

        if video_reference:
            try:
                print("Анализ видео-референса...")
                resolved_video = self._resolve_media_input(video_reference, "reference_video")
                video_meta = {
                    "label": resolved_video.get("label"),
                    "content_type": resolved_video.get("content_type")
                }
                video_analysis = self.pipeline.analyzer.analyze(
                    media_source=resolved_video.get("source"),
                    content_type=resolved_video.get("content_type"),
                    use_gemini=False
                )
                metadata_model = video_analysis.get("metadata", {}).get("model_used")
                video_meta["fallback_used"] = metadata_model != "gemini-1.5-pro"
                combined_analysis = self._merge_analyses(image_aggregate, video_analysis, video_meta)
                print("Видео-референс проанализирован успешно")
            except Exception as video_error:
                print(f"Ошибка при анализе видео-референса (продолжаем без него): {video_error}")

        combined_analysis["image_analysis_summary"] = image_aggregate.get("reference_summary")
        combined_analysis["image_reference_notes"] = image_aggregate.get("reference_notes")
        combined_analysis["image_sources"] = image_aggregate.get("sources")
        if video_analysis:
            combined_analysis["video_analysis"] = video_analysis
        if video_meta:
            combined_analysis.setdefault("video_reference_info", video_meta)

        results["step1_analysis"] = combined_analysis

        # Шаг 2: Генерация промпта
        logger.info("Шаг 2: Генерация промпта...")

        # Проверяем, есть ли структурированные данные для генерации промпта
        has_main_objects = bool(combined_analysis.get("main_objects"))
        has_materials = bool(combined_analysis.get("materials"))
        has_lighting = bool(combined_analysis.get("lighting"))
        has_raw_analysis = bool(combined_analysis.get("reference_notes") or combined_analysis.get("image_analysis_summary"))

        if not (has_main_objects or has_materials or has_lighting):
            # Если нет структурированных данных, пытаемся использовать текстовые данные
            if has_raw_analysis:
                logger.warning("Нет структурированных данных анализа, используем текстовое описание")
                # Создаем минимальную структуру из текстовых данных
                combined_analysis["raw_analysis"] = combined_analysis.get("image_analysis_summary", "") or " ".join(combined_analysis.get("reference_notes", []))
            else:
                logger.error("Анализ не вернул данных для генерации промпта!")
                results["step2_prompt"] = "Ошибка: Анализ изображений не вернул структурированных данных. Проверьте доступность OpenAI API и формат изображений."
                generated_prompt = None

        if not results.get("step2_prompt"):
            try:
                prompt_result = self.pipeline.generator.generate(
                    scene_description=combined_analysis,
                    platform=platform,
                    use_case=use_case,
                    polish_with_perplexity=polish_with_perplexity
                )

                generated_prompt = prompt_result["prompt"]

                # Добавляем дополнительные инструкции если есть
                if additional_prompt:
                    generated_prompt = f"{generated_prompt}. {additional_prompt}"

                results["step2_prompt"] = generated_prompt
                logger.info("Промпт сгенерирован успешно")
            except Exception as prompt_error:
                logger.error(f"Ошибка генерации промпта: {prompt_error}", exc_info=True)
                results["step2_prompt"] = f"Ошибка генерации промпта: {str(prompt_error)}"
                generated_prompt = None

        # Шаг 3: Генерация видео
        logger.info("Шаг 3: Генерация видео...")

        generated_prompt = results.get("step2_prompt")
        if not generated_prompt or generated_prompt.startswith("Ошибка"):
            results["step3_video_generation"] = {
                "status": "failed",
                "error": "Не удалось сгенерировать промпт",
                "provider": None
            }
            return results

        prefer_kie = prefer_kie_api if prefer_kie_api is not None else self.use_kie_default
        generation_attempts: List[Dict] = []
        final_generation: Optional[Dict] = None
        google_error: Optional[str] = None

        def wrap_google(payload: Dict) -> Dict:
            return {
                "provider": "google",
                "status": payload.get("status", "unknown"),
                "task_id": payload.get("task_id"),
                "video_url": payload.get("video_url"),
                "raw": payload
            }

        def wrap_kie(payload: Dict) -> Dict:
            code = payload.get("code")
            status = "submitted" if code == 200 or (isinstance(code, int) and 200 <= code < 300) else "error"
            return {
                "provider": "kie.ai",
                "status": status,
                "task_id": payload.get("data", {}).get("taskId") or payload.get("taskId"),
                "raw": payload
            }

        run_google = self.veo_client is not None and not prefer_kie
        if run_google:
            try:
                google_payload = self.veo_client.generate_video_from_images(
                    prompt=generated_prompt,
                    reference_images=prepared_images_for_generation,
                    video_reference=video_reference,
                    duration_seconds=duration_seconds,
                    aspect_ratio=aspect_ratio,
                    quality=quality
                )
                final_generation = wrap_google(google_payload)
                generation_attempts.append({"provider": "google", "status": final_generation["status"]})
                if final_generation.get("status") == "processing":
                    logger.info(f"Видео генерируется (Google), task_id: {final_generation.get('task_id')}")
            except Exception as google_exception:
                google_error = str(google_exception)
                logger.error(f"Google Veo генерация не удалась: {google_error}", exc_info=True)
                generation_attempts.append({"provider": "google", "status": "failed", "error": google_error})

        should_try_kie = self.kie_client is not None and (prefer_kie or final_generation is None or final_generation.get("status") == "failed")

        if should_try_kie:
            kie_image_urls: List[str] = []
            upload_notes: List[str] = []

            for item in prepared_images_for_generation:
                existing_url = item.get("url")
                if isinstance(existing_url, str) and existing_url.startswith(("http://", "https://")):
                    kie_image_urls.append(existing_url)
                    continue

                if not self.image_uploader:
                    upload_notes.append(f"{item.get('filename') or 'image'}: нет публичного URL и uploader не настроен")
                    continue

                data = item.get("data")
                content_type = item.get("content_type") or "image/jpeg"

                if data is None and item.get("path"):
                    try:
                        data = Path(item["path"]).read_bytes()
                    except Exception as fs_err:
                        upload_notes.append(f"{item.get('filename') or 'image'}: не удалось прочитать файл ({fs_err})")
                        data = None

                if data is None and item.get("url"):
                    try:
                        resp = self._download_media_with_headers(item["url"])
                        data = resp.content
                        content_type = item.get("content_type") or resp.headers.get("content-type") or content_type
                    except requests.RequestException as req_err:
                        upload_notes.append(f"{item.get('filename') or 'image'}: не удалось скачать для загрузки ({req_err})")
                        data = None

                if not data:
                    upload_notes.append(f"{item.get('filename') or 'image'}: нет доступных данных для загрузки")
                    continue

                try:
                    uploaded_url = self.image_uploader.upload_image(data, item.get("filename"), content_type)
                    kie_image_urls.append(uploaded_url)
                    upload_notes.append(f"{item.get('filename') or 'image'} загружено -> {uploaded_url}")
                except Exception as upload_error:
                    upload_notes.append(f"{item.get('filename') or 'image'}: ошибка загрузки ({upload_error})")

            include_images = len(kie_image_urls) == len(prepared_images_for_generation) and len(kie_image_urls) > 0

            try:
                generation_type = "REFERENCE_2_VIDEO" if include_images else "TEXT_2_VIDEO"

                # Всегда используем veo3_fast по умолчанию
                kie_model = os.getenv("KIE_VEO_MODEL")
                if not kie_model:
                    kie_model = "veo3_fast"

                kie_payload = self.kie_client.generate_video(
                    prompt=generated_prompt,
                    image_urls=kie_image_urls if include_images else None,
                    model=kie_model,
                    aspect_ratio=aspect_ratio,
                    generation_type=generation_type
                )
                final_generation = wrap_kie(kie_payload)
                attempt_record = {"provider": "kie.ai", "status": final_generation["status"]}

                if upload_notes:
                    final_generation.setdefault("notes", []).extend(upload_notes)

                if not include_images and prepared_images_for_generation:
                    attempt_record["note"] = "Изображения не переданы (нет публичных URL)"
                    final_generation.setdefault("notes", []).append("Images skipped: no external URLs available")

                generation_attempts.append(attempt_record)
                if final_generation.get("status") == "submitted":
                    print(f"Видео задача отправлена в Kie.ai, task_id: {final_generation.get('task_id')}")
            except Exception as kie_exception:
                generation_attempts.append({"provider": "kie.ai", "status": "failed", "error": str(kie_exception)})
                print(f"[!] Kie.ai генерация не удалась: {kie_exception}")

        if final_generation is None:
            failure_payload: Dict[str, Union[str, List[Dict]]] = {
                "status": "failed",
                "provider": None,
                "details": generation_attempts
            }
            if google_error:
                failure_payload["message"] = f"Google Veo: {google_error}"
            else:
                failure_payload["message"] = "Video generation failed"
            failure_payload["error"] = failure_payload.get("message")
            results["step3_video_generation"] = failure_payload
        else:
            if google_error and final_generation.get("provider") == "kie.ai":
                final_generation.setdefault("notes", []).append(f"Google fallback: {google_error}")
            results["step3_video_generation"] = final_generation

        if generation_attempts:
            results["generation_attempts"] = generation_attempts

        return results

    def _aggregate_image_analyses(self, analyses: List[Dict]) -> Dict:
        """Объединяет результаты анализа нескольких изображений"""
        aggregated = {
            "main_objects": [],
            "materials": defaultdict(list),
            "lighting": defaultdict(list),
            "camera_work": defaultdict(list),
            "composition": defaultdict(list),
            "background_surfaces": defaultdict(list),
            "style_mood": defaultdict(list),
            "reference_notes": [],
            "sources": []
        }

        for idx, item in enumerate(analyses, start=1):
            analysis = item.get("analysis") if isinstance(item, dict) else item
            meta = item.get("meta") if isinstance(item, dict) else {}

            if not analysis or analysis.get("error"):
                aggregated["reference_notes"].append(
                    f"Image {idx}: analysis unavailable - {analysis.get('error') if analysis else 'no data'}"
                )
                continue

            label = meta.get("label", f"Image {idx}")
            aggregated["reference_notes"].append(analysis.get("raw_analysis", f"Structured data extracted from {label}"))
            aggregated["sources"].append({
                "label": label,
                "content_type": meta.get("content_type"),
                "model_used": analysis.get("metadata", {}).get("model_used")
            })

            aggregated["main_objects"].extend(analysis.get("main_objects", []))

            for field in ["materials", "lighting", "camera_work", "composition", "background_surfaces", "style_mood"]:
                value = analysis.get(field)
                if isinstance(value, dict):
                    for key, val in value.items():
                        if isinstance(val, list):
                            aggregated[field][key].extend(val)
                        else:
                            aggregated[field][key].append(val)

        # Преобразуем defaultdict обратно в обычные структуры
        for field in ["materials", "lighting", "camera_work", "composition", "background_surfaces", "style_mood"]:
            aggregated[field] = {
                key: _dedupe_preserve_order(values) if isinstance(values, list) else values
                for key, values in aggregated[field].items()
            }

        aggregated["main_objects"] = _dedupe_preserve_order(aggregated["main_objects"])
        aggregated["source_count"] = len(analyses)

        # Создаем резюме
        summary_parts = []
        if aggregated["main_objects"]:
            summary_parts.append(f"Primary subjects: {', '.join(aggregated['main_objects'])}.")
        if aggregated["materials"]:
            summary_parts.append(f"Materials overview: {json.dumps(aggregated['materials'], ensure_ascii=False)}.")
        if aggregated["lighting"]:
            summary_parts.append(f"Lighting cues: {json.dumps(aggregated['lighting'], ensure_ascii=False)}.")
        if aggregated["composition"]:
            summary_parts.append(f"Composition notes: {json.dumps(aggregated['composition'], ensure_ascii=False)}.")
        aggregated["reference_summary"] = " ".join(summary_parts)

        return aggregated

    def _merge_analyses(self, image_analysis: Dict, video_analysis: Dict, video_meta: Optional[Dict] = None) -> Dict:
        """Объединяет анализы изображения и видео

        Приоритеты:
        - Объекты: из изображений (изображения показывают что именно снимать)
        - Движение камеры: из видео (видео показывает как двигать камеру)
        - Освещение: из видео (видео показывает динамику света)
        - Материалы: объединяем (из обоих источников)
        - Композиция: из изображений (статика)
        - Motion dynamics: только из видео
        """
        merged = dict(image_analysis) if image_analysis else {}

        if not video_analysis:
            return merged

        # Объекты: приоритет изображениям (они показывают что именно снимать)
        image_objects = image_analysis.get("main_objects", []) if image_analysis else []
        video_objects = video_analysis.get("main_objects", [])
        # Объединяем, но изображения в начале
        merged["main_objects"] = _dedupe_preserve_order(image_objects + video_objects)

        def merge_field(field_name: str, prefer_video: bool = False):
            image_field = image_analysis.get(field_name, {}) if image_analysis else {}
            video_field = video_analysis.get(field_name, {})

            if prefer_video and video_field:
                # Для видео-приоритетных полей (camera_work, motion) используем видео
                return video_field

            # Для остальных объединяем
            combined = dict(image_field)
            if isinstance(video_field, dict):
                for key, val in video_field.items():
                    if key in combined and isinstance(combined[key], list) and isinstance(val, list):
                        combined[key] = _dedupe_preserve_order(combined[key] + val)
                    else:
                        # Если ключа нет в изображениях или это не список, берем из видео
                        combined[key] = val
            return combined

        # Материалы: объединяем из обоих источников
        merged["materials"] = merge_field("materials")

        # Освещение: приоритет видео (показывает динамику света)
        merged["lighting"] = merge_field("lighting", prefer_video=True) or image_analysis.get("lighting", {})

        # Движение камеры: ТОЛЬКО из видео (это ключевое отличие видео от изображений)
        merged["camera_work"] = video_analysis.get("camera_work", {}) or merge_field("camera_work", prefer_video=True)

        # Композиция: из изображений (статичная композиция)
        merged["composition"] = image_analysis.get("composition", {}) if image_analysis else {}

        # Фон: из изображений
        merged["background_surfaces"] = image_analysis.get("background_surfaces", {}) if image_analysis else {}

        # Движение: ТОЛЬКО из видео
        merged["motion_dynamics"] = video_analysis.get("motion_dynamics", {})

        # Стиль и настроение: объединяем
        merged["style_mood"] = merge_field("style_mood")

        # Добавляем текстовые резюме с явным указанием источников
        notes = list(image_analysis.get("reference_notes", [])) if image_analysis else []
        if video_analysis.get("raw_analysis"):
            notes.append(f"[VIDEO REFERENCE] {video_analysis['raw_analysis']}")

        summary = []
        if image_analysis:
            image_summary = image_analysis.get("reference_summary") or image_analysis.get("raw_analysis")
            if image_summary:
                summary.append(f"[IMAGES] {image_summary}")

        if video_analysis.get("raw_analysis"):
            summary.append(f"[VIDEO] {video_analysis['raw_analysis']}")

        merged["integrated_summary"] = " ".join(summary)
        merged["reference_notes"] = notes

        # Добавляем флаг что использовались оба источника
        merged["has_image_reference"] = bool(image_analysis)
        merged["has_video_reference"] = True

        if video_meta:
            merged["video_reference_info"] = {
                "source": video_meta.get("label"),
                "content_type": video_meta.get("content_type"),
                "fallback_used": video_meta.get("fallback_used")
            }

        return merged

    def generate_video_with_prompt(
        self,
        prompt: str,
        reference_images: List[Union[str, Path, bytes, Dict]],
        video_reference: Optional[Union[str, Path, bytes, Dict]] = None,
        duration_seconds: int = 5,
        aspect_ratio: str = "16:9",
        quality: str = "high",
        prefer_kie_api: Optional[bool] = None
    ) -> Dict:
        """
        Генерация видео с уже готовым промптом (после редактирования)

        Args:
            prompt: Готовый промпт (возможно отредактированный)
            reference_images: Список референсных изображений
            video_reference: Опциональное референсное видео
            duration_seconds: Длительность видео
            aspect_ratio: Соотношение сторон
            quality: Качество
            prefer_kie_api: Принудительно использовать Kie.ai API

        Returns:
            Dict с результатом генерации видео
        """
        # Подготавливаем изображения для генерации
        prepared_images_for_generation: List[Dict] = []

        for idx, image_entry in enumerate(reference_images, start=1):
            resolved = self._resolve_media_input(image_entry, f"reference_image_{idx}")
            source = resolved.get("source")
            if source is None:
                continue

            prepared_images_for_generation.append({
                "data": source if isinstance(source, bytes) else None,
                "url": source if isinstance(source, str) else None,
                "path": source if isinstance(source, Path) else None,
                "content_type": resolved.get("content_type") or "image/jpeg",
                "filename": resolved.get("label")
            })

        # Шаг 3: Генерация видео
        print("Шаг 3: Генерация видео с готовым промптом...")

        prefer_kie = prefer_kie_api if prefer_kie_api is not None else self.use_kie_default
        generation_attempts: List[Dict] = []
        final_generation: Optional[Dict] = None
        google_error: Optional[str] = None

        def wrap_google(payload: Dict) -> Dict:
            return {
                "provider": "google",
                "status": payload.get("status", "unknown"),
                "task_id": payload.get("task_id"),
                "video_url": payload.get("video_url"),
                "raw": payload
            }

        def wrap_kie(payload: Dict) -> Dict:
            code = payload.get("code")
            status = "submitted" if code == 200 or (isinstance(code, int) and 200 <= code < 300) else "error"
            return {
                "provider": "kie.ai",
                "status": status,
                "task_id": payload.get("data", {}).get("taskId") or payload.get("taskId"),
                "raw": payload
            }

        run_google = self.veo_client is not None and not prefer_kie
        if run_google:
            try:
                google_payload = self.veo_client.generate_video_from_images(
                    prompt=prompt,
                    reference_images=prepared_images_for_generation,
                    video_reference=video_reference,
                    duration_seconds=duration_seconds,
                    aspect_ratio=aspect_ratio,
                    quality=quality
                )
                final_generation = wrap_google(google_payload)
                generation_attempts.append({"provider": "google", "status": final_generation["status"]})
                if final_generation.get("status") == "processing":
                    print(f"Видео генерируется (Google), task_id: {final_generation.get('task_id')}")
            except Exception as google_exception:
                google_error = str(google_exception)
                generation_attempts.append({"provider": "google", "status": "failed", "error": google_error})
                print(f"[!] Google Veo генерация не удалась: {google_error}")

        should_try_kie = self.kie_client is not None and (prefer_kie or final_generation is None or final_generation.get("status") == "failed")

        if should_try_kie:
            kie_image_urls: List[str] = []
            upload_notes: List[str] = []

            for item in prepared_images_for_generation:
                existing_url = item.get("url")
                if isinstance(existing_url, str) and existing_url.startswith(("http://", "https://")):
                    kie_image_urls.append(existing_url)
                    continue

                if not self.image_uploader:
                    upload_notes.append(f"{item.get('filename') or 'image'}: нет публичного URL и uploader не настроен")
                    continue

                data = item.get("data")
                content_type = item.get("content_type") or "image/jpeg"

                if data is None and item.get("path"):
                    try:
                        data = Path(item["path"]).read_bytes()
                    except Exception as fs_err:
                        upload_notes.append(f"{item.get('filename') or 'image'}: не удалось прочитать файл ({fs_err})")
                        data = None

                if data is None and item.get("url"):
                    try:
                        resp = self._download_media_with_headers(item["url"])
                        data = resp.content
                        content_type = item.get("content_type") or resp.headers.get("content-type") or content_type
                    except requests.RequestException as req_err:
                        upload_notes.append(f"{item.get('filename') or 'image'}: не удалось скачать для загрузки ({req_err})")
                        data = None

                if not data:
                    upload_notes.append(f"{item.get('filename') or 'image'}: нет доступных данных для загрузки")
                    continue

                try:
                    uploaded_url = self.image_uploader.upload_image(data, item.get("filename"), content_type)
                    kie_image_urls.append(uploaded_url)
                    upload_notes.append(f"{item.get('filename') or 'image'} загружено -> {uploaded_url}")
                except Exception as upload_error:
                    upload_notes.append(f"{item.get('filename') or 'image'}: ошибка загрузки ({upload_error})")

            include_images = len(kie_image_urls) == len(prepared_images_for_generation) and len(kie_image_urls) > 0

            try:
                generation_type = "REFERENCE_2_VIDEO" if include_images else "TEXT_2_VIDEO"

                kie_model = os.getenv("KIE_VEO_MODEL")
                if not kie_model:
                    kie_model = "veo3" if quality.lower() == "high" else "veo3_fast"

                kie_payload = self.kie_client.generate_video(
                    prompt=prompt,
                    image_urls=kie_image_urls if include_images else None,
                    model=kie_model,
                    aspect_ratio=aspect_ratio,
                    generation_type=generation_type
                )
                final_generation = wrap_kie(kie_payload)
                attempt_record = {"provider": "kie.ai", "status": final_generation["status"]}

                if upload_notes:
                    final_generation.setdefault("notes", []).extend(upload_notes)

                if not include_images and prepared_images_for_generation:
                    attempt_record["note"] = "Изображения не переданы (нет публичных URL)"
                    final_generation.setdefault("notes", []).append("Images skipped: no external URLs available")

                generation_attempts.append(attempt_record)
                if final_generation.get("status") == "submitted":
                    print(f"Видео задача отправлена в Kie.ai, task_id: {final_generation.get('task_id')}")
            except Exception as kie_exception:
                generation_attempts.append({"provider": "kie.ai", "status": "failed", "error": str(kie_exception)})
                print(f"[!] Kie.ai генерация не удалась: {kie_exception}")

        if final_generation is None:
            failure_payload: Dict[str, Union[str, List[Dict]]] = {
                "status": "failed",
                "provider": None,
                "details": generation_attempts
            }
            if google_error:
                failure_payload["message"] = f"Google Veo: {google_error}"
            else:
                failure_payload["message"] = "Video generation failed"
            failure_payload["error"] = failure_payload.get("message")
            return {"step3_video_generation": failure_payload}
        else:
            if google_error and final_generation.get("provider") == "kie.ai":
                final_generation.setdefault("notes", []).append(f"Google fallback: {google_error}")
            result = {"step3_video_generation": final_generation}
            if generation_attempts:
                result["generation_attempts"] = generation_attempts
            return result

    def check_status(self, task_id: str, provider: str = "google") -> Dict:
        """
        Проверяет статус генерации видео

        Args:
            task_id: ID задачи генерации
            provider: "google" или "kie" - провайдер для проверки

        Returns:
            Dict со статусом и video_url если готово
        """
        if provider == "kie" and self.kie_client:
            return self.kie_client.check_task_status(task_id)
        elif provider == "google" and self.veo_client:
            return self.veo_client.check_video_status(task_id)
        else:
            raise ValueError(f"Провайдер {provider} не доступен или не инициализирован")

    def check_kie_status(self, task_id: str) -> Dict:
        """Проверяет статус генерации видео в Kie.ai"""
        if not self.kie_client:
            raise ValueError("Kie.ai клиент не инициализирован")
        return self.kie_client.check_task_status(task_id)

    def _download_media_with_headers(self, url: str, max_retries: int = 3) -> requests.Response:
        """
        Скачивает медиа по URL с правильными заголовками браузера для обхода блокировок
        Возвращает Response объект для дальнейшей обработки
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
                return response
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

    def _resolve_media_input(self, item: Union[Dict, bytes, str, Path], default_label: str) -> Dict:
        """Приводит вход медиа к стандартному представлению"""
        result = {
            "label": default_label,
            "content_type": None,
            "source": None
        }

        if isinstance(item, dict):
            result["label"] = item.get("filename") or item.get("label") or default_label
            result["content_type"] = item.get("content_type")
            if item.get("data") is not None:
                result["source"] = item["data"]
            elif item.get("url"):
                result["source"] = item["url"]
                result["content_type"] = result["content_type"] or item.get("url_content_type")
            elif item.get("path"):
                result["source"] = item["path"]
                if not result["content_type"]:
                    result["content_type"] = mimetypes.guess_type(str(item["path"]))[0]
        else:
            result["source"] = item
            if isinstance(item, (str, Path)):
                if hasattr(item, "name"):
                    result["label"] = getattr(item, "name")
                else:
                    result["label"] = default_label
        return result


if __name__ == "__main__":
    # Пример использования
    generator = Veo31VideoGenerator()

    result = generator.generate_from_references(
        reference_images=[
            "https://example.com/ring1.jpg",
            "https://example.com/ring2.jpg",
            "https://example.com/ring3.jpg"
        ],
        video_reference="https://example.com/reference_video.mp4",
        duration_seconds=10,
        aspect_ratio="16:9"
    )

    print(result)

