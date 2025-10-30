"""
Full workflow for video generation in Veo 3.1:
1. Reference video analysis (if available)
2. Reference image analysis (3 images)
3. Prompt generation
4. Video generation in Veo 3.1
"""

import os
from typing import List, Dict, Optional, Union
from pathlib import Path
from collections import defaultdict
import json
import mimetypes
import requests

# Import logging
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
    """Full pipeline for video generation in Veo 3.1 with references"""

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
        Initialize generator

        Args:
            openai_api_key: OpenAI API key for analysis
            gemini_api_key: Google API key for Veo 3.1
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
                print("[OK] Veo 3.1 client (Google) initialized")
            except Exception as gemini_error:
                print(f"[!] Failed to initialize Veo 3.1 client: {gemini_error}")
        else:
            print("[i] GEMINI_API_KEY not found, skipping Veo 3.1 (Google) client")

        self.kie_client: Optional[KieVeoClient] = None
        self.use_kie_default = prefer_kie_default
        if self.use_kie_default is None:
            self.use_kie_default = os.getenv("USE_KIE_API", "false").strip().lower() in {"1", "true", "yes", "on"}

        kie_key = kie_api_key or os.getenv("KIE_API_KEY")
        if kie_key:
            try:
                self.kie_client = KieVeoClient(api_key=kie_key)
                print("[OK] Kie.ai client initialized")
            except Exception as kie_error:
                print(f"[!] Failed to initialize Kie.ai client: {kie_error}")

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
        Only reference analysis and prompt generation (without video generation)
        Logic: video first, then images

        Args:
            reference_images: List of reference images
            video_reference: Optional reference video
            platform: Platform (veo3)
            use_case: Task type
            polish_with_perplexity: Polish prompt via Perplexity

        Returns:
            Dict with analysis and prompt
        """
        if not reference_images:
            raise ValueError("At least one reference image required")

        combined_analysis = {}
        video_analysis = None
        video_meta = None

        # STEP 1: Video analysis (if available) - FIRST
        if video_reference:
            try:
                logger.info("Step 1.1: Analyzing video reference via OpenAI...")
                resolved_video = self._resolve_media_input(video_reference, "reference_video")
                video_source = resolved_video.get("source")

                if video_source is None:
                    logger.warning("Failed to get video reference source")
                else:
                    video_analysis = self.pipeline.analyzer.analyze(
                        media_source=video_source,
                        content_type=resolved_video.get("content_type"),
                        use_gemini=False  # Always use OpenAI
                    )
                    video_meta = {
                        "label": resolved_video.get("label"),
                        "content_type": resolved_video.get("content_type")
                    }
                    metadata_model = video_analysis.get("metadata", {}).get("model_used")
                    video_meta["model_used"] = metadata_model
                    logger.info(f"Video reference analyzed successfully via {metadata_model}")
            except Exception as video_error:
                logger.error(f"Error analyzing video reference: {video_error}", exc_info=True)
                video_analysis = {"error": str(video_error)}
                video_meta = {"error": str(video_error)}

        # STEP 2: Image analysis - SECOND
        logger.info("Step 1.2: Analyzing images via OpenAI...")
        image_analysis_payloads = []
        prepared_images_for_generation: List[Dict] = []

        for idx, image_entry in enumerate(reference_images, start=1):
            try:
                resolved = self._resolve_media_input(image_entry, f"reference_image_{idx}")
                source = resolved.get("source")

                if source is None:
                    logger.warning(f"Image {idx}: no data")
                    image_analysis_payloads.append({
                        "analysis": {"error": "No image data provided"},
                        "meta": resolved
                    })
                    continue

                content_type = resolved.get("content_type") or "image/jpeg"
                logger.info(f"Analyzing image {idx}: {resolved.get('label')}")

                analysis = self.pipeline.analyzer.analyze(
                    media_source=source,
                    content_type=content_type,
                    use_gemini=False  # Always use OpenAI
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
                logger.error(f"Image analysis error {idx}: {image_error}", exc_info=True)
                image_analysis_payloads.append({
                    "analysis": {"error": str(image_error)},
                    "meta": {"label": f"reference_image_{idx}", "error": True}
                })

        # Aggregate image analysis results
        image_aggregate = self._aggregate_image_analyses(image_analysis_payloads)
        combined_analysis = dict(image_aggregate)

        # Merge with video analysis if available
        if video_analysis:
            combined_analysis = self._merge_analyses(image_aggregate, video_analysis, video_meta)
            combined_analysis["video_analysis"] = video_analysis
        if video_meta:
            combined_analysis.setdefault("video_reference_info", video_meta)

        combined_analysis["image_analysis_summary"] = image_aggregate.get("reference_summary")
        combined_analysis["image_reference_notes"] = image_aggregate.get("reference_notes")
        combined_analysis["image_sources"] = image_aggregate.get("sources")

        # STEP 3: Prompt generation
        logger.info("Step 2: Generating prompt...")

        # Check if structured data exists for prompt generation
        has_main_objects = bool(combined_analysis.get("main_objects"))
        has_materials = bool(combined_analysis.get("materials"))
        has_lighting = bool(combined_analysis.get("lighting"))
        has_raw_analysis = bool(combined_analysis.get("reference_notes") or combined_analysis.get("image_analysis_summary"))

        if not (has_main_objects or has_materials or has_lighting):
            # If no structured data, try using text data
            if has_raw_analysis:
                logger.warning("No structured analysis data, using text description")
                combined_analysis["raw_analysis"] = combined_analysis.get("image_analysis_summary", "") or " ".join(combined_analysis.get("reference_notes", []))
            else:
                logger.error("Analysis returned no data for prompt generation!")
                return {
                    "step1_analysis": combined_analysis,
                    "step2_prompt": "Error: Image analysis did not return structured data. Check OpenAI API availability and image format.",
                    "prompt": "Error: Image analysis did not return structured data. Check OpenAI API availability and image format.",
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

            # Check if prompt generation actually succeeded (not an error message)
            if generated_prompt.startswith("Prompt generation error:") or generated_prompt.startswith("Error:"):
                logger.error(f"Prompt generation returned error: {generated_prompt}")
                raise ValueError(f"Prompt generation failed: {generated_prompt}")

            logger.info("Prompt generated successfully")
        except Exception as prompt_error:
            logger.error(f"Prompt generation error: {prompt_error}", exc_info=True)
            generated_prompt = f"Prompt generation error: {str(prompt_error)}"
            # Mark as not ready if there was an error
            return {
                "step1_analysis": combined_analysis,
                "step2_prompt": generated_prompt,
                "prompt": generated_prompt,
                "ready_for_editing": False,
                "error": str(prompt_error)
            }

        return {
            "step1_analysis": combined_analysis,
            "step2_prompt": generated_prompt,
            "prompt": generated_prompt,  # Duplicate for compatibility
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
        Full cycle: analysis → prompt → video generation

        Args:
            reference_images: List of 3 reference images
            video_reference: Optional reference video for style analysis
            platform: Platform (veo3)
            use_case: Task type
            duration_seconds: Video duration
            aspect_ratio: Aspect ratio
            quality: Quality (for Google Veo). For Kie.ai converted to model:
                "high" -> "veo3", otherwise -> "veo3_fast"
            additional_prompt: Additional prompt instructions
            prefer_kie_api: Force use Kie.ai API (if available)

        Returns:
            Dict with generation result
        """
        results = {
            "step1_analysis": None,
            "step2_prompt": None,
            "step3_video_generation": None
        }

        # Step 1: Reference analysis
        logger.info("Step 1: Analyzing reference materials...")

        if not reference_images:
            raise ValueError("At least one reference image required")

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
                logger.debug(f"Analyzing image {idx}: {resolved.get('label')}")
                analysis = self.pipeline.analyzer.analyze(
                    media_source=source,
                    content_type=content_type,
                    use_gemini=False
                )
            except Exception as image_error:
                logger.error(f"Error analyzing image {idx}: {image_error}", exc_info=True)
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
                print("Analyzing video reference...")
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
                print("Video reference analyzed successfully")
            except Exception as video_error:
                print(f"Error analyzing video reference (continuing without it): {video_error}")

        combined_analysis["image_analysis_summary"] = image_aggregate.get("reference_summary")
        combined_analysis["image_reference_notes"] = image_aggregate.get("reference_notes")
        combined_analysis["image_sources"] = image_aggregate.get("sources")
        if video_analysis:
            combined_analysis["video_analysis"] = video_analysis
        if video_meta:
            combined_analysis.setdefault("video_reference_info", video_meta)

        results["step1_analysis"] = combined_analysis

        # Step 2: Prompt generation
        logger.info("Step 2: Generating prompt...")

        # Check if structured data exists for prompt generation
        has_main_objects = bool(combined_analysis.get("main_objects"))
        has_materials = bool(combined_analysis.get("materials"))
        has_lighting = bool(combined_analysis.get("lighting"))
        has_raw_analysis = bool(combined_analysis.get("reference_notes") or combined_analysis.get("image_analysis_summary"))

        if not (has_main_objects or has_materials or has_lighting):
            # If no structured data, try using text data
            if has_raw_analysis:
                logger.warning("No structured analysis data, using text description")
                # Create minimal structure from text data
                combined_analysis["raw_analysis"] = combined_analysis.get("image_analysis_summary", "") or " ".join(combined_analysis.get("reference_notes", []))
            else:
                logger.error("Analysis returned no data for prompt generation!")
                results["step2_prompt"] = "Error: Image analysis did not return structured data. Check OpenAI API availability and image format."
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

                # Add additional instructions if available
                if additional_prompt:
                    generated_prompt = f"{generated_prompt}. {additional_prompt}"

                results["step2_prompt"] = generated_prompt
                logger.info("Prompt generated successfully")
            except Exception as prompt_error:
                logger.error(f"Prompt generation error: {prompt_error}", exc_info=True)
                results["step2_prompt"] = f"Prompt generation error: {str(prompt_error)}"
                generated_prompt = None

        # Step 3: Video generation
        logger.info("Step 3: Generating video...")

        generated_prompt = results.get("step2_prompt")
        if not generated_prompt or generated_prompt.startswith("Error"):
            results["step3_video_generation"] = {
                "status": "failed",
                "error": "Failed to generate prompt",
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
                    logger.info(f"Video generating (Google), task_id: {final_generation.get('task_id')}")
            except Exception as google_exception:
                google_error = str(google_exception)
                logger.error(f"Google Veo generation failed: {google_error}", exc_info=True)
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
                    upload_notes.append(f"{item.get('filename') or 'image'}: no public URL and uploader not configured")
                    continue

                data = item.get("data")
                content_type = item.get("content_type") or "image/jpeg"

                if data is None and item.get("path"):
                    try:
                        data = Path(item["path"]).read_bytes()
                    except Exception as fs_err:
                        upload_notes.append(f"{item.get('filename') or 'image'}: failed to read file ({fs_err})")
                        data = None

                if data is None and item.get("url"):
                    try:
                        resp = self._download_media_with_headers(item["url"])
                        data = resp.content
                        content_type = item.get("content_type") or resp.headers.get("content-type") or content_type
                    except requests.RequestException as req_err:
                        upload_notes.append(f"{item.get('filename') or 'image'}: failed to download for upload ({req_err})")
                        data = None

                if not data:
                    upload_notes.append(f"{item.get('filename') or 'image'}: no available data for upload")
                    continue

                try:
                    uploaded_url = self.image_uploader.upload_image(data, item.get("filename"), content_type)
                    kie_image_urls.append(uploaded_url)
                    upload_notes.append(f"{item.get('filename') or 'image'} uploaded -> {uploaded_url}")
                except Exception as upload_error:
                    upload_notes.append(f"{item.get('filename') or 'image'}: upload error ({upload_error})")

            include_images = len(kie_image_urls) == len(prepared_images_for_generation) and len(kie_image_urls) > 0

            try:
                generation_type = "REFERENCE_2_VIDEO" if include_images else "TEXT_2_VIDEO"

                # Always use veo3_fast by default
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
                    attempt_record["note"] = "Images not passed (no public URLs)"
                    final_generation.setdefault("notes", []).append("Images skipped: no external URLs available")

                generation_attempts.append(attempt_record)
                if final_generation.get("status") == "submitted":
                    print(f"Video task submitted to Kie.ai, task_id: {final_generation.get('task_id')}")
            except Exception as kie_exception:
                generation_attempts.append({"provider": "kie.ai", "status": "failed", "error": str(kie_exception)})
                print(f"[!] Kie.ai generation failed: {kie_exception}")

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
        """Combines results of multiple image analyses"""
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

        # Convert defaultdict back to normal structures
        for field in ["materials", "lighting", "camera_work", "composition", "background_surfaces", "style_mood"]:
            aggregated[field] = {
                key: _dedupe_preserve_order(values) if isinstance(values, list) else values
                for key, values in aggregated[field].items()
            }

        aggregated["main_objects"] = _dedupe_preserve_order(aggregated["main_objects"])
        aggregated["source_count"] = len(analyses)

        # Create summary
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
        """Merges image and video analyses

        Priorities:
        - Objects: from images (images show what exactly to shoot)
        - Camera movement: from video (video shows how to move camera)
        - Lighting: from video (video shows light dynamics)
        - Materials: combine (from both sources)
        - Composition: from images (static)
        - Motion dynamics: only from video
        """
        merged = dict(image_analysis) if image_analysis else {}

        if not video_analysis:
            return merged

        # Objects: priority to images (they show what exactly to shoot)
        image_objects = image_analysis.get("main_objects", []) if image_analysis else []
        video_objects = video_analysis.get("main_objects", [])
        # Combine, but images first
        merged["main_objects"] = _dedupe_preserve_order(image_objects + video_objects)

        def merge_field(field_name: str, prefer_video: bool = False):
            image_field = image_analysis.get(field_name, {}) if image_analysis else {}
            video_field = video_analysis.get(field_name, {})

            if prefer_video and video_field:
                # For video-priority fields (camera_work, motion) use video
                return video_field

            # For others combine
            combined = dict(image_field)
            if isinstance(video_field, dict):
                for key, val in video_field.items():
                    if key in combined and isinstance(combined[key], list) and isinstance(val, list):
                        combined[key] = _dedupe_preserve_order(combined[key] + val)
                    else:
                        # If key not in images or not a list, take from video
                        combined[key] = val
            return combined

        # Materials: combine from both sources
        merged["materials"] = merge_field("materials")

        # Lighting: priority to video (shows light dynamics)
        merged["lighting"] = merge_field("lighting", prefer_video=True) or image_analysis.get("lighting", {})

        # Camera movement: ONLY from video (this is key difference between video and images)
        merged["camera_work"] = video_analysis.get("camera_work", {}) or merge_field("camera_work", prefer_video=True)

        # Composition: from images (static composition)
        merged["composition"] = image_analysis.get("composition", {}) if image_analysis else {}

        # Background: from images
        merged["background_surfaces"] = image_analysis.get("background_surfaces", {}) if image_analysis else {}

        # Movement: ONLY from video
        merged["motion_dynamics"] = video_analysis.get("motion_dynamics", {})

        # Style and mood: combine
        merged["style_mood"] = merge_field("style_mood")

        # Add text summaries with explicit source indication
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

        # Add flag that both sources were used
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
        Video generation with ready prompt (after editing)

        Args:
            prompt: Ready prompt (possibly edited)
            reference_images: List of reference images
            video_reference: Optional reference video
            duration_seconds: Video duration
            aspect_ratio: Aspect ratio
            quality: Quality
            prefer_kie_api: Force use Kie.ai API

        Returns:
            Dict with video generation result
        """
        # Prepare images for generation
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

        # Step 3: Video generation
        print("Step 3: Generating video with ready prompt...")

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
                    print(f"Video generating (Google), task_id: {final_generation.get('task_id')}")
            except Exception as google_exception:
                google_error = str(google_exception)
                generation_attempts.append({"provider": "google", "status": "failed", "error": google_error})
                print(f"[!] Google Veo generation failed: {google_error}")

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
                    upload_notes.append(f"{item.get('filename') or 'image'}: no public URL and uploader not configured")
                    continue

                data = item.get("data")
                content_type = item.get("content_type") or "image/jpeg"

                if data is None and item.get("path"):
                    try:
                        data = Path(item["path"]).read_bytes()
                    except Exception as fs_err:
                        upload_notes.append(f"{item.get('filename') or 'image'}: failed to read file ({fs_err})")
                        data = None

                if data is None and item.get("url"):
                    try:
                        resp = self._download_media_with_headers(item["url"])
                        data = resp.content
                        content_type = item.get("content_type") or resp.headers.get("content-type") or content_type
                    except requests.RequestException as req_err:
                        upload_notes.append(f"{item.get('filename') or 'image'}: failed to download for upload ({req_err})")
                        data = None

                if not data:
                    upload_notes.append(f"{item.get('filename') or 'image'}: no available data for upload")
                    continue

                try:
                    uploaded_url = self.image_uploader.upload_image(data, item.get("filename"), content_type)
                    kie_image_urls.append(uploaded_url)
                    upload_notes.append(f"{item.get('filename') or 'image'} uploaded -> {uploaded_url}")
                except Exception as upload_error:
                    upload_notes.append(f"{item.get('filename') or 'image'}: upload error ({upload_error})")

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
                    attempt_record["note"] = "Images not passed (no public URLs)"
                    final_generation.setdefault("notes", []).append("Images skipped: no external URLs available")

                generation_attempts.append(attempt_record)
                if final_generation.get("status") == "submitted":
                    print(f"Video task submitted to Kie.ai, task_id: {final_generation.get('task_id')}")
            except Exception as kie_exception:
                generation_attempts.append({"provider": "kie.ai", "status": "failed", "error": str(kie_exception)})
                print(f"[!] Kie.ai generation failed: {kie_exception}")

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
        Checks video generation status

        Args:
            task_id: Generation task ID
            provider: "google" or "kie" - provider to check

        Returns:
            Dict with status and video_url if ready
        """
        if provider == "kie" and self.kie_client:
            return self.kie_client.check_task_status(task_id)
        elif provider == "google" and self.veo_client:
            return self.veo_client.check_video_status(task_id)
        else:
            raise ValueError(f"Provider {provider} is not available or not initialized")

    def check_kie_status(self, task_id: str) -> Dict:
        """Checks video generation status in Kie.ai"""
        if not self.kie_client:
            raise ValueError("Kie.ai client is not initialized")
        return self.kie_client.check_task_status(task_id)

    def _download_media_with_headers(self, url: str, max_retries: int = 3) -> requests.Response:
        """
        Downloads media from URL with proper browser headers to bypass blocks
        Returns Response object for further processing
        """
        from urllib.parse import urlparse

        # Form proper Referer from URL domain
        parsed_url = urlparse(url)
        referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"

        # Different User-Agents for attempts
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

        # Retry logic with different approaches (WITHOUT Sec-Fetch headers!)
        last_error = None
        for attempt in range(max_retries):
            try:
                # Try different header sets
                if attempt == 0:
                    # First attempt: minimal headers without Referer
                    headers = {
                        'User-Agent': user_agents[0],
                        'Accept': '*/*'
                    }
                elif attempt == 1:
                    # Second attempt: with Referer
                    headers = {
                        'User-Agent': user_agents[1],
                        'Accept': '*/*',
                        'Referer': referer
                    }
                else:
                    # Third attempt: full set but without Sec-Fetch
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
                    # For 403 try next approach
                    continue
                # For other HTTP errors immediately re-raise
                if e.response.status_code != 403:
                    raise
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    continue
                raise

        # If all attempts failed, re-raise the last error
        if last_error:
            raise last_error

    def _resolve_media_input(self, item: Union[Dict, bytes, str, Path], default_label: str) -> Dict:
        """Converts media input to standard representation"""
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
    # Usage example
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

