"""Workflow orchestration for combined video+image prompt generation."""

from __future__ import annotations

import json
import os
import uuid
import mimetypes
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Union, Any

from pipeline import VideoPromptPipeline
from kie_api_client import KieVeoClient
from media_uploader import build_image_uploader, ImageUploader


@dataclass
class PromptWorkflowState:
    """Keeps the intermediate state between analysis and final generation."""

    workflow_id: str
    platform: str
    use_case: str
    video_source: Optional[str]
    image_reference_url: Optional[str]
    video_analysis: Dict[str, Any]
    image_analysis: Dict[str, Any]
    combined_description: Dict[str, Any]
    draft_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        """Representation that can be sent via API responses."""
        public_payload = asdict(self)
        # internal fields should not leak (e.g. we don't expose combined_description twice)
        workflow_status = (
            self.metadata.get("workflow", {}).get("status")
            if isinstance(self.metadata, dict)
            else None
        )
        if workflow_status:
            public_payload["status"] = workflow_status
        else:
            public_payload.setdefault("status", "awaiting_user_confirmation")
        return public_payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PromptWorkflowState":
        """Recreates a state object from JSON-compatible payload."""
        valid_fields = {item.name for item in fields(cls)}
        init_payload: Dict[str, Any] = {}
        for key in valid_fields:
            if key in payload:
                init_payload[key] = payload[key]
        return cls(**init_payload)  # type: ignore[arg-type]


class WorkflowStateStorage:
    """Abstract interface for persisting workflow states."""

    def load_all(self) -> Dict[str, PromptWorkflowState]:  # pragma: no cover - interface definition
        raise NotImplementedError

    def save_state(self, state: PromptWorkflowState) -> None:  # pragma: no cover - interface definition
        raise NotImplementedError

    def delete_state(self, workflow_id: str) -> None:  # pragma: no cover - interface definition
        raise NotImplementedError

    def get_state(self, workflow_id: str) -> Optional[PromptWorkflowState]:  # pragma: no cover - interface definition
        raise NotImplementedError


class FileWorkflowStateStorage(WorkflowStateStorage):
    """Simple JSON file storage for workflow states."""

    def __init__(self, storage_path: Union[str, Path]) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _read_raw(self) -> Dict[str, Any]:
        if not self.storage_path.exists():
            return {}
        try:
            with self.storage_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            # Corrupted storage, start afresh but keep backup copy for troubleshooting
            backup_path = self.storage_path.with_suffix(self.storage_path.suffix + ".bak")
            try:
                if backup_path.exists():
                    backup_path.unlink()
                self.storage_path.replace(backup_path)
            except OSError:
                try:
                    self.storage_path.unlink()
                except OSError:
                    pass
            return {}

    def _write_raw(self, data: Dict[str, Any]) -> None:
        tmp_path = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(self.storage_path)

    def load_all(self) -> Dict[str, PromptWorkflowState]:
        raw = self._read_raw()
        states: Dict[str, PromptWorkflowState] = {}
        for workflow_id, payload in raw.items():
            try:
                states[workflow_id] = PromptWorkflowState.from_dict(payload)
            except TypeError:
                continue
        return states

    def save_state(self, state: PromptWorkflowState) -> None:
        with self._lock:
            raw = self._read_raw()
            raw[state.workflow_id] = asdict(state)
            self._write_raw(raw)

    def delete_state(self, workflow_id: str) -> None:
        with self._lock:
            raw = self._read_raw()
            if workflow_id in raw:
                raw.pop(workflow_id)
                self._write_raw(raw)

    def get_state(self, workflow_id: str) -> Optional[PromptWorkflowState]:
        raw = self._read_raw()
        payload = raw.get(workflow_id)
        if not payload:
            return None
        try:
            return PromptWorkflowState.from_dict(payload)
        except TypeError:
            return None


class VideoImagePromptWorkflow:
    """Coordinates the new workflow: analyze video, analyze image, wait for user edit, then send to Kie.ai."""

    def __init__(
        self,
        pipeline: VideoPromptPipeline,
        kie_api_key: Optional[str] = None,
        state_storage: Optional[WorkflowStateStorage] = None,
    ) -> None:
        self.pipeline = pipeline
        self.kie_client: Optional[KieVeoClient] = None
        if kie_api_key:
            try:
                self.kie_client = KieVeoClient(api_key=kie_api_key)
            except Exception as error:  # pragma: no cover - network errors
                raise RuntimeError(f"Не удалось инициализировать Kie.ai клиента: {error}")

        self.image_uploader: Optional[ImageUploader] = build_image_uploader()
        storage_path = os.getenv("WORKFLOW_STATE_PATH", "workflow_states.json")
        self._state_storage: WorkflowStateStorage = state_storage or FileWorkflowStateStorage(storage_path)
        self._active_workflows: Dict[str, PromptWorkflowState] = self._state_storage.load_all()

    @property
    def state_storage(self) -> WorkflowStateStorage:
        """Expose storage for read-only operations (e.g. API layer)."""
        return self._state_storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start_workflow(
        self,
        video_source: Union[str, Path, bytes],
        image_source: Union[str, Path, bytes],
        *,
        platform: str = "veo3",
        use_case: str = "general",
        polish_with_perplexity: Optional[bool] = None,
    ) -> PromptWorkflowState:
        """Runs the mandatory first steps and returns a state waiting for user edits."""
        video_analysis = self.pipeline.analyzer.analyze(
            media_source=video_source,
            use_gemini=False,
        )

        image_reference_payload = self._resolve_image_reference(image_source)
        image_for_analysis = image_reference_payload["analysis_source"]
        image_analysis = self.pipeline.analyzer.analyze(
            media_source=image_for_analysis,
            content_type=image_reference_payload.get("content_type"),
            use_gemini=False,
        )

        combined_description = self._combine_analyses(video_analysis, image_analysis)
        combined_description.setdefault("workflow_context", {})["stage"] = "video_camera_plus_image_subject"

        prompt_result = self.pipeline.generator.generate(
            scene_description=combined_description,
            platform=platform,
            use_case=use_case,
            polish_with_perplexity=polish_with_perplexity,
        )

        workflow_state = PromptWorkflowState(
            workflow_id=uuid.uuid4().hex,
            platform=platform,
            use_case=use_case,
            video_source=self._normalise_media_source(video_source),
            image_reference_url=image_reference_payload.get("url"),
            video_analysis=video_analysis,
            image_analysis=image_analysis,
            combined_description=combined_description,
            draft_prompt=prompt_result["prompt"],
            metadata={
                "prompt_generation": prompt_result["metadata"],
                "analysis": {
                    "video": video_analysis.get("metadata", {}),
                    "image": image_analysis.get("metadata", {}),
                },
            },
        )

        self._active_workflows[workflow_state.workflow_id] = workflow_state
        self._state_storage.save_state(workflow_state)
        return workflow_state

    def finalize_workflow(
        self,
        workflow_id: str,
        *,
        user_prompt: Optional[str] = None,
        image_url_override: Optional[str] = None,
        model: str = "veo3_fast",
        aspect_ratio: str = "16:9",
        enable_translation: bool = True,
    ) -> Dict[str, Any]:
        """Submits the final prompt and image URL to Kie.ai for generation."""
        if self.kie_client is None:
            raise RuntimeError("Kie.ai клиент не инициализирован. Укажите KIE_API_KEY для генерации видео")

        state = self._active_workflows.get(workflow_id)
        if state is None:
            raise KeyError(f"Workflow с идентификатором {workflow_id} не найден")

        prompt_to_send = (user_prompt or "").strip() or state.draft_prompt
        image_url = image_url_override or state.image_reference_url

        if not image_url:
            raise ValueError("Для генерации видео требуется публичный URL изображения (image_url)")

        response = self.kie_client.generate_video(
            prompt=prompt_to_send,
            image_urls=[image_url],
            model=model,
            aspect_ratio=aspect_ratio,
            enable_translation=enable_translation,
        )

        result_payload = {
            "workflow_id": workflow_id,
            "status": "submitted_to_kie",
            "prompt": prompt_to_send,
            "image_urls": [image_url],
            "kie_response": response,
        }

        # Optionally keep task id for follow-up status checks
        task_id = None
        if isinstance(response, dict):
            task_id = response.get("data", {}).get("taskId") or response.get("taskId")
        if task_id:
            kie_meta = state.metadata.setdefault("kie", {})
            kie_meta["task_id"] = task_id
            kie_meta["status"] = "submitted"
            kie_meta["last_response"] = response
            result_payload["task_id"] = task_id

        workflow_meta = state.metadata.setdefault("workflow", {})
        workflow_meta["status"] = "submitted_to_kie"

        self._active_workflows[workflow_id] = state
        self._state_storage.save_state(state)

        return result_payload

    def get_workflow_state(self, workflow_id: str) -> PromptWorkflowState:
        state = self._active_workflows.get(workflow_id)
        if state is None:
            state = self._state_storage.get_state(workflow_id)
            if state:
                self._active_workflows[workflow_id] = state
        if state is None:
            raise KeyError(f"Workflow с идентификатором {workflow_id} не найден")
        return state

    def refresh_generation_status(self, workflow_id: str) -> Dict[str, Any]:
        """Checks the remote generation task and updates workflow metadata."""
        if self.kie_client is None:
            raise RuntimeError(
                "Kie.ai клиент не инициализирован. Проверьте KIE_API_KEY для проверки статуса"
            )

        state = self.get_workflow_state(workflow_id)
        kie_meta = state.metadata.setdefault("kie", {})
        task_id = kie_meta.get("task_id")
        if not task_id:
            raise ValueError(
                "Workflow ещё не был отправлен на генерацию или task_id отсутствует"
            )

        status_payload = self.kie_client.check_task_status(task_id)
        kie_meta["last_status"] = status_payload
        if isinstance(status_payload, dict):
            status_value = status_payload.get("status") or kie_meta.get("status")
            if status_value:
                kie_meta["status"] = status_value
            video_url = status_payload.get("video_url")
            if video_url:
                kie_meta["video_url"] = video_url

        workflow_meta = state.metadata.setdefault("workflow", {})
        workflow_meta["status"] = kie_meta.get("status", "processing")
        if kie_meta.get("video_url"):
            workflow_meta.setdefault("result", {})["video_url"] = kie_meta["video_url"]

        public_payload = state.to_public_dict()
        public_payload["task_id"] = task_id
        public_payload["kie_status"] = status_payload

        if kie_meta.get("status") == "completed":
            workflow_meta["status"] = "completed"
            self._active_workflows.pop(workflow_id, None)
            self._state_storage.delete_state(workflow_id)
        else:
            self._active_workflows[workflow_id] = state
            self._state_storage.save_state(state)

        return public_payload

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_image_reference(self, image_source: Union[str, Path, bytes]) -> Dict[str, Any]:
        """Ensures we have both analysis data and a URL for downstream Kie.ai call."""
        if isinstance(image_source, bytes):
            if not self.image_uploader:
                raise ValueError("Получен байтовый контент изображения, но загрузчик не настроен. Укажите S3 конфигурацию либо предоставьте публичный URL.")
            url = self.image_uploader.upload_image(image_source, filename="reference.jpg", content_type="image/jpeg")
            return {
                "analysis_source": image_source,
                "url": url,
                "content_type": "image/jpeg",
            }

        source_path = Path(image_source) if isinstance(image_source, Path) else None
        source_str = str(image_source) if not isinstance(image_source, Path) else str(image_source)

        if source_str.startswith(("http://", "https://")):
            content_type = mimetypes.guess_type(source_str)[0]
            return {
                "analysis_source": source_str,
                "url": source_str,
                "content_type": content_type,
            }

        if source_path and source_path.exists():
            data = source_path.read_bytes()
            content_type = mimetypes.guess_type(source_path.name)[0] or "image/jpeg"
            if not self.image_uploader:
                raise ValueError("Для локального изображения требуется настроенный загрузчик S3 (см. S3_UPLOAD_* переменные) либо укажите прямой URL.")
            url = self.image_uploader.upload_image(data, filename=source_path.name, content_type=content_type)
            return {
                "analysis_source": data,
                "url": url,
                "content_type": content_type,
            }

        raise ValueError("Не удалось определить источник изображения. Укажите публичный URL или путь к файлу.")

    def _combine_analyses(self, video_analysis: Dict[str, Any], image_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a unified scene description emphasising camera motion from video and object detail from image."""
        video_copy = json.loads(json.dumps(video_analysis, ensure_ascii=False)) if video_analysis else {}
        image_copy = json.loads(json.dumps(image_analysis, ensure_ascii=False)) if image_analysis else {}

        combined = video_copy or {}
        combined.setdefault("analysis_sources", {})
        combined["analysis_sources"]["video"] = {
            "summary": video_copy.get("raw_analysis"),
            "camera_motion": video_copy.get("camera_motion"),
            "media": video_copy.get("media"),
        }
        combined["analysis_sources"]["image"] = {
            "subjects": image_copy.get("subjects"),
            "composition": image_copy.get("composition"),
            "color_and_post": image_copy.get("color_and_post"),
        }

        if image_copy.get("subjects"):
            combined["subjects"] = image_copy["subjects"]
        else:
            combined.setdefault("subjects", video_copy.get("subjects", []))

        combined.setdefault("reference_image_analysis", image_copy)
        combined.setdefault("video_subjects", video_copy.get("subjects"))

        synthesis_notes = {
            "camera_motion_from_video": "Используй ключевые кадры и поддержку камеры из video_analysis для описания движения.",
            "object_from_image": "Основной объект, материалы и текстуры бери из reference_image_analysis.subjects.",
            "lighting_merge_hint": "Сочетай схему освещения: если в video_analysis есть lighting, используй её как основу, уточнив детали из изображения." if video_copy.get("lighting") else "Опиши освещение, ориентируясь на изображение и общую стилистику.",
        }
        combined.setdefault("workflow_context", {})
        combined["workflow_context"].setdefault("instructions", synthesis_notes)

        return combined

    @staticmethod
    def _normalise_media_source(media_source: Union[str, Path, bytes]) -> Optional[str]:
        if isinstance(media_source, bytes):
            return None
        if isinstance(media_source, Path):
            return str(media_source)
        return media_source


__all__ = [
    "PromptWorkflowState",
    "VideoImagePromptWorkflow",
    "WorkflowStateStorage",
    "FileWorkflowStateStorage",
]
