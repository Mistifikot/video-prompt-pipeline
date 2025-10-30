"""
FastAPI сервер для доступа к пайплайну через REST API
"""

import os
import sys
# Устанавливаем кодировку UTF-8 для вывода (особенно важно для Windows cmd)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

from typing import Optional
from pathlib import Path
import traceback
import requests

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# Импорт конфигурации и логирования
try:
    from config import (
        OPENAI_API_KEY, GEMINI_API_KEY, PPLX_API_KEY, KIE_API_KEY,
        PORT, HOST, DEBUG, validate_config
    )
    from logger_utils import logger
except ImportError:
    # Fallback если модули не найдены
    import logging
    logger = logging.getLogger(__name__)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PPLX_API_KEY = os.getenv("PPLX_API_KEY")
    KIE_API_KEY = os.getenv("KIE_API_KEY")
    PORT = int(os.getenv("PORT", "8000"))
    HOST = os.getenv("HOST", "0.0.0.0")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    def validate_config():
        errors = []
        if not OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY не найден")
        return len(errors) == 0, errors

from pipeline import VideoPromptPipeline
from workflow_manager import VideoImagePromptWorkflow

# Импорт Veo 3.1 генератора (опционально)
try:
    from veo31_generator import Veo31VideoGenerator
    VEO31_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Veo 3.1 модуль недоступен: {e}")
    VEO31_AVAILABLE = False
    Veo31VideoGenerator = None

# Валидация конфигурации
is_valid, config_errors = validate_config()
if not is_valid:
    for error in config_errors:
        logger.error(error)
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY обязателен для работы приложения!")

app = FastAPI(title="Video Prompt Pipeline API", version="1.0.0")


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"", "none"}:
        return None
    return value_str in {"1", "true", "yes", "on"}

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация пайплайна
try:
    pipeline = VideoPromptPipeline(
        openai_api_key=OPENAI_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        use_gemini_for_analysis=os.getenv("USE_GEMINI", "false").lower() == "true",
        perplexity_api_key=PPLX_API_KEY,
        auto_perplexity_polish=None
    )
    logger.info("Pipeline инициализирован успешно")
except Exception as e:
    logger.error(f"Ошибка инициализации pipeline: {e}")
    raise

try:
    workflow_orchestrator = VideoImagePromptWorkflow(
        pipeline=pipeline,
        kie_api_key=KIE_API_KEY,
    )
    logger.info("Workflow orchestrator инициализирован")
except Exception as workflow_error:
    workflow_orchestrator = None
    logger.error(f"Не удалось инициализировать workflow orchestrator: {workflow_error}")

# Инициализация Veo 3.1 генератора
veo31_generator = None
if VEO31_AVAILABLE:
    try:
        if GEMINI_API_KEY or KIE_API_KEY:
            veo31_generator = Veo31VideoGenerator(
                openai_api_key=OPENAI_API_KEY,
                gemini_api_key=GEMINI_API_KEY,
                perplexity_api_key=PPLX_API_KEY,
                kie_api_key=KIE_API_KEY,
                prefer_kie_default=None
            )
            logger.info("Veo 3.1 генератор инициализирован")
        else:
            logger.info("Veo 3.1 генератор не инициализирован (нет Gemini или Kie API key)")
    except Exception as e:
        logger.error(f"Не удалось инициализировать Veo 3.1 генератор: {e}", exc_info=True)
else:
    logger.info("Veo 3.1 функции недоступны (модуль не загружен)")


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "Video Prompt Pipeline",
        "endpoints": {
            "analyze": "/analyze",
            "generate": "/generate",
            "process": "/process",
            "ui": "/ui",
            "veo31_ui": "/veo31",
            "veo31_generate": "/veo31/generate",
            "veo31_generate_from_urls": "/veo31/generate-from-urls",
            "veo31_analyze_and_prompt": "/veo31/analyze-and-prompt",
            "veo31_generate_with_prompt": "/veo31/generate-with-prompt",
            "veo31_status": "/veo31/status/{task_id}",
            "veo31_kie_status": "/veo31/kie-status/{task_id}",
            "workflow_start": "/workflow/start",
            "workflow_submit": "/workflow/submit",
            "workflow_state": "/workflow/{workflow_id}",
            "workflow_status": "/workflow/{workflow_id}/status"
        }
    }

@app.get("/ui")
async def serve_ui():
    """Веб-интерфейс"""
    ui_path = Path(__file__).parent / "web_interface.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    else:
        return {"error": "UI file not found"}

@app.get("/veo31")
async def serve_veo31_ui():
    """Веб-интерфейс для Veo 3.1"""
    ui_path = Path(__file__).parent / "veo31_interface.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    else:
        return {"error": "Veo 3.1 UI file not found"}


@app.post("/analyze")
async def analyze_media(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    use_gemini: bool = Form(False)
):
    """
    Агент 1: Анализ медиа (изображение/видео)

    Принимает либо URL, либо файл
    """
    try:
        if not url and not file:
            raise HTTPException(status_code=400, detail="Необходимо указать url или загрузить file")

        if url:
            logger.info(f"Анализ медиа по URL: {url[:100]}...")
            scene_analysis = pipeline.analyzer.analyze(
                media_source=url,
                use_gemini=use_gemini
            )
        else:
            logger.info(f"Анализ загруженного файла: {file.filename}")
            file_data = await file.read()
            scene_analysis = pipeline.analyzer.analyze(
                media_source=file_data,
                content_type=file.content_type,
                use_gemini=use_gemini
            )

        logger.info("Анализ завершен успешно")
        return JSONResponse(content=scene_analysis)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка анализа медиа: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


@app.post("/workflow/start")
async def workflow_start(
    video_url: str = Form(...),
    image_url: str = Form(...),
    platform: str = Form("veo3"),
    use_case: str = Form("general"),
    polish_with_perplexity: Optional[str] = Form(None),
):
    """Полный старт workflow: анализ видео, анализ изображения, подготовка драфта промпта."""
    if workflow_orchestrator is None:
        raise HTTPException(status_code=500, detail="Workflow orchestrator не инициализирован")

    try:
        workflow_state = workflow_orchestrator.start_workflow(
            video_source=video_url,
            image_source=image_url,
            platform=platform,
            use_case=use_case,
            polish_with_perplexity=_parse_optional_bool(polish_with_perplexity),
        )
        return JSONResponse(content=workflow_state.to_public_dict())
    except Exception as exc:
        logger.error(f"Ошибка запуска workflow: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка запуска workflow: {exc}")


@app.get("/workflow/{workflow_id}")
async def workflow_state(workflow_id: str):
    """Возвращает сохраненное состояние workflow (драфт промпта + анализы)."""
    if workflow_orchestrator is None:
        raise HTTPException(status_code=503, detail="Workflow orchestrator не инициализирован")

    try:
        state = workflow_orchestrator.get_workflow_state(workflow_id)
        return JSONResponse(content=state.to_public_dict())
    except KeyError:
        raise HTTPException(status_code=404, detail="Workflow не найден")


@app.get("/workflow/{workflow_id}/status")
async def workflow_status(workflow_id: str):
    """Проверяет статус генерации видео для workflow через Kie.ai."""
    if workflow_orchestrator is None:
        raise HTTPException(status_code=503, detail="Workflow orchestrator не инициализирован")

    try:
        status_payload = workflow_orchestrator.refresh_generation_status(workflow_id)
        return JSONResponse(content=status_payload)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} не найден")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive branch for unexpected API errors
        logger.error(f"Ошибка проверки статуса workflow {workflow_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка проверки статуса: {exc}")


@app.post("/workflow/submit")
async def workflow_submit(
    workflow_id: str = Form(...),
    prompt_override: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    model: str = Form("veo3_fast"),
    aspect_ratio: str = Form("16:9"),
    enable_translation: Optional[str] = Form(None),
):
    """Принимает подтвержденный пользователем промпт и отправляет задачу в Kie.ai."""
    if workflow_orchestrator is None:
        raise HTTPException(status_code=500, detail="Workflow orchestrator не инициализирован")

    try:
        enable_translation_flag = _parse_optional_bool(enable_translation)
        result = workflow_orchestrator.finalize_workflow(
            workflow_id=workflow_id,
            user_prompt=prompt_override,
            image_url_override=image_url,
            model=model,
            aspect_ratio=aspect_ratio,
            enable_translation=True if enable_translation_flag is None else enable_translation_flag,
        )
        return JSONResponse(content=result)
    except KeyError as missing:
        raise HTTPException(status_code=404, detail=str(missing))
    except Exception as exc:
        logger.error(f"Ошибка завершения workflow: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка завершения workflow: {exc}")


@app.post("/generate")
async def generate_prompt(
    scene_description: str = Form(...),
    platform: str = Form("veo3"),
    use_case: str = Form("general")
):
    """
    Агент 2: Генерация промпта из описания сцены

    scene_description - JSON строка с описанием сцены
    """
    try:
        import json

        # Валидация платформы
        valid_platforms = ["veo3", "sora2", "seedream"]
        if platform not in valid_platforms:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемая платформа: {platform}. Доступны: {', '.join(valid_platforms)}"
            )

        scene_dict = json.loads(scene_description)
        logger.info(f"Генерация промпта для платформы: {platform}, use_case: {use_case}")

        result = pipeline.generator.generate(
            scene_description=scene_dict,
            platform=platform,
            use_case=use_case
        )

        logger.info("Промпт сгенерирован успешно")
        return JSONResponse(content=result)

    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        raise HTTPException(status_code=400, detail=f"Неверный формат JSON в scene_description: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка генерации промпта: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")


@app.post("/process")
async def process_full(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    platform: str = Form("veo3"),
    use_case: str = Form("general"),
    return_intermediate: bool = Form(False),
    use_gemini: bool = Form(False)
):
    """
    Полный пайплайн: анализ + генерация промпта

    Принимает либо URL, либо файл
    Возвращает готовый промт
    """
    try:
        # Временно переключаем использование Gemini если указано
        original_use_gemini = pipeline.use_gemini
        if use_gemini:
            pipeline.use_gemini = True

        if url:
            result = pipeline.process(
                media_source=url,
                platform=platform,
                use_case=use_case,
                return_intermediate=return_intermediate
            )
        elif file:
            file_data = await file.read()
            result = pipeline.process(
                media_source=file_data,
                platform=platform,
                use_case=use_case,
                content_type=file.content_type,
                return_intermediate=return_intermediate
            )
        else:
            raise HTTPException(status_code=400, detail="Необходимо указать url или загрузить file")

        # Восстанавливаем оригинальное значение
        pipeline.use_gemini = original_use_gemini

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Файл не найден: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка валидации: {str(e)}")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Ошибка внешнего API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.post("/process-multiple")
async def process_multiple_platforms(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    platforms: str = Form("veo3,sora2"),  # Список через запятую
    use_case: str = Form("general"),
    use_gemini: bool = Form(False)
):
    """
    Полный пайплайн для нескольких платформ одновременно
    """
    try:
        platform_list = [p.strip() for p in platforms.split(",")]

        if url:
            result = pipeline.process_multiple_platforms(
                media_source=url,
                platforms=platform_list,
                use_case=use_case
            )
        elif file:
            file_data = await file.read()
            result = pipeline.process_multiple_platforms(
                media_source=file_data,
                platforms=platform_list,
                use_case=use_case,
                content_type=file.content_type
            )
        else:
            raise HTTPException(status_code=400, detail="Необходимо указать url или загрузить file")

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Файл не найден: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка валидации: {str(e)}")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Ошибка внешнего API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.post("/veo31/analyze-and-prompt")
async def analyze_and_generate_prompt_only(
    reference_image1: UploadFile = File(...),
    reference_image2: Optional[UploadFile] = File(None),
    reference_image3: Optional[UploadFile] = File(None),
    video_reference: Optional[UploadFile] = File(None),
    video_reference_url: Optional[str] = Form(None),
    platform: str = Form("veo3"),
    polish_with_perplexity: Optional[bool] = Form(None)
):
    """
    Только анализ референсов и генерация промпта (без генерации видео)
    Позволяет отредактировать промпт перед генерацией
    """
    if not veo31_generator:
        raise HTTPException(
            status_code=503,
            detail="Veo 3.1 генератор не инициализирован. Проверьте OPENAI_API_KEY в .env"
        )

    try:
        logger.info(f"Анализ и генерация промпта: platform={platform}")

        # Собираем список референсных изображений
        reference_images = []

        img1_data = await reference_image1.read()
        reference_images.append({
            "data": img1_data,
            "content_type": reference_image1.content_type or "image/jpeg",
            "filename": reference_image1.filename or "reference_image_1"
        })

        if reference_image2:
            img2_data = await reference_image2.read()
            reference_images.append({
                "data": img2_data,
                "content_type": reference_image2.content_type or "image/jpeg",
                "filename": reference_image2.filename or "reference_image_2"
            })

        if reference_image3:
            img3_data = await reference_image3.read()
            reference_images.append({
                "data": img3_data,
                "content_type": reference_image3.content_type or "image/jpeg",
                "filename": reference_image3.filename or "reference_image_3"
            })

        # Видео-референс: либо файл, либо URL
        video_ref = None
        if video_reference:
            video_bytes = await video_reference.read()
            video_ref = {
                "data": video_bytes,
                "content_type": video_reference.content_type or "video/mp4",
                "filename": video_reference.filename or "reference_video"
            }
            logger.info(f"Видео-референс загружен: {video_reference.filename}")
        elif video_reference_url:
            video_ref = video_reference_url.strip()
            logger.info(f"Видео-референс URL: {video_reference_url[:100]}...")

        # Выполняем только анализ и генерацию промпта
        result = veo31_generator.analyze_and_generate_prompt(
            reference_images=reference_images,
            video_reference=video_ref,
            platform=platform,
            use_case="general",  # Всегда используем общий use_case
            polish_with_perplexity=polish_with_perplexity
        )

        # Совместимость: дублируем поле промпта в ключ "prompt"
        if isinstance(result, dict) and "step2_prompt" in result:
            result.setdefault("prompt", result.get("step2_prompt"))

        logger.info("Анализ и генерация промпта завершены успешно")
        return JSONResponse(content=result)

    except ValueError as e:
        logger.error(f"Ошибка валидации: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка анализа и генерации промпта: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


@app.post("/veo31/generate-with-prompt")
async def generate_video_with_prompt(
    prompt: str = Form(...),
    reference_image1: UploadFile = File(...),
    reference_image2: Optional[UploadFile] = File(None),
    reference_image3: Optional[UploadFile] = File(None),
    video_reference: Optional[UploadFile] = File(None),
    video_reference_url: Optional[str] = Form(None),
    aspect_ratio: str = Form("16:9"),
    prefer_kie_api: bool = Form(False)
):
    """
    Генерация видео с уже готовым промптом (после редактирования)

    Args:
        prompt: Готовый промпт (возможно отредактированный)
        reference_images: Референсные изображения
        video_reference: Опциональное видео-референс
        aspect_ratio: Соотношение сторон
        prefer_kie_api: Использовать Kie.ai API
    """
    if not veo31_generator:
        raise HTTPException(
            status_code=503,
            detail="Veo 3.1 генератор не инициализирован"
        )

    try:
        logger.info(f"Генерация видео с промптом: aspect={aspect_ratio}, prefer_kie={prefer_kie_api}")

        # Валидация параметров
        valid_aspect_ratios = ["16:9", "9:16", "1:1", "4:3", "21:9"]
        if aspect_ratio not in valid_aspect_ratios:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый aspect_ratio: {aspect_ratio}. Доступны: {', '.join(valid_aspect_ratios)}"
            )

        if not prompt or len(prompt.strip()) < 10:
            raise HTTPException(status_code=400, detail="Промпт слишком короткий (минимум 10 символов)")

        # Veo генерирует всегда 8 секунд, качество veo3_fast
        duration_seconds = 8
        quality = "standard"  # Не используется напрямую, но передается для совместимости

        # Собираем список референсных изображений
        reference_images = []

        img1_data = await reference_image1.read()
        reference_images.append({
            "data": img1_data,
            "content_type": reference_image1.content_type or "image/jpeg",
            "filename": reference_image1.filename or "reference_image_1"
        })

        if reference_image2:
            img2_data = await reference_image2.read()
            reference_images.append({
                "data": img2_data,
                "content_type": reference_image2.content_type or "image/jpeg",
                "filename": reference_image2.filename or "reference_image_2"
            })

        if reference_image3:
            img3_data = await reference_image3.read()
            reference_images.append({
                "data": img3_data,
                "content_type": reference_image3.content_type or "image/jpeg",
                "filename": reference_image3.filename or "reference_image_3"
            })

        # Видео-референс
        video_ref = None
        if video_reference:
            video_bytes = await video_reference.read()
            video_ref = {
                "data": video_bytes,
                "content_type": video_reference.content_type or "video/mp4",
                "filename": video_reference.filename or "reference_video"
            }
        elif video_reference_url:
            video_ref = video_reference_url.strip()

        # Генерируем видео с готовым промптом
        # Veo всегда генерирует 8 секунд, качество veo3_fast
        result = veo31_generator.generate_video_with_prompt(
            prompt=prompt,
            reference_images=reference_images,
            video_reference=video_ref,
            duration_seconds=8,  # Veo всегда генерирует 8 секунд
            aspect_ratio=aspect_ratio,
            quality="standard",  # Не используется, но передается для совместимости
            prefer_kie_api=prefer_kie_api
        )

        logger.info("Генерация видео завершена успешно")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Ошибка валидации: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка генерации видео с промптом: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")


@app.post("/veo31/generate")
async def generate_veo31_video(
    reference_image1: UploadFile = File(...),
    reference_image2: Optional[UploadFile] = File(None),
    reference_image3: Optional[UploadFile] = File(None),
    video_reference: Optional[UploadFile] = File(None),
    video_reference_url: Optional[str] = Form(None),
    platform: str = Form("veo3"),
    use_case: str = Form("product_video"),
    duration_seconds: int = Form(5),
    aspect_ratio: str = Form("16:9"),
    quality: str = Form("high"),
    additional_prompt: Optional[str] = Form(None),
    polish_with_perplexity: Optional[bool] = Form(None),
    prefer_kie_api: bool = Form(False)
):
    """
    Генерация видео в Veo 3.1 с референсными изображениями

    Требуется минимум 1 изображение, можно загрузить до 3
    Опционально можно загрузить видео-референс
    """
    if not veo31_generator:
        raise HTTPException(
            status_code=503,
            detail="Veo 3.1 генератор не инициализирован. Проверьте GEMINI_API_KEY в .env"
        )

    try:
        # Собираем список референсных изображений
        reference_images = []

        # Первое изображение обязательно
        img1_data = await reference_image1.read()
        reference_images.append({
            "data": img1_data,
            "content_type": reference_image1.content_type or "image/jpeg",
            "filename": reference_image1.filename or "reference_image_1"
        })

        # Второе изображение (опционально)
        if reference_image2:
            img2_data = await reference_image2.read()
            reference_images.append({
                "data": img2_data,
                "content_type": reference_image2.content_type or "image/jpeg",
                "filename": reference_image2.filename or "reference_image_2"
            })

        # Третье изображение (опционально)
        if reference_image3:
            img3_data = await reference_image3.read()
            reference_images.append({
                "data": img3_data,
                "content_type": reference_image3.content_type or "image/jpeg",
                "filename": reference_image3.filename or "reference_image_3"
            })

        # Видео-референс: либо файл, либо URL
        video_ref = None
        if video_reference:
            # Если загружен файл
            video_bytes = await video_reference.read()
            video_ref = {
                "data": video_bytes,
                "content_type": video_reference.content_type or "video/mp4",
                "filename": video_reference.filename or "reference_video"
            }
        elif video_reference_url:
            # Если указан URL - передаем строку
            video_ref = video_reference_url.strip()

        # Генерируем видео
        try:
            logger.info(f"Генерация видео Veo 3.1: platform={platform}, duration={duration_seconds}s")
            result = veo31_generator.generate_from_references(
                reference_images=reference_images,
                video_reference=video_ref,
                platform=platform,
                use_case=use_case,
                duration_seconds=duration_seconds,
                aspect_ratio=aspect_ratio,
                quality=quality,
                additional_prompt=additional_prompt,
                polish_with_perplexity=polish_with_perplexity,
                prefer_kie_api=prefer_kie_api
            )

            logger.info("Генерация видео завершена успешно")
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Ошибка валидации при генерации видео: {e}")
            return JSONResponse(content={
                "step1_analysis": None,
                "step2_prompt": None,
                "step3_video_generation": {
                    "error": str(e),
                    "status": "failed"
                }
            }, status_code=200)
        except Exception as e:
            logger.error(f"Ошибка в процессе генерации: {e}", exc_info=True)
            return JSONResponse(content={
                "step1_analysis": None,
                "step2_prompt": "Ошибка при анализе референсов",
                "step3_video_generation": {
                    "error": str(e),
                    "status": "failed"
                }
            }, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Критическая ошибка: {str(e)}")


@app.post("/veo31/generate-from-urls")
async def generate_veo31_video_from_urls(
    reference_image_urls: str = Form(...),  # Список URL через запятую
    video_reference_url: Optional[str] = Form(None),
    platform: str = Form("veo3"),
    aspect_ratio: str = Form("16:9"),
    additional_prompt: Optional[str] = Form(None),
    polish_with_perplexity: Optional[bool] = Form(None)
):
    """
    Анализ референсов и генерация промпта по URL (без генерации видео)

    reference_image_urls: URL изображений через запятую (минимум 1, максимум 3)
    """
    if not veo31_generator:
        raise HTTPException(
            status_code=503,
            detail="Veo 3.1 генератор не инициализирован. Проверьте GEMINI_API_KEY в .env"
        )

    try:
        logger.info(f"Анализ и генерация промпта из URL: platform={platform}")

        # Парсим URL изображений
        image_urls = [url.strip() for url in reference_image_urls.split(",") if url.strip()]

        if not image_urls:
            raise HTTPException(status_code=400, detail="Необходимо указать минимум один URL изображения")

        if len(image_urls) > 3:
            logger.warning(f"Указано {len(image_urls)} URL, используем первые 3")
            image_urls = image_urls[:3]  # Берем только первые 3

        logger.info(f"URL изображений: {len(image_urls)} шт.")
        if video_reference_url:
            logger.info(f"Видео-референс URL: {video_reference_url[:100]}...")

        # Генерируем только промпт (без видео)
        result = veo31_generator.analyze_and_generate_prompt(
            reference_images=image_urls,
            video_reference=video_reference_url,
            platform=platform,
            use_case="general",  # Используем общий use_case
            polish_with_perplexity=polish_with_perplexity
        )

        # Сохраняем настройки для последующей генерации видео
        result["generation_settings"] = {
            "aspect_ratio": aspect_ratio,
            "additional_prompt": additional_prompt
        }

        # Совместимость: дублируем поле промпта в ключ "prompt"
        if isinstance(result, dict) and "step2_prompt" in result:
            result.setdefault("prompt", result.get("step2_prompt"))

        logger.info("Анализ и генерация промпта из URL завершены успешно")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Ошибка валидации при анализе из URL: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка анализа из URL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")


@app.post("/veo31/generate-with-prompt-urls")
async def generate_video_with_prompt_urls(
    prompt: str = Form(...),
    reference_image_urls: str = Form(...),
    video_reference_url: Optional[str] = Form(None),
    aspect_ratio: str = Form("16:9"),
    prefer_kie_api: bool = Form(False)
):
    """
    Генерация видео с промптом по URL изображений
    """
    if not veo31_generator:
        raise HTTPException(
            status_code=503,
            detail="Veo 3.1 генератор не инициализирован"
        )

    try:
        logger.info(f"Генерация видео с промптом по URL: aspect={aspect_ratio}, prefer_kie={prefer_kie_api}")

        # Валидация параметров
        valid_aspect_ratios = ["16:9", "9:16", "1:1", "4:3", "21:9"]
        if aspect_ratio not in valid_aspect_ratios:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый aspect_ratio: {aspect_ratio}. Доступны: {', '.join(valid_aspect_ratios)}"
            )

        if not prompt or len(prompt.strip()) < 10:
            raise HTTPException(status_code=400, detail="Промпт слишком короткий (минимум 10 символов)")

        # Парсим URL изображений
        image_urls = [url.strip() for url in reference_image_urls.split(",") if url.strip()]
        if not image_urls:
            raise HTTPException(status_code=400, detail="Необходимо указать минимум один URL изображения")

        # Генерируем видео с готовым промптом
        result = veo31_generator.generate_video_with_prompt(
            prompt=prompt,
            reference_images=image_urls,
            video_reference=video_reference_url.strip() if video_reference_url else None,
            duration_seconds=8,  # Veo всегда генерирует 8 секунд
            aspect_ratio=aspect_ratio,
            quality="standard",  # Не используется, но передается для совместимости
            prefer_kie_api=prefer_kie_api
        )

        logger.info("Генерация видео с промптом по URL завершена успешно")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Ошибка валидации: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка генерации видео с промптом по URL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")


@app.get("/veo31/status/{task_id}")
async def check_veo31_status(task_id: str):
    """Проверка статуса генерации видео"""
    if not veo31_generator:
        raise HTTPException(status_code=503, detail="Veo 3.1 генератор не инициализирован")

    try:
        status = veo31_generator.check_status(task_id)
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка проверки статуса: {str(e)}")


@app.get("/veo31/kie-status/{task_id}")
async def check_kie_status(task_id: str):
    """Проверка статуса задачи Kie.ai и получение видео URL если готово"""
    if not veo31_generator or not veo31_generator.kie_client:
        raise HTTPException(status_code=503, detail="Kie.ai клиент не инициализирован")

    try:
        status = veo31_generator.kie_client.check_task_status(task_id)

        # Форматируем ответ для удобства использования
        result = {
            "provider": "kie.ai",
            "task_id": task_id,
            "status": status.get("status", "unknown"),
            "raw_response": status.get("raw_response")
        }

        # Если видео готово, добавляем video_url
        if status.get("video_url"):
            result["video_url"] = status["video_url"]
            result["status"] = "completed"
            result["method"] = status.get("method", "api")
            logger.info(f"Видео готово для task_id {task_id}: {status['video_url']}")

        # Если есть ошибка или заметка, добавляем их
        if status.get("error"):
            result["error"] = status["error"]
        if status.get("note"):
            result["note"] = status["note"]

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Ошибка проверки статуса Kie.ai для task_id {task_id}: {e}", exc_info=True)

        # Возвращаем статус "processing" вместо исключения, чтобы интерфейс продолжал проверку
        error_msg = str(e)
        return JSONResponse(content={
            "provider": "kie.ai",
            "status": "processing",
            "task_id": task_id,
            "error": error_msg,
            "note": "Ошибка при проверке статуса. Система продолжит проверку. Рекомендуется использовать callback URL или проверять вручную на https://app.kie.ai"
        }, status_code=200)


if __name__ == "__main__":
    try:
        logger.info(f"Запуск сервера на {HOST}:{PORT}")
        uvicorn.run(app, host=HOST, port=PORT)
    except KeyboardInterrupt:
        logger.info("Сервер остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}", exc_info=True)
        sys.exit(1)

