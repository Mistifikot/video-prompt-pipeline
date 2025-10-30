"""
FastAPI server for accessing pipeline via REST API
"""

import os
import sys
# Set UTF-8 encoding for output (especially important for Windows cmd)
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

# Import configuration and logging
try:
    from config import (
        OPENAI_API_KEY, GEMINI_API_KEY, PPLX_API_KEY, KIE_API_KEY,
        PORT, HOST, DEBUG, validate_config
    )
    from logger_utils import logger
except ImportError:
    # Fallback if modules not found
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
            errors.append("OPENAI_API_KEY not found")
        return len(errors) == 0, errors

from pipeline import VideoPromptPipeline
from workflow_manager import VideoImagePromptWorkflow

# Import Veo 3.1 generator (optional)
try:
    from veo31_generator import Veo31VideoGenerator
    VEO31_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Veo 3.1 module unavailable: {e}")
    VEO31_AVAILABLE = False
    Veo31VideoGenerator = None

# Configuration validation
is_valid, config_errors = validate_config()
if not is_valid:
    for error in config_errors:
        logger.error(error)
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for application to work!")

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

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
try:
    pipeline = VideoPromptPipeline(
        openai_api_key=OPENAI_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        use_gemini_for_analysis=os.getenv("USE_GEMINI", "false").lower() == "true",
        perplexity_api_key=PPLX_API_KEY,
        auto_perplexity_polish=None
    )
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Pipeline initialization error: {e}")
    raise

try:
    workflow_orchestrator = VideoImagePromptWorkflow(
        pipeline=pipeline,
        kie_api_key=KIE_API_KEY,
    )
    logger.info("Workflow orchestrator initialized")
except Exception as workflow_error:
    workflow_orchestrator = None
    logger.error(f"Failed to initialize workflow orchestrator: {workflow_error}")

# Initialize Veo 3.1 generator
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
            logger.info("Veo 3.1 generator initialized")
        else:
            logger.info("Veo 3.1 generator not initialized (no Gemini or Kie API key)")
    except Exception as e:
        logger.error(f"Failed to initialize Veo 3.1 generator: {e}", exc_info=True)
else:
    logger.info("Veo 3.1 functions unavailable (module not loaded)")


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
    """Web interface"""
    ui_path = Path(__file__).parent / "web_interface.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    else:
        return {"error": "UI file not found"}

@app.get("/veo31")
async def serve_veo31_ui():
    """Web interface for Veo 3.1"""
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
    Agent 1: Media analysis (image/video)

    Accepts either URL or file
    """
    try:
        if not url and not file:
            raise HTTPException(status_code=400, detail="Must specify url or upload file")

        if url:
            logger.info(f"Media analysis by URL: {url[:100]}...")
            scene_analysis = pipeline.analyzer.analyze(
                media_source=url,
                use_gemini=use_gemini
            )
        else:
            logger.info(f"Analysis of uploaded file: {file.filename}")
            file_data = await file.read()
            scene_analysis = pipeline.analyzer.analyze(
                media_source=file_data,
                content_type=file.content_type,
                use_gemini=use_gemini
            )

        logger.info("Analysis completed successfully")
        return JSONResponse(content=scene_analysis)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Media analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/workflow/start")
async def workflow_start(
    video_url: str = Form(...),
    image_url: str = Form(...),
    platform: str = Form("veo3"),
    use_case: str = Form("general"),
    polish_with_perplexity: Optional[str] = Form(None),
):
    """Full workflow start: video analysis, image analysis, draft prompt preparation."""
    if workflow_orchestrator is None:
        raise HTTPException(status_code=500, detail="Workflow orchestrator not initialized")

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
        logger.error(f"Workflow start error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow start error: {exc}")


@app.get("/workflow/{workflow_id}")
async def workflow_state(workflow_id: str):
    """Returns saved workflow state (draft prompt + analyses)."""
    if workflow_orchestrator is None:
        raise HTTPException(status_code=503, detail="Workflow orchestrator not initialized")

    try:
        state = workflow_orchestrator.get_workflow_state(workflow_id)
        return JSONResponse(content=state.to_public_dict())
    except KeyError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@app.get("/workflow/{workflow_id}/status")
async def workflow_status(workflow_id: str):
    """Checks video generation status for workflow via Kie.ai."""
    if workflow_orchestrator is None:
        raise HTTPException(status_code=503, detail="Workflow orchestrator not initialized")

    try:
        status_payload = workflow_orchestrator.refresh_generation_status(workflow_id)
        return JSONResponse(content=status_payload)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive branch for unexpected API errors
        logger.error(f"Workflow status check error {workflow_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Status check error: {exc}")


@app.post("/workflow/submit")
async def workflow_submit(
    workflow_id: str = Form(...),
    prompt_override: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    model: str = Form("veo3_fast"),
    aspect_ratio: str = Form("16:9"),
    enable_translation: Optional[str] = Form(None),
):
    """Accepts user-confirmed prompt and sends task to Kie.ai."""
    if workflow_orchestrator is None:
        raise HTTPException(status_code=500, detail="Workflow orchestrator not initialized")

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
        logger.error(f"Workflow completion error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow completion error: {exc}")


@app.post("/generate")
async def generate_prompt(
    scene_description: str = Form(...),
    platform: str = Form("veo3"),
    use_case: str = Form("general")
):
    """
    Agent 2: Prompt generation from scene description

    scene_description - JSON string with scene description
    """
    try:
        import json

        # Platform validation
        valid_platforms = ["veo3", "sora2", "seedream"]
        if platform not in valid_platforms:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported platform: {platform}. Available: {', '.join(valid_platforms)}"
            )

        scene_dict = json.loads(scene_description)
        logger.info(f"Prompt generation for platform: {platform}, use_case: {use_case}")

        result = pipeline.generator.generate(
            scene_description=scene_dict,
            platform=platform,
            use_case=use_case
        )

        logger.info("Prompt generated successfully")
        return JSONResponse(content=result)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in scene_description: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prompt generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


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
    Full pipeline: analysis + prompt generation

    Accepts either URL or file
    Returns ready prompt
    """
    try:
        # Temporarily switch Gemini usage if specified
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
            raise HTTPException(status_code=400, detail="Must specify url or upload file")

        # Restore original value
        pipeline.use_gemini = original_use_gemini

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"External API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/process-multiple")
async def process_multiple_platforms(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    platforms: str = Form("veo3,sora2"),  # Comma-separated list
    use_case: str = Form("general"),
    use_gemini: bool = Form(False)
):
    """
    Full pipeline for multiple platforms simultaneously
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
            raise HTTPException(status_code=400, detail="Must specify url or upload file")

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"External API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


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
    Only reference analysis and prompt generation (without video generation)
    Allows editing prompt before generation
    """
    if not veo31_generator:
        raise HTTPException(
            status_code=503,
            detail="Veo 3.1 generator not initialized. Check OPENAI_API_KEY in .env"
        )

    try:
        logger.info(f"Analysis and prompt generation: platform={platform}")

        # Collect list of reference images
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

        # Video reference: either file or URL
        video_ref = None
        if video_reference:
            video_bytes = await video_reference.read()
            video_ref = {
                "data": video_bytes,
                "content_type": video_reference.content_type or "video/mp4",
                "filename": video_reference.filename or "reference_video"
            }
            logger.info(f"Video reference uploaded: {video_reference.filename}")
        elif video_reference_url:
            video_ref = video_reference_url.strip()
            logger.info(f"Video reference URL: {video_reference_url[:100]}...")

        # Perform only analysis and prompt generation
        result = veo31_generator.analyze_and_generate_prompt(
            reference_images=reference_images,
            video_reference=video_ref,
            platform=platform,
            use_case="general",  # Always use general use_case
            polish_with_perplexity=polish_with_perplexity
        )

        # Compatibility: duplicate prompt field in "prompt" key
        if isinstance(result, dict) and "step2_prompt" in result:
            result.setdefault("prompt", result.get("step2_prompt"))

        logger.info("Analysis and prompt generation completed successfully")
        return JSONResponse(content=result)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis and prompt generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


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
    Video generation with ready prompt (after editing)

    Args:
        prompt: Ready prompt (possibly edited)
        reference_images: Reference images
        video_reference: Optional video reference
        aspect_ratio: Aspect ratio
        prefer_kie_api: Use Kie.ai API
    """
    if not veo31_generator:
        raise HTTPException(
            status_code=503,
            detail="Veo 3.1 generator not initialized"
        )

    try:
        logger.info(f"Video generation with prompt: aspect={aspect_ratio}, prefer_kie={prefer_kie_api}")

        # Parameter validation
        valid_aspect_ratios = ["16:9", "9:16", "1:1", "4:3", "21:9"]
        if aspect_ratio not in valid_aspect_ratios:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported aspect_ratio: {aspect_ratio}. Available: {', '.join(valid_aspect_ratios)}"
            )

        if not prompt or len(prompt.strip()) < 10:
            raise HTTPException(status_code=400, detail="Prompt too short (minimum 10 characters)")

        # Veo always generates 8 seconds, quality veo3_fast
        duration_seconds = 8
        quality = "standard"  # Not used directly, but passed for compatibility

        # Collect list of reference images
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

        # Video reference
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

        # Generate video with ready prompt
        # Veo always generates 8 seconds, quality veo3_fast
        result = veo31_generator.generate_video_with_prompt(
            prompt=prompt,
            reference_images=reference_images,
            video_reference=video_ref,
            duration_seconds=8,  # Veo always generates 8 seconds
            aspect_ratio=aspect_ratio,
            quality="standard",  # Not used, but passed for compatibility
            prefer_kie_api=prefer_kie_api
        )

        logger.info("Video generation completed successfully")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Video generation with prompt error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


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
    Video generation in Veo 3.1 with reference images

    Minimum 1 image required, can upload up to 3
    Optionally can upload video reference
    """
    if not veo31_generator:
        raise HTTPException(
            status_code=503,
            detail="Veo 3.1 generator not initialized. Check GEMINI_API_KEY in .env"
        )

    try:
        # Collect list of reference images
        reference_images = []

        # First image is required
        img1_data = await reference_image1.read()
        reference_images.append({
            "data": img1_data,
            "content_type": reference_image1.content_type or "image/jpeg",
            "filename": reference_image1.filename or "reference_image_1"
        })

        # Second image (optional)
        if reference_image2:
            img2_data = await reference_image2.read()
            reference_images.append({
                "data": img2_data,
                "content_type": reference_image2.content_type or "image/jpeg",
                "filename": reference_image2.filename or "reference_image_2"
            })

        # Third image (optional)
        if reference_image3:
            img3_data = await reference_image3.read()
            reference_images.append({
                "data": img3_data,
                "content_type": reference_image3.content_type or "image/jpeg",
                "filename": reference_image3.filename or "reference_image_3"
            })

        # Video reference: either file or URL
        video_ref = None
        if video_reference:
            # If file uploaded
            video_bytes = await video_reference.read()
            video_ref = {
                "data": video_bytes,
                "content_type": video_reference.content_type or "video/mp4",
                "filename": video_reference.filename or "reference_video"
            }
        elif video_reference_url:
            # If URL specified - pass string
            video_ref = video_reference_url.strip()

        # Generate video
        try:
            logger.info(f"Veo 3.1 video generation: platform={platform}, duration={duration_seconds}s")
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

            logger.info("Video generation completed successfully")
            return JSONResponse(content=result)

        except ValueError as e:
            logger.error(f"Validation error during video generation: {e}")
            return JSONResponse(content={
                "step1_analysis": None,
                "step2_prompt": None,
                "step3_video_generation": {
                    "error": str(e),
                    "status": "failed"
                }
            }, status_code=200)
        except Exception as e:
            logger.error(f"Error during generation process: {e}", exc_info=True)
            return JSONResponse(content={
                "step1_analysis": None,
                "step2_prompt": "Error analyzing references",
                "step3_video_generation": {
                    "error": str(e),
                    "status": "failed"
                }
            }, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Critical error: {str(e)}")


@app.post("/veo31/generate-from-urls")
async def generate_veo31_video_from_urls(
    reference_image_urls: str = Form(...),  # Comma-separated list of URLs
    video_reference_url: Optional[str] = Form(None),
    platform: str = Form("veo3"),
    aspect_ratio: str = Form("16:9"),
    additional_prompt: Optional[str] = Form(None),
    polish_with_perplexity: Optional[bool] = Form(None)
):
    """
    Reference analysis and prompt generation by URL (without video generation)

    reference_image_urls: Comma-separated image URLs (minimum 1, maximum 3)
    """
    if not veo31_generator:
        raise HTTPException(
            status_code=503,
            detail="Veo 3.1 generator not initialized. Check GEMINI_API_KEY in .env"
        )

    try:
        logger.info(f"Analysis and prompt generation from URL: platform={platform}")

        # Parse image URLs
        image_urls = [url.strip() for url in reference_image_urls.split(",") if url.strip()]

        if not image_urls:
            raise HTTPException(status_code=400, detail="Must specify at least one image URL")

        if len(image_urls) > 3:
            logger.warning(f"{len(image_urls)} URLs specified, using first 3")
            image_urls = image_urls[:3]  # Take only first 3

        logger.info(f"Image URLs: {len(image_urls)} items")
        if video_reference_url:
            logger.info(f"Video reference URL: {video_reference_url[:100]}...")

        # Generate only prompt (without video)
        result = veo31_generator.analyze_and_generate_prompt(
            reference_images=image_urls,
            video_reference=video_reference_url,
            platform=platform,
            use_case="general",  # Use general use_case
            polish_with_perplexity=polish_with_perplexity
        )

        # Save settings for subsequent video generation
        result["generation_settings"] = {
            "aspect_ratio": aspect_ratio,
            "additional_prompt": additional_prompt
        }

        # Compatibility: duplicate prompt field in "prompt" key
        if isinstance(result, dict) and "step2_prompt" in result:
            result.setdefault("prompt", result.get("step2_prompt"))

        # Check if there were errors
        if isinstance(result, dict):
            if result.get("error") or (result.get("step2_prompt", "").startswith("Prompt generation error:") or
                                       result.get("step2_prompt", "").startswith("Error:")):
                logger.warning(f"Analysis completed with errors: {result.get('error', 'Unknown error')}")
                return JSONResponse(content=result, status_code=500)
            elif not result.get("ready_for_editing", True):
                logger.warning("Analysis completed but not ready for editing")
                return JSONResponse(content=result, status_code=500)

        logger.info("Analysis and prompt generation from URL completed successfully")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error during URL analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"URL analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/veo31/generate-with-prompt-urls")
async def generate_video_with_prompt_urls(
    prompt: str = Form(...),
    reference_image_urls: str = Form(...),
    video_reference_url: Optional[str] = Form(None),
    aspect_ratio: str = Form("16:9"),
    prefer_kie_api: bool = Form(False)
):
    """
    Video generation with prompt by image URLs
    """
    if not veo31_generator:
        raise HTTPException(
            status_code=503,
            detail="Veo 3.1 generator not initialized"
        )

    try:
        logger.info(f"Video generation with prompt by URL: aspect={aspect_ratio}, prefer_kie={prefer_kie_api}")

        # Parameter validation
        valid_aspect_ratios = ["16:9", "9:16", "1:1", "4:3", "21:9"]
        if aspect_ratio not in valid_aspect_ratios:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported aspect_ratio: {aspect_ratio}. Available: {', '.join(valid_aspect_ratios)}"
            )

        if not prompt or len(prompt.strip()) < 10:
            raise HTTPException(status_code=400, detail="Prompt too short (minimum 10 characters)")

        # Parse image URLs
        image_urls = [url.strip() for url in reference_image_urls.split(",") if url.strip()]
        if not image_urls:
            raise HTTPException(status_code=400, detail="Must specify at least one image URL")

        # Generate video with ready prompt
        result = veo31_generator.generate_video_with_prompt(
            prompt=prompt,
            reference_images=image_urls,
            video_reference=video_reference_url.strip() if video_reference_url else None,
            duration_seconds=8,  # Veo always generates 8 seconds
            aspect_ratio=aspect_ratio,
            quality="standard",  # Not used, but passed for compatibility
            prefer_kie_api=prefer_kie_api
        )

        logger.info("Video generation with prompt by URL completed successfully")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Video generation with prompt by URL error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/veo31/status/{task_id}")
async def check_veo31_status(task_id: str):
    """Check video generation status"""
    if not veo31_generator:
        raise HTTPException(status_code=503, detail="Veo 3.1 generator not initialized")

    try:
        status = veo31_generator.check_status(task_id)
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check error: {str(e)}")


@app.get("/veo31/kie-status/{task_id}")
async def check_kie_status(task_id: str):
    """Check Kie.ai task status and get video URL if ready"""
    if not veo31_generator or not veo31_generator.kie_client:
        raise HTTPException(status_code=503, detail="Kie.ai client not initialized")

    try:
        status = veo31_generator.kie_client.check_task_status(task_id)

        # Format response for convenience
        result = {
            "provider": "kie.ai",
            "task_id": task_id,
            "status": status.get("status", "unknown"),
            "raw_response": status.get("raw_response")
        }

        # If video ready, add video_url
        # Also check raw_response for video URLs if not in status
        video_url = status.get("video_url")
        if not video_url and status.get("raw_response"):
            raw = status.get("raw_response")
            # Check various formats in raw_response
            if isinstance(raw, dict):
                # Check data.data.info.resultUrls format (nested data structure)
                if raw.get("data", {}).get("data", {}).get("info", {}).get("resultUrls"):
                    result_urls = raw["data"]["data"]["info"]["resultUrls"]
                    if isinstance(result_urls, list) and len(result_urls) > 0:
                        video_url = result_urls[0]
                    elif isinstance(result_urls, str):
                        video_url = result_urls
                # Check data.data.info.resultUrls format (single data level)
                if not video_url and raw.get("data", {}).get("info", {}).get("resultUrls"):
                    result_urls = raw["data"]["info"]["resultUrls"]
                    if isinstance(result_urls, list) and len(result_urls) > 0:
                        video_url = result_urls[0]
                    elif isinstance(result_urls, str):
                        video_url = result_urls
                # Check data.data.resultUrls format (nested data without info)
                if not video_url and raw.get("data", {}).get("data", {}).get("resultUrls"):
                    result_urls = raw["data"]["data"]["resultUrls"]
                    if isinstance(result_urls, list) and len(result_urls) > 0:
                        video_url = result_urls[0]
                    elif isinstance(result_urls, str):
                        video_url = result_urls
                # Check data.data.resultUrls format (single data level without info)
                if not video_url and raw.get("data", {}).get("resultUrls"):
                    result_urls = raw["data"]["resultUrls"]
                    if isinstance(result_urls, list) and len(result_urls) > 0:
                        video_url = result_urls[0]
                    elif isinstance(result_urls, str):
                        video_url = result_urls
                # Check result.video_url format
                if not video_url and raw.get("result", {}).get("video_url"):
                    video_url = raw["result"]["video_url"]
                # Check direct video_url
                if not video_url and raw.get("video_url"):
                    video_url = raw["video_url"]

        if video_url:
            result["video_url"] = video_url
            result["status"] = "completed"
            result["method"] = status.get("method", "api")
            logger.info(f"Video ready for task_id {task_id}: {video_url}")

        # If error or note exists, add them
        if status.get("error"):
            result["error"] = status["error"]
        if status.get("note"):
            result["note"] = status["note"]

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Kie.ai status check error for task_id {task_id}: {e}", exc_info=True)

        # Return "processing" status instead of exception so interface continues checking
        error_msg = str(e)
        return JSONResponse(content={
            "provider": "kie.ai",
            "status": "processing",
            "task_id": task_id,
            "error": error_msg,
            "note": "Error checking status. System will continue checking. Recommended to use callback URL or check manually at https://app.kie.ai"
        }, status_code=200)


if __name__ == "__main__":
    try:
        logger.info(f"Starting server on {HOST}:{PORT}")
        uvicorn.run(app, host=HOST, port=PORT)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server startup error: {e}", exc_info=True)
        sys.exit(1)

