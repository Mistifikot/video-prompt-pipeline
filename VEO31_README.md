# Veo 3.1 - Video Generation from Samples

## Description

Complete workflow for generating videos in Google Veo 3.1 based on reference images and videos.

## Features

### 1. Reference Analysis
- Analysis of 3 reference images (minimum 1)
- Optional analysis of reference video to understand movement style
- Automatic merging of analyses

### 2. Prompt Generation
- Creating a prompt based on analysis
- Optimization for Veo 3.1
- Ability to add additional instructions

### 3. Video Generation
- Sending request to Veo 3.1 API
- Support for reference images
- Support for video reference
- Parameter configuration (duration, aspect ratio, quality)

## Usage

### Option 1: Web Interface

1. Start the server: `start_server.bat`
2. Open the interface: `start_veo31_ui.bat`
   Or manually: `veo31_interface.html`
3. Upload:
   - 3 images (minimum 1)
   - Optionally: video reference
4. Configure parameters
5. Click "Generate video"

### Option 2: Via API

#### File upload:
```bash
curl -X POST "http://localhost:8000/veo31/generate" \
  -F "reference_image1=@image1.jpg" \
  -F "reference_image2=@image2.jpg" \
  -F "reference_image3=@image3.jpg" \
  -F "video_reference=@reference.mp4" \
  -F "use_case=product_video" \
  -F "duration_seconds=10" \
  -F "aspect_ratio=16:9" \
  -F "quality=high"
```

#### Via URL:
```bash
curl -X POST "http://localhost:8000/veo31/generate-from-urls" \
  -F "reference_image_urls=https://example.com/img1.jpg,https://example.com/img2.jpg,https://example.com/img3.jpg" \
  -F "video_reference_url=https://example.com/ref.mp4" \
  -F "use_case=product_video" \
  -F "duration_seconds=10"
```

### Option 3: Python code

```python
from veo31_generator import Veo31VideoGenerator

generator = Veo31VideoGenerator()

result = generator.generate_from_references(
    reference_images=[
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
        "https://example.com/image3.jpg"
    ],
    video_reference="https://example.com/reference.mp4",  # Optional
    use_case="product_video",
    duration_seconds=10,
    aspect_ratio="16:9",
    quality="high"
)

print(result["step2_prompt"])  # Generated prompt
print(result["step3_video_generation"]["video_url"])  # Video URL
```

## Parameters

### References
- `reference_images`: List of 1-3 images (URL, path or bytes)
- `video_reference`: Optional video for movement style analysis

### Generation
- `duration_seconds`: Video duration (5-60 seconds)
- `aspect_ratio`: Aspect ratio ("16:9", "9:16", "1:1")
- `quality`: Quality ("high", "standard")
- `use_case`: Task type (product_video, hero_image, gemstone_closeup, luxury_brand, general)
- `additional_prompt`: Additional instructions for the prompt

## Workflow

```
1. Upload references
   ↓
2. Analyze images + video (if available)
   ↓
3. Generate prompt based on analysis
   ↓
4. Send to Veo 3.1 API with references and prompt
   ↓
5. Get video
```

## Status Check

If video is generated asynchronously:
```bash
curl "http://localhost:8000/veo31/status/{task_id}"
```

Or in Python:
```python
generator.check_status(task_id)
```

## Important

⚠️ **Note about Veo 3.1 API:**
The exact endpoint and request structure may differ from the implementation.
Current implementation uses structure based on Gemini API.
When official Veo 3.1 API documentation is received, updates may be required:
- `veo31_client.py` - change endpoint and request format
- May require using Vertex AI instead of Generative AI API

## Requirements

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` in .env file (for Veo 3.1)
- `OPENAI_API_KEY` (for image analysis)
- Internet connection for API work

## Troubleshooting

**Error: "Veo 3.1 generator not initialized"**
- Check that `GEMINI_API_KEY` exists in .env file
- Restart the server

**Error: "API endpoint not found"**
- Veo 3.1 API may use a different endpoint
- Check official Google documentation
- May require using Vertex AI

**Video generation takes too long**
- Video generation can take several minutes
- Use status check via `/veo31/status/{task_id}`
