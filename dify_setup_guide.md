# Dify Workflow Setup Guide

## Quick Setup for Video Prompt Pipeline

### Step 1: Registration

1. Sign up at [dify.ai](https://dify.ai)
2. Sign in to your account

### Step 2: API Key Configuration

1. Go to **Settings** → **Model Provider**
2. Add OpenAI API Key:
   - Provider: OpenAI
   - API Key: your OpenAI key

3. (Optional) Add Google Gemini:
   - Provider: Google
   - API Key: your Gemini key

### Step 3: Create Workflow

1. Go to **Workflows** → **Create Workflow**
2. Select type: **Chatflow**
3. Use YAML import:

   - Click menu (three dots) → **Import from YAML**
   - Upload `dify_workflow.yaml`

   OR create manually:

#### Start Node (Inputs)
- Add variables:
  - `video_url` (text-input, required)
  - `platform` (select: veo3, sora2, seedream)
  - `use_case` (select: product_video, hero_image, gemstone_closeup, luxury_brand, general)

#### HTTP Request Node (Download Media)
- Type: HTTP Request
- Method: GET
- URL: `{{start.video_url}}`
- Response Format: File/Binary

#### LLM Node (Agent 1: Visual Analyzer)
- Model: OpenAI GPT-4o
- Temperature: 0.3
- Max Tokens: 2000
- Prompt:
  ```
  Analyze this image/video in extreme detail as a professional jewelry photographer.

  [Upload file from HTTP Request node]

  Extract and describe:
  1. Main objects
  2. Materials
  3. Lighting
  4. Camera work
  5. Composition
  6. Background and surfaces
  7. Motion and dynamics
  8. Style and mood

  Output as JSON with these fields: main_objects, materials, lighting, camera_work, composition, background_surfaces, motion_dynamics, style_mood
  ```

#### LLM Node (Agent 2: Prompt Generator)
- Model: OpenAI GPT-4o
- Temperature: 0.7
- Max Tokens: 1000
- Prompt:
  ```
  You are an expert prompt engineer for AI video generators.

  Platform: {{start.platform}}
  Use case: {{start.use_case}}

  Create an optimized prompt based on this scene analysis:
  {{analyze-node.text}}

  Rules for {{start.platform}}:
  - [Platform-specific rules]

  Return ONLY the final prompt in English, ready to use.
  ```

#### Answer Node
- Output: `{{generate-node.text}}`
- Metadata: Include platform and use_case

### Step 4: Publish and API

1. Save workflow
2. Click **Publish**
3. Go to **API Access**
4. Create API Key

### Step 5: Testing

#### Via API:
```bash
curl -X POST 'https://api.dify.ai/v1/chat-messages' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": {
      "video_url": "https://example.com/jewelry.jpg",
      "platform": "veo3",
      "use_case": "product_video"
    },
    "user": "user-123"
  }'
```

#### Via UI:
1. Open workflow
2. Fill inputs:
   - Video URL
   - Platform
   - Use Case
3. Click **Run**

## Alternative: Using Python API Server

If you want to use our FastAPI server with Dify:

1. Start the server:
   ```bash
   cd video-prompt-pipeline
   python api_server.py
   ```

2. In Dify create HTTP Request node:
   - URL: `http://your-server:8000/process`
   - Method: POST
   - Body: Form Data
     - `url`: `{{start.video_url}}`
     - `platform`: `{{start.platform}}`
     - `use_case`: `{{start.use_case}}`

## Dify Features

- **Visual Editor**: Easy workflow configuration via UI
- **Versioning**: Workflow version saving
- **Analytics**: Usage tracking
- **Team Collaboration**: Team work

## Troubleshooting

**Error: "File upload failed"**
- Make sure HTTP Request node correctly downloads the file
- Check URL availability

**Error: "Model provider not configured"**
- Check API key settings in Settings

**Timeout during analysis**
- Increase timeout for HTTP Request node
- For videos use Gemini (works better with video)
