# N8N Workflow Setup Guide

## Quick Setup for Video Prompt Pipeline

### Step 1: Registration and Access

1. Sign up at [n8n.cloud](https://n8n.cloud) (cloud version) or install self-hosted:
   ```bash
   npm install n8n -g
   n8n start
   ```

### Step 2: Import Workflow

1. In N8N create a new workflow
2. Import `n8n_workflow.json`:
   - Click menu (three dots) → Import from File
   - Select `n8n_workflow.json`

### Step 3: API Key Configuration

1. Go to **Settings** → **Credentials**
2. Create credential for OpenAI:
   - Name: "OpenAI API"
   - API Key: your OpenAI key

### Step 4: Node Configuration

#### "Webhook Trigger" Node
- Method: POST
- Path: `video-prompt`
- Response Mode: Last Node

#### "Download Media" Node
- Method: GET
- URL: `={{ $json.body.video_url }}`
- Response Format: File

#### "Agent 1: Visual Analyzer" Node
- Model: `gpt-4o`
- Temperature: `0.3`
- Max Tokens: `2000`
- Add binary data from previous node to attachments

#### "Agent 2: Prompt Generator" Node
- Model: `gpt-4o`
- Temperature: `0.7`
- Max Tokens: `1000`
- System prompt configured automatically

### Step 5: Activation and Testing

1. Save workflow
2. Activate it (toggle in top right corner)
3. Copy Production Webhook URL

### Step 6: Testing

```bash
curl -X POST "YOUR_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/jewelry.jpg",
    "platform": "veo3",
    "use_case": "product_video"
  }'
```

## Alternative: Using FastAPI Server

If you don't want to use N8N directly, you can use our FastAPI server:

1. Start the server:
   ```bash
   cd video-prompt-pipeline
   python api_server.py
   ```

2. In N8N create HTTP Request node:
   - Method: POST
   - URL: `http://localhost:8000/process`
   - Body: Form Data
     - `url`: `={{ $json.body.video_url }}`
     - `platform`: `={{ $json.body.platform }}`
     - `use_case`: `={{ $json.body.use_case }}`

## Example Payloads

### Image for Veo 3
```json
{
  "video_url": "https://example.com/jewelry.jpg",
  "platform": "veo3",
  "use_case": "product_video"
}
```

### Video for Sora 2
```json
{
  "video_url": "https://example.com/jewelry-showcase.mp4",
  "platform": "sora2",
  "use_case": "luxury_brand"
}
```

## Troubleshooting

**Error: "API key not found"**
- Check that credential is created and properly connected to nodes

**Error: "Failed to load media"**
- Check URL availability
- Make sure URL is publicly accessible

**Timeout during video analysis**
- Increase timeout in Download Media node
- For large videos use Gemini (requires additional configuration)
