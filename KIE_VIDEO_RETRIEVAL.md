# Retrieving Videos from Kie.ai After Generation

## What was done

Updated the system to automatically retrieve ready videos from Kie.ai API after generation completion.

## Changes

### 1. `kie_api_client.py`
- ✅ Fixed status check endpoint: `https://api.kie.ai/v1/video/task/{task_id}`
- ✅ Added response parsing to extract `video_url` from different response formats
- ✅ `check_task_status()` method now returns structured response with `video_url` if video is ready

### 2. `veo31_generator.py`
- ✅ Added `check_kie_status()` method for checking Kie.ai status
- ✅ Updated `check_status()` method with support for `provider` parameter

### 3. `api_server.py`
- ✅ Updated `/veo31/kie-status/{task_id}` endpoint to return `video_url` in structured format

### 4. `veo31_interface.html`
- ✅ Improved automatic status check logic
- ✅ Added support for getting `video_url` from updated response format
- ✅ Automatic video player display when video is ready
- ✅ Improved handling of different statuses (`processing`, `completed`, `failed`)

## Usage

### Via API

#### 1. Start generation
```bash
curl -X POST "http://localhost:8000/veo31/generate-from-urls" \
  -F "reference_image_urls=https://images.unsplash.com/photo-1656543802898-41c8c46683a7" \
  -F "prefer_kie_api=true" \
  -F "use_case=product_video" \
  -F "duration_seconds=5" \
  -F "aspect_ratio=16:9"
```

Response will contain `task_id`:
```json
{
  "step3_video_generation": {
    "provider": "kie.ai",
    "status": "submitted",
    "task_id": "88513be422a2f971886273b32867fd6f"
  }
}
```

#### 2. Check status and get video
```bash
curl "http://localhost:8000/veo31/kie-status/88513be422a2f971886273b32867fd6f"
```

Response when video is ready:
```json
{
  "provider": "kie.ai",
  "task_id": "88513be422a2f971886273b32867fd6f",
  "status": "completed",
  "video_url": "https://..."
}
```

### Via Web Interface

1. Open `veo31_interface.html` or `http://localhost:8000/veo31`
2. Upload images or specify URLs
3. Select "Use Kie.ai API"
4. Click "Generate video"
5. Interface will automatically check status every 10 seconds
6. When video is ready, it will automatically display in the player

## Kie.ai API Response Format

According to documentation, endpoint `/v1/video/task/{task_id}` returns:

```json
{
  "status": "completed",
  "result": {
    "video_url": "https://tempfile.aiquickdraw.com/v/8b95fc92b0a5e9fff2b13256dad8b135_1761820876.mp4"
  }
}
```

Our code also supports alternative formats:
- `status.video_url` (directly in root)
- `status.data.video_url` (via data)
- `status.result.videoUrl` (camelCase)
- `status.result.video` (just "video")
- `status.url` (if it's a video link)
- Automatic search for links to `tempfile.aiquickdraw.com` via regex

**Video link format:**
Videos from Kie.ai are usually returned in format:
```
https://tempfile.aiquickdraw.com/v/{hash}_{timestamp}.mp4
```

These links are directly accessible and can be used in `<video>` tags or for downloading.

## Automatic Status Check

Web interface automatically checks status:
- **Every 10 seconds** for Kie.ai
- **Every 5 seconds** for Google Veo

Checking stops when:
- Video is ready (`status: "completed"` and `video_url` exists)
- Error occurred (`status: "failed"` or `status: "error"`)

## Troubleshooting

### Video not displaying
1. Check that `task_id` is correct
2. Make sure generation is completed on Kie.ai platform
3. Check browser console for errors
4. Check server logs for `[OK] Video ready!`

### Endpoint not working
If endpoint `/v1/video/task/{task_id}` doesn't work:
- Check that API key is correct
- Check that base_url is correct: `https://api.kie.ai`
- Check Kie.ai documentation for API changes

### Status shows "unknown"
This means endpoint was not found. System will continue checking, but you can:
- Check status manually at https://app.kie.ai
- Use task_id from response for manual check

## Example Successful Workflow

1. **Generation started:**
   ```json
   {
     "step3_video_generation": {
       "provider": "kie.ai",
       "status": "submitted",
       "task_id": "abc123"
     }
   }
   ```

2. **Status check (video still generating):**
   ```json
   {
     "status": "processing",
     "task_id": "abc123"
   }
   ```

3. **Video ready:**
   ```json
   {
     "status": "completed",
     "task_id": "abc123",
     "video_url": "https://cdn.kie.ai/videos/abc123.mp4"
   }
   ```

4. **In interface automatically:**
   - Status checking stops
   - Video player displays
   - "Download" and "Copy link" buttons appear
