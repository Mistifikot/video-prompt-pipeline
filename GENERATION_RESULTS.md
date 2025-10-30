# How you will receive video after generation

## 🎥 Video receiving scenarios

### ✅ Scenario 1: You HAVE access to Veo 3.1 API

**What will happen:**

1. **Upload**: You upload 3 images + video reference (optional)
2. **Analysis**: System analyzes your references (5-15 seconds)
3. **Prompt generation**: Optimized prompt for Veo 3.1 is created
4. **Generation launch**: Request is sent to Veo 3.1 API
5. **Waiting**: 11 seconds - 6 minutes (usually 1-2 minutes)
6. **Result in web interface**:
   - ✅ **Video player** — watch video directly in browser
   - 💾 **"Download video" button** — save MP4 file to computer
   - 🔗 **"Copy link" button** — get direct link to video

**Where video will be:**
- **In web interface** — built-in player for viewing
- **On Google servers** — direct link valid for 48 hours
- **On your computer** — after downloading

---

### ⚠️ Scenario 2: You DON'T have access to Veo 3.1 API (current situation)

**What will happen:**

1. **Upload**: You upload 3 images + video reference (optional)
2. **Analysis**: System analyzes your references (5-15 seconds)
3. **Prompt generation**: Optimized prompt for Veo 3.1 is created
4. **Generation attempt**: API will return access error
5. **Result in web interface**:
   - ✅ **Ready prompt** — professionally composed description
   - 📋 **"Copy prompt" button** — copy to clipboard
   - 💡 **Instructions** — how to use prompt in Google AI Studio

**What to do next:**
1. Copy the prompt (button in interface)
2. Open [Google AI Studio](https://aistudio.google.com)
3. Select **Veo 3.1** model
4. Paste the prompt
5. Upload 3 images
6. Click Generate
7. Wait for result (1-6 minutes)
8. Download video from AI Studio

---

## 📊 Result structure in API

### Successful generation:
```json
{
  "step1_analysis": { /* reference analysis */ },
  "step2_prompt": "A cinematic shot of...",
  "step3_video_generation": {
    "status": "completed",
    "video_url": "https://generativelanguage.googleapis.com/.../video.mp4",
    "duration": 8,
    "resolution": "1080p"
  }
}
```

### Generation in progress:
```json
{
  "step1_analysis": { /* reference analysis */ },
  "step2_prompt": "A cinematic shot of...",
  "step3_video_generation": {
    "status": "processing",
    "task_id": "task_xyz123"
  }
}
```
*Interface automatically checks status every 10 seconds*

### Access error:
```json
{
  "step1_analysis": { /* reference analysis */ },
  "step2_prompt": "A cinematic shot of...",
  "step3_video_generation": {
    "status": "failed",
    "error": "400 Client Error: Bad Request"
  }
}
```
*You will still get prompt for manual use*

---

## 🖥️ How result looks in web interface

### If video generated:
```
┌─────────────────────────────────────┐
│ ✓ Analysis completed                │
├─────────────────────────────────────┤
│ Generated prompt:                    │
│ ┌─────────────────────────────────┐ │
│ │ A cinematic shot of a beautiful │ │
│ │ woman with dark hair...         │ │
│ └─────────────────────────────────┘ │
│ [📋 Copy prompt]                    │
├─────────────────────────────────────┤
│ ✓ Video ready!                      │
│ ┌─────────────────────────────────┐ │
│ │  ▶ VIDEO PLAYER                 │ │
│ │  [===============]              │ │
│ └─────────────────────────────────┘ │
│ [💾 Download video]                 │
│ [🔗 Copy link]                      │
└─────────────────────────────────────┘
```

### If no API access:
```
┌─────────────────────────────────────┐
│ ✓ Analysis completed                │
├─────────────────────────────────────┤
│ Generated prompt:                    │
│ ┌─────────────────────────────────┐ │
│ │ A cinematic shot of a beautiful │ │
│ │ woman with dark hair...         │ │
│ └─────────────────────────────────┘ │
│ [📋 Copy prompt]                    │
├─────────────────────────────────────┤
│ ⚠ Video generation unavailable      │
│ 400 Client Error: Bad Request       │
│                                     │
│ What to do:                         │
│ 1. Copy prompt above                │
│ 2. Open Google AI Studio           │
│ 3. Paste prompt and generate       │
└─────────────────────────────────────┘
```

---

## ⏱️ How long video is stored

- **On Google servers**: 48 hours (2 days)
- **Recommendation**: Download video immediately after generation
- **Link will stop working** after 2 days

---

## 💡 Recommendations

1. **Download immediately**: Don't postpone video download
2. **Save prompt**: Even if video is generated, prompt may be useful
3. **Check size**: Veo 3.1 generates videos ~5-50MB (depends on duration)
4. **Use right-click**: "Save video as..." to choose folder

---

## 🔗 Video link format

Typical link from Veo 3.1:
```
https://generativelanguage.googleapis.com/v1beta/files/xyz123abc/download?key=AIza...
```

This link:
- ✅ Works in browser
- ✅ Can be inserted in video tag
- ✅ Can be downloaded directly
- ⏱️ Valid for 48 hours

