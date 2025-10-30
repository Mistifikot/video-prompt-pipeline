# Veo 3.1 API - Important Information

## 🔑 How to Get Access

### Step 1: Get Gemini API Key

1. Open [Google AI Studio](https://aistudio.google.com)
2. Sign in with Google account
3. Go to "API Keys" section
4. Click "Create API key"
5. Copy the key (starts with `AIza`)
6. Add to `.env` file: `GEMINI_API_KEY=AIzaXXXXXXXX`

### Step 2: Check Veo 3.1 Access

⚠️ **IMPORTANT**: Veo 3.1 API may be unavailable if:
- You don't have access to preview (closed beta)
- API is not yet launched in your region
- Special approval from Google is required

## 💰 Pricing

- **Veo 3.1 Standard**: $0.40 per second of video
- **Veo 3.1 Fast**: $0.15 per second of video
- **8-second video**: $3.20 (Standard) or $1.20 (Fast)

## 📋 Reference Image Requirements

- **Quantity**: 1-3 images
- **Format**: PNG, JPEG, JPG
- **Size**: up to 8MB each
- **Resolution**: 720p or higher recommended
- **Aspect ratio**: 16:9 only when using references
- **Video duration**: 8 seconds only when using references

## 🚀 Official API

Model: `veo-3.1-generate-preview` or `veo-3.1-fast-generate-preview`

Parameters:
- `prompt`: text description (up to 1024 tokens)
- `referenceImages`: array of 1-3 images
- `aspectRatio`: "16:9" or "9:16"
- `resolution`: "720p" or "1080p"
- `durationSeconds`: 4, 6 or 8 (8 only for references)

## ⏱️ Processing Time

- Minimum: 11 seconds
- Maximum: 6 minutes (during peak hours)
- Storage: 2 days on server

## 🔒 Limitations

- All videos are marked with SynthID
- Safety filters (may block content)
- Regional restrictions (EU, UK)
- Audio may be blocked (<5% of cases)

## 📝 Prompt Recommendations

Prompt structure:
1. **Action**: how objects are animated
2. **Style**: desired animation style
3. **Camera movement** (optional)
4. **Atmosphere** (optional)

Example:
```
The video opens with a medium, eye-level shot of a beautiful woman
with dark hair and warm brown eyes. She wears a magnificent, high-fashion
flamingo dress with layers of pink and fuchsia feathers. She walks with
serene confidence through crystal-clear, shallow turquoise water...
```

## 🛠️ Current Implementation Status

Our system uses:
- ✅ Gemini API for video/image analysis
- ✅ OpenAI GPT-4 for prompt generation
- ⚠️ Veo 3.1 API (requires preview access)

If Veo 3.1 is unavailable, the system:
1. Analyzes references
2. Generates optimized prompt
3. Returns prompt for manual use in Google AI Studio
