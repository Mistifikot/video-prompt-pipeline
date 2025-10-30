"""
Agent 1: Visual semantic analyzer
Analyzes images/videos and extracts detailed scene description
"""

import os
import json
import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Any

import mimetypes

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image


class VisualAnalyzer:
    """Visual content analyzer for extracting structured scene descriptions"""

    def __init__(self, openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.preferred_model = "gpt-4o"  # GPT-4o supports vision and video
        self.logger = logging.getLogger(self.__class__.__name__)
        self._http = self._build_retry_session()

    def _build_retry_session(self) -> requests.Session:
        """Creates requests.Session with retries for unstable networks"""
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=1.5,
            status_forcelist=(408, 409, 425, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _post_with_retries(
        self,
        url: str,
        *,
        max_attempts: int = 4,
        backoff_factor: float = 2.0,
        raise_for_status: bool = True,
        **kwargs,
    ) -> requests.Response:
        """Makes POST request with additional exponential backoff between attempts"""
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._http.post(url, **kwargs)
                if raise_for_status:
                    response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                last_error = exc
                self.logger.warning(
                    "POST %s failed on attempt %s/%s: %s", url, attempt, max_attempts, exc
                )
                if attempt >= max_attempts:
                    raise
                sleep_seconds = backoff_factor ** (attempt - 1)
                time.sleep(sleep_seconds)
        # If we reached here, raise the last error
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected empty retry loop state")

    def _get_analysis_prompt(self) -> str:
        """Returns universal prompt for media analysis"""
        return """
ROLE

You are a senior Director of Photography (DP), camera operator, and color supervisor. Analyze the provided media input (single image, sequence of frames, or video) exactly as a professional film crew would. Do not invent details—mark unknowns as null and report confidence.

OBJECTIVE

1) Extract the most complete description of scene, optics, camera motion, lighting, composition, backgrounds, textures, and post effects.

2) If input is a video, reconstruct the camera path over time (keyframes, durations, motion types, stabilization, micro-jitters).

3) Produce a precise transfer-prompt and parameter set so a video generator can replicate the camera move and lighting on a NEW subject.

OUTPUT FORMAT

Return two blocks: (A) Human-readable report and (B) Machine-readable JSON (strictly following the schema below), then (C) a Transfer-Prompt for the generator.

- Camera motion terminology: pan, tilt, dolly, truck, pedestal, crane(jib), arc/orbit, roll, zoom, handheld, steadicam, gimbal, drone.
- Lighting roles: key, fill, rim/back, practical, ambient, bounce.
- Color temperature in Kelvin.
- If possible, estimate fps, exposure and shutter-angle from motion blur.
- Use units: meters, millimeters, degrees, seconds, Kelvin.
- Keep wording factual and concise. No flowery language.

WHAT TO EXTRACT

1) Media metadata
- media_type: image | video | frames
- duration_s (if video), fps (estimated or exact), resolution_px (W×H), aspect_ratio
- playback_speed_anomalies: speed-ramp / slow-motion / time-lapse / none

2) Subjects & materials
- subjects: object classes, shape, scale relative to frame
- materials/textures: metal/wood/glass/fabric/skin/other; glossy/satin/matte; micro-relief/facets
- dominant_colors (hex), reflectivity/translucency

3) Composition & framing
- shot_size: extreme close-up / close-up / medium / wide / extreme wide
- techniques: rule of thirds, symmetry, leading lines, negative space, perspective/parallax
- depth_of_field: shallow/medium/deep; bokeh character (disc shape, cat's-eye, onion-ring)

4) Optics & focus
- estimated focal_length_mm and FOV (fov_deg)
- aperture (T or f), focus_distance_m, focus pulls (with timecodes)
- distortion (barrel/pincushion/mustache), vignetting, chromatic aberration

5) Camera motion (video/frames)
- motion types: pan/tilt/dolly/truck/pedestal/crane/arc/roll/zoom/handheld/steadicam/gimbal/drone
- trajectory character: linear/arc/orbit/spiral; direction vs subject
- speed & acceleration (m/s or deg/s), easing curves (linear/easeIn/easeOut/easeInOut/bezier)
- micro-jitter: amplitude (px or deg), frequency (Hz), stabilization (optical/digital/none), crop%
- global 2D motion model (translation/rotation/scale) if derivable

6) Lighting & environment
- lighting rig: list sources (key/fill/rim/practical/ambient/bounce)
  For each source: position (x,y,z relative to subject/camera), distance (m), size (m),
  modifier (softbox/umbrella/grid/flag/diffusion/gobo/none), softness, relative intensity,
  color_temp_K, tint (G/M), incidence angle (deg), notable reflections/speculars
- environment: interior/exterior, time of day, weather, dominant reflectors/windows/neon, reflections/caustics

7) Color, tone & post
- grading/look: contrast (low/mid/high), saturation (low/mid/high),
  hue shifts in shadows/mids/highlights, overall gamma curve
- white_balance_K, LUT/ACES-like workflow (if evident), grain/noise, halation/bloom,
  sharpening, compression artifacts, other VFX/SFX

8) Editing & rhythm (if cuts exist)
- transitions: cut/dissolve/whip/match/speed-ramp/freeze/overlay
- cut durations, sync to beats/accents, viewer guidance intent

9) Audio (if present)
- rhythm/beats/SFX, sync to keyframes or motion

10) Ambiguities & confidence
- list all assumptions and per-section confidence scores in [0..1]

=== OUTPUT (A): Human-Readable Report ===
- One-paragraph scene synopsis.
- Bullet details for sections 1–9 with timecodes where applicable. Short, precise sentences.

=== OUTPUT (B): Machine-Readable JSON (STRICT SCHEMA) ===
{
  "media": {
    "type": "image|video|frames",
    "duration_s": null | number,
    "fps": null | number,
    "resolution_px": {"w": number, "h": number},
    "aspect_ratio": "string",
    "playback_speed_anomalies": ["none" | "speed_ramp" | "slow_motion" | "time_lapse"]
  },
  "subjects": [
    {
      "name": "string",
      "class": "string",
      "approx_scale_in_frame": "XS|S|M|L|XL",
      "materials": [{"type":"metal|glass|gem|wood|fabric|skin|other","finish":"glossy|satin|matte","notes":"string"}],
      "dominant_colors_hex": ["#RRGGBB", "..."]
    }
  ],
  "composition": {
    "shot_size": "ECU|CU|MCU|MS|MLS|WS|EWS",
    "framing": {"rule_of_thirds": true, "symmetry": false, "leading_lines": true, "negative_space":"low|mid|high"},
    "depth_of_field": "shallow|medium|deep",
    "bokeh_notes": "string"
  },
  "optics": {
    "focal_length_mm": null | number,
    "fov_deg": null | number,
    "aperture_T": null | number,
    "focus_distance_m": null | number,
    "focus_pulls": [{"t_s": number, "to_m": number}],
    "distortion": "none|barrel|pincushion|mustache",
    "vignetting": "none|low|mid|high",
    "chromatic_aberration": "none|low|mid|high"
  },
  "camera_motion": {
    "support": "tripod|handheld|steadicam|gimbal|dolly|crane|drone|mixed",
    "global_model_2d": {"translation_px_per_s":[number, number], "rotation_deg_per_s": number, "scale_change_per_s": number},
    "stabilization": {"type":"none|optical|digital","crop_pct": null | number},
    "micro_jitter": {"amplitude_px": null | number, "frequency_hz": null | number},
    "keyframes": [
      {
        "t_s": number,
        "type": ["pan","tilt","dolly","truck","pedestal","crane","arc","roll","zoom"],
        "position_m": {"x": null | number, "y": null | number, "z": null | number},
        "rotation_euler_deg": {"yaw": number, "pitch": number, "roll": number},
        "distance_to_subject_m": null | number,
        "focal_length_mm": null | number,
        "fov_deg": null | number,
        "focus_distance_m": null | number,
        "ease": "linear|easeIn|easeOut|easeInOut|bezier",
        "duration_s": null | number,
        "notes": "string"
      }
    ]
  },
  "lighting": {
    "environment": {"type":"interior|exterior","time_of_day":"string","weather":"string","reflections":"string"},
    "lights": [
      {
        "role":"key|fill|rim|back|practical|ambient|bounce",
        "position_m": {"x": null | number, "y": null | number, "z": null | number},
        "distance_m": null | number,
        "size_m": null | number,
        "modifier":"none|softbox|umbrella|grid|flag|diffusion|gobo",
        "softness":"hard|semi|soft",
        "intensity_rel": "low|mid|high",
        "color_temp_K": null | number,
        "tint":"string|null",
        "angle_deg": null | number,
        "notes":"string"
      }
    ]
  },
  "color_and_post": {
    "white_balance_K": null | number,
    "look": {"contrast":"low|mid|high","saturation":"low|mid|high","shadows_hue":"string","mids_hue":"string","highlights_hue":"string"},
    "grain_noise":"none|low|mid|high",
    "halation_bloom":"none|low|mid|high",
    "sharpness":"soft|neutral|crisp",
    "compression_artifacts":"none|low|mid|high",
    "vfx_sfx":"string"
  },
  "editing": {
    "cuts":[{"at_s": number, "type":"cut|dissolve|whip|match|freeze|speed_ramp"}],
    "music_sync":{"beats_s":[number], "notes":"string"}
  },
  "uncertainty": [{"section":"string","assumption":"string","confidence": number}],
  "confidence_overall": number
}

=== OUTPUT (C): Transfer-Prompt for the Generator ===
1) Write a concise scene description for a NEW subject, keeping:
   — the same camera motion (per keyframes),
   — the same lighting rig (roles, approximate positions, softness, color temps),
   — the same frame parameters (fps, duration, aspect, focal/FOV, DOF),
   — the same edit tempo/beat sync (if present).

2) Provide a machine block:
{
  "apply_to_subject": "{{NEW_SUBJECT}}",
  "duration_s": <from media.duration_s>,
  "fps": <from media.fps>,
  "camera_path": <camera_motion.keyframes>,
  "camera_support": <camera_motion.support>,
  "stabilization": <camera_motion.stabilization>,
  "lens_mm": <optics.focal_length_mm>,
  "aperture_T": <optics.aperture_T>,
  "focus_distance_m": <optics.focus_distance_m>,
  "lighting_rig": <lighting.lights>,
  "color_look": <color_and_post.look>,
  "white_balance_K": <color_and_post.white_balance_K>,
  "notes": "replicate motion/lighting exactly; scale distances if subject size differs"
}

3) If critical parameters are unknown, set null and briefly explain assumptions.

QUALITY CHECKS (before returning):
- Every keyframe has t_s and at least one of: rotation_euler_deg, position_m, distance_to_subject_m, focal_length_mm/FOV.
- Handheld implies non-zero micro_jitter.
- At least one light marked as key (or explain dominant source).
- Color temperature provided (K) or null with reason.
- Every major section includes a confidence score [0..1].

TONE
Factual, concise, reproducible. No stylistic flourishes. Use standard terminology.
"""

    def _download_media(self, url: str, max_retries: int = 3) -> Tuple[bytes, str]:
        """
        Downloads media from URL and returns data + MIME type
        Uses browser headers and retry logic to bypass blocks
        """
        from urllib.parse import urlparse

        # Build correct Referer from URL domain
        parsed_url = urlparse(url)
        referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"

        # Different User-Agent for attempts
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

        # Retry logic with different approaches (WITHOUT Sec-Fetch headers!)
        last_error = None
        for attempt in range(max_retries):
            try:
                # Add delay between retries (exponential backoff)
                if attempt > 0:
                    delay = min(2 ** attempt, 10)  # Max 10 seconds
                    time.sleep(delay)

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

                response = self._http.get(url, headers=headers, timeout=30, allow_redirects=True)
                response.raise_for_status()
                content_type = response.headers.get('content-type', '')
                return response.content, content_type
            except requests.exceptions.Timeout as e:
                last_error = e
                error_msg = f"Connection timeout (attempt {attempt + 1}/{max_retries}): {str(e)}"
                if attempt < max_retries - 1:
                    print(f"[WARNING] {error_msg}, retrying...")
                    continue
                raise requests.exceptions.RequestException(f"Failed to download image after {max_retries} attempts: {error_msg}")
            except requests.exceptions.ConnectionError as e:
                last_error = e
                error_msg = f"Connection error (attempt {attempt + 1}/{max_retries}): {str(e)}"
                if attempt < max_retries - 1:
                    print(f"[WARNING] {error_msg}, retrying...")
                    continue
                raise requests.exceptions.RequestException(f"Failed to download image after {max_retries} attempts: {error_msg}")
            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response.status_code == 403 and attempt < max_retries - 1:
                    # For 403 try next approach
                    continue
                # For other HTTP errors immediately raise
                if e.response.status_code != 403:
                    raise
            except requests.exceptions.RequestException as e:
                last_error = e
                error_msg = f"Request error (attempt {attempt + 1}/{max_retries}): {str(e)}"
                if attempt < max_retries - 1:
                    print(f"[WARNING] {error_msg}, retrying...")
                    continue
                raise requests.exceptions.RequestException(f"Failed to download image after {max_retries} attempts: {error_msg}")

        # If all attempts failed, raise the last error
        if last_error:
            raise requests.exceptions.RequestException(f"Failed to download image after {max_retries} attempts: {str(last_error)}")

    def _is_video(self, content_type: str, file_path: Optional[str] = None) -> bool:
        """Determines if media is video"""
        video_mimes = ['video/', 'application/x-mpegURL']
        video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v']
        if content_type:
            if any(mime in content_type for mime in video_mimes):
                return True
        if file_path:
            path_str = str(file_path).lower()
            if any(path_str.endswith(ext) for ext in video_extensions):
                return True
            mime, _ = mimetypes.guess_type(file_path)
            if mime and 'video' in mime:
                return True
        return False

    def _resize_image(self, image: Image.Image, max_dim: int = 1280) -> Image.Image:
        """Scales image so that the longest side does not exceed max_dim."""
        if max(image.size) <= max_dim:
            return image
        resized = image.copy()
        resized.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return resized

    def _encode_image(self, image: Image.Image, quality: int = 85) -> Tuple[bytes, str]:
        """Converts image to JPEG and returns bytes along with MIME type."""
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue(), "image/jpeg"

    def _prepare_image_bytes(self, image_bytes: bytes, max_dim: int = 1280) -> Tuple[bytes, str]:
        """Prepares image for sending to model: resize and re-encoding."""
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                processed = self._resize_image(img, max_dim=max_dim)
                return self._encode_image(processed)
        except Exception as exc:
            self.logger.debug("Image preprocessing failed, using original bytes: %s", exc)
            return image_bytes, "image/jpeg"

    def _extract_video_frames(
        self,
        video_data: bytes,
        max_frames: int = 8,
    ) -> Tuple[List[bytes], Dict[str, Any]]:
        """Extracts key frames and basic metadata from video"""
        try:
            # Save temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_file.write(video_data)
                tmp_path = tmp_file.name

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return [], {}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = float(total_frames / fps) if fps and fps > 0 else None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames_to_extract = max_frames if total_frames <= 0 else min(max_frames, total_frames)
            frame_interval = max(1, (total_frames // frames_to_extract) if total_frames else 1)

            sampled_indices: List[int] = []
            extracted_frames: List[bytes] = []
            resized_gray_frames: List[np.ndarray] = []
            brightness_values: List[float] = []

            scale_factor = 1.0
            if width > 0:
                scale_factor = min(1.0, 320.0 / float(width))

            current_index = 0
            while current_index < total_frames and len(extracted_frames) < frames_to_extract:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
                ret, frame = cap.read()
                if not ret:
                    current_index += frame_interval
                    continue

                sampled_indices.append(current_index)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                processed = self._resize_image(img)
                frame_bytes, _ = self._encode_image(processed)
                extracted_frames.append(frame_bytes)

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if scale_factor < 1.0:
                    gray_frame = cv2.resize(
                        gray_frame,
                        (0, 0),
                        fx=scale_factor,
                        fy=scale_factor,
                        interpolation=cv2.INTER_AREA,
                    )
                resized_gray_frames.append(gray_frame)
                brightness_values.append(float(np.mean(gray_frame)))

                current_index += frame_interval

            cap.release()

            # Delete temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

            metadata: Dict[str, Any] = {
                "frame_count": total_frames if total_frames > 0 else None,
                "fps": float(fps) if fps and fps > 0 else None,
                "duration_s": round(duration, 3) if duration else None,
                "resolution_px": {"w": width, "h": height} if width and height else None,
                "sampled_frames": len(extracted_frames),
                "sample_indices": sampled_indices,
            }

            if brightness_values:
                metadata["luma_stats"] = {
                    "avg": round(float(np.mean(brightness_values)), 3),
                    "min": round(float(np.min(brightness_values)), 3),
                    "max": round(float(np.max(brightness_values)), 3),
                    "std": round(float(np.std(brightness_values)), 3),
                }

            motion_metadata = self._estimate_video_motion(resized_gray_frames, metadata.get("fps"))
            if motion_metadata:
                metadata["auto_motion"] = motion_metadata

            return extracted_frames, metadata

        except Exception as e:
            self.logger.error(f"Frame extraction error: {e}")
            return [], {}

    def _estimate_video_motion(
        self,
        gray_frames: List[np.ndarray],
        fps: Optional[float],
    ) -> Dict[str, Any]:
        """Attempts to estimate camera and object motion based on optical flow"""
        if len(gray_frames) < 2:
            return {}

        flow_magnitudes: List[float] = []
        translation_magnitudes: List[float] = []
        rotation_scores: List[float] = []
        direction_vectors: List[Tuple[float, float]] = []

        for prev_gray, next_gray in zip(gray_frames[:-1], gray_frames[1:]):
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    next_gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=25,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
            except cv2.error:
                continue

            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            mean_x = float(np.mean(flow_x))
            mean_y = float(np.mean(flow_y))
            translation_vector = (mean_x, mean_y)
            direction_vectors.append(translation_vector)
            translation_magnitudes.append(float(np.hypot(mean_x, mean_y)))
            flow_magnitudes.append(float(np.mean(np.hypot(flow_x, flow_y))))

            h, w = prev_gray.shape
            ys, xs = np.mgrid[0:h, 0:w]
            rel_x = (xs - (w / 2.0)) / max(w, 1)
            rel_y = (ys - (h / 2.0)) / max(h, 1)
            rotation = float(np.mean((rel_x * flow_y) - (rel_y * flow_x)))
            rotation_scores.append(rotation)

        if not flow_magnitudes:
            return {}

        avg_flow = float(np.mean(flow_magnitudes))
        median_translation = float(np.median(translation_magnitudes)) if translation_magnitudes else 0.0
        median_rotation = float(np.median(rotation_scores)) if rotation_scores else 0.0
        rotation_strength = float(np.median(np.abs(rotation_scores))) if rotation_scores else 0.0

        translation_threshold = 0.08
        translation_high_threshold = 0.18
        rotation_threshold = 0.015

        if median_translation < translation_threshold and rotation_strength > rotation_threshold:
            dominant_motion = "static_camera_subject_rotation"
        elif median_translation >= translation_threshold:
            dominant_motion = "camera_motion"
        else:
            dominant_motion = "mostly_static"

        camera_support_guess = "tripod"
        if dominant_motion == "camera_motion":
            camera_support_guess = "stabilized"
            if median_translation > translation_high_threshold:
                camera_support_guess = "handheld"

        subject_motion = "none"
        if dominant_motion == "static_camera_subject_rotation":
            subject_motion = "axial_rotation"
        elif rotation_strength > rotation_threshold * 1.5:
            subject_motion = "combined"

        avg_direction_x = float(np.mean([vec[0] for vec in direction_vectors])) if direction_vectors else 0.0
        avg_direction_y = float(np.mean([vec[1] for vec in direction_vectors])) if direction_vectors else 0.0
        avg_direction_deg = None
        if avg_direction_x or avg_direction_y:
            avg_direction_deg = round(float(np.degrees(np.arctan2(avg_direction_y, avg_direction_x))), 2)

        motion_summary: Dict[str, Any] = {
            "dominant_motion": dominant_motion,
            "average_flow_px": round(avg_flow, 4),
            "median_translation_px": round(median_translation, 4),
            "median_rotation_score": round(median_rotation, 5),
            "rotation_strength": round(rotation_strength, 5),
            "camera_support_guess": camera_support_guess,
            "subject_motion": subject_motion,
            "sample_pairs": len(flow_magnitudes),
            "confidence": round(min(1.0, len(flow_magnitudes) / 4.0), 3),
        }

        if avg_direction_deg is not None:
            motion_summary["camera_translation_direction_deg"] = avg_direction_deg

        if fps:
            motion_summary["pair_interval_s"] = round(float(1.0 / fps), 4)

        if rotation_strength > rotation_threshold:
            motion_summary["rotation_direction"] = "counter_clockwise" if median_rotation > 0 else "clockwise"

        return motion_summary

    def _format_video_insights(self, metadata: Dict[str, Any]) -> str:
        """Formats text block with objective video metrics for model hinting"""
        if not metadata:
            return ""

        lines: List[str] = []
        fps = metadata.get("fps")
        duration = metadata.get("duration_s")
        resolution = metadata.get("resolution_px")

        if fps:
            lines.append(f"- Estimated FPS: {fps}")
        if duration:
            lines.append(f"- Duration: {duration} sec")
        if resolution:
            lines.append(
                f"- Native resolution: {resolution.get('w')}x{resolution.get('h')}"
            )

        luma_stats = metadata.get("luma_stats")
        if isinstance(luma_stats, dict) and luma_stats:
            lines.append(
                "- Average scene luminance (0-255): "
                f"avg={luma_stats.get('avg')}, min={luma_stats.get('min')}, max={luma_stats.get('max')}"
            )

        auto_motion = metadata.get("auto_motion")
        if isinstance(auto_motion, dict) and auto_motion:
            dominant = auto_motion.get("dominant_motion")
            if dominant:
                lines.append(f"- Dominant motion classification: {dominant}")
            subject_motion = auto_motion.get("subject_motion")
            if subject_motion and subject_motion != "none":
                lines.append(f"- Subject motion heuristic: {subject_motion}")
            support = auto_motion.get("camera_support_guess")
            if support:
                lines.append(f"- Likely camera support: {support}")
            rotation_dir = auto_motion.get("rotation_direction")
            if rotation_dir:
                lines.append(f"- Rotation direction: {rotation_dir}")
            translation_mag = auto_motion.get("median_translation_px")
            if translation_mag is not None:
                lines.append(f"- Median camera translation flow: {translation_mag} px")
            avg_flow = auto_motion.get("average_flow_px")
            if avg_flow is not None:
                lines.append(f"- Average optical-flow magnitude: {avg_flow} px")

        if not lines:
            return ""

        lines.append(
            "- Integrate these metrics into camera_motion and lighting notes. If the camera is static but the subject rotates, spell it out."
        )

        return "VIDEO STRUCTURAL INSIGHTS (objective measurements):\n" + "\n".join(lines)

    def _encode_media_to_base64(self, media_data: bytes) -> str:
        """Encodes media to base64"""
        return base64.b64encode(media_data).decode('utf-8')

    @staticmethod
    def _extract_json_from_text(analysis_text: str) -> Dict:
        """Attempts to extract JSON from model response."""
        if not analysis_text:
            return {"raw_analysis": "", "format": "empty_response"}

        import re

        # Look for ```json ... ``` blocks
        code_block_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
        for match in code_block_pattern.findall(analysis_text):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        json_start = analysis_text.find('{')
        json_end = analysis_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            candidate = analysis_text[json_start:json_end]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        return {
            "raw_analysis": analysis_text,
            "format": "text_response"
        }

    def _analyze_with_openai(self, media_data: bytes, content_type: str, is_video: bool) -> Dict:
        """Analysis via OpenAI GPT-4o (supports images and video)"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found")

        analysis_prompt = self._get_analysis_prompt()
        video_metadata: Dict[str, Any] = {}

        # For video extract several frames
        if is_video:
            frames, video_metadata = self._extract_video_frames(media_data, max_frames=8)
            if not frames:
                raise ValueError("Failed to extract frames from video")

            insights_block = self._format_video_insights(video_metadata)
            prompt_text = analysis_prompt
            if insights_block:
                prompt_text = (
                    f"{analysis_prompt}\n\n{insights_block}\n"
                    "Ensure the JSON camera_motion keyframes and transfer prompt strictly follow these measured cues."
                )

            # Analyze several frames to understand dynamics
            content_parts = [{
                "type": "text",
                "text": prompt_text
            }]

            for frame in frames[:4]:  # Use first 4 frames for analysis
                base64_frame = self._encode_media_to_base64(frame)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_frame}"
                    }
                })
        else:
            processed_bytes, detected_type = self._prepare_image_bytes(media_data)
            base64_image = self._encode_media_to_base64(processed_bytes)
            if detected_type:
                content_type = detected_type

            # Determine format for OpenAI
            if 'png' in content_type or 'image/png' in content_type:
                media_format = "png"
            elif 'jpeg' in content_type or 'jpg' in content_type:
                media_format = "jpeg"
            elif 'gif' in content_type:
                media_format = "gif"
            elif 'webp' in content_type:
                media_format = "webp"
            else:
                media_format = (detected_type.split('/')[-1] if detected_type else "jpeg")

            content_parts = [{
                "type": "text",
                "text": analysis_prompt
            }, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{media_format};base64,{base64_image}"
                }
            }]

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.preferred_model,
            "messages": [
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            "max_tokens": 4000,  # Increased for detailed analysis with camera, lighting, etc.
            "temperature": 0.3
        }

        response = self._post_with_retries(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,  # Increased timeout for detailed analysis
        )
        result = response.json()
        analysis_text = result['choices'][0]['message']['content']

        analysis_json = self._extract_json_from_text(analysis_text)
        if "raw_analysis" not in analysis_json:
            analysis_json["raw_analysis"] = analysis_text

        if video_metadata:
            analysis_json.setdefault("metadata", {})
            analysis_json["metadata"]["auto_video_insights"] = video_metadata

            # Additionally pass motion heuristics if model didn't fill camera_motion block
            auto_motion = video_metadata.get("auto_motion")
            if auto_motion:
                if not analysis_json.get("camera_motion"):
                    analysis_json["camera_motion"] = {
                        "support": auto_motion.get("camera_support_guess"),
                        "dominant_motion": auto_motion.get("dominant_motion"),
                        "notes": "Auto-computed fallback; please validate",
                    }
                # CRITICAL: Add object motion if present
                subject_motion = auto_motion.get("subject_motion")
                if subject_motion and subject_motion != "none":
                    # Add to camera_motion for prompt generator access
                    if isinstance(analysis_json.get("camera_motion"), dict):
                        analysis_json["camera_motion"]["subject_motion"] = subject_motion
                        analysis_json["camera_motion"]["subject_motion_notes"] = (
                            "Object movement detected: " + subject_motion +
                            ". Ensure prompt includes micro-movements or subject rotation."
                        )
                    # Also add to top level for easy access
                    analysis_json["subject_motion_detected"] = subject_motion

            media_block = analysis_json.setdefault("media", {})
            if video_metadata.get("fps") and not media_block.get("fps"):
                media_block["fps"] = video_metadata["fps"]
            if video_metadata.get("duration_s") and not media_block.get("duration_s"):
                media_block["duration_s"] = video_metadata["duration_s"]
            if video_metadata.get("resolution_px") and not media_block.get("resolution_px"):
                media_block["resolution_px"] = video_metadata["resolution_px"]
            if not media_block.get("type"):
                media_block["type"] = "video"

        return analysis_json

    def _analyze_with_gemini(self, media_data: bytes, content_type: str, is_video: bool) -> Dict:
        """Analysis via Google Gemini (works better with video)"""
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found")

        # Determine MIME type
        if is_video:
            if 'mp4' in content_type.lower():
                mime_type = "video/mp4"
            elif 'webm' in content_type.lower():
                mime_type = "video/webm"
            elif 'mov' in content_type.lower():
                mime_type = "video/quicktime"
            else:
                mime_type = "video/mp4"

            # For video use Gemini File API (supports video directly)
            return self._analyze_video_with_gemini_file_api(media_data, mime_type)
        else:
            # For images use inline base64
            processed_bytes, detected_type = self._prepare_image_bytes(media_data)
            base64_media = self._encode_media_to_base64(processed_bytes)
            mime_type = detected_type or content_type or "image/jpeg"

            # Use the same universal prompt as for OpenAI
            analysis_prompt = self._get_analysis_prompt()

            headers = {
                "Content-Type": "application/json"
            }

            payload = {
                "contents": [{
                    "parts": [
                        {
                            "text": analysis_prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_media
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 4000  # Increased for detailed analysis
                }
            }

            response = self._post_with_retries(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.gemini_api_key}",
                headers=headers,
                json=payload,
                timeout=60,
            )
            result = response.json()
            analysis_text = result['candidates'][0]['content']['parts'][0]['text']

            analysis_json = self._extract_json_from_text(analysis_text)
            if "raw_analysis" not in analysis_json:
                analysis_json["raw_analysis"] = analysis_text

            return analysis_json

    def _analyze_video_with_gemini_file_api(self, video_data: bytes, mime_type: str) -> Dict:
        """Uploads video via Gemini File API and analyzes"""
        # Step 1: Upload file via File API
        upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.gemini_api_key}"

        headers = {
            'X-Goog-Upload-Protocol': 'multipart'
        }

        files = {
            'metadata': ('metadata', json.dumps({
                "file": {
                    "display_name": "video_analysis"
                }
            }), 'application/json; charset=UTF-8'),
            'file': ('video_analysis.mp4', video_data, mime_type)
        }

        response = self._post_with_retries(
            upload_url,
            headers=headers,
            files=files,
            timeout=120,
            raise_for_status=False,
        )
        if response.status_code >= 400:
            raise ValueError(f"Gemini File API error {response.status_code}: {response.text}")
        file_metadata = response.json()
        file_uri = file_metadata.get('file', {}).get('uri')

        if not file_uri:
            raise ValueError("Failed to upload video to Gemini File API")

        # Step 2: Wait for file processing (optional, can skip for fast requests)
        time.sleep(2)  # Small delay for file processing

        # Step 3: Analyze video via generateContent API
        analysis_prompt = self._get_analysis_prompt()

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": analysis_prompt
                    },
                    {
                        "file_data": {
                            "mime_type": mime_type,
                            "file_uri": file_uri
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2000
            }
        }

        response = self._post_with_retries(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.gemini_api_key}",
            headers=headers,
            json=payload,
            timeout=120,
        )
        result = response.json()
        analysis_text = result['candidates'][0]['content']['parts'][0]['text']

        # Delete file after analysis (optional)
        try:
            delete_url = f"https://generativelanguage.googleapis.com/v1beta/{file_uri}?key={self.gemini_api_key}"
            self._http.delete(delete_url, timeout=10)
        except:
            pass  # Ignore deletion errors

        analysis_json = self._extract_json_from_text(analysis_text)
        if "raw_analysis" not in analysis_json:
            analysis_json["raw_analysis"] = analysis_text

        return analysis_json

    def analyze(
        self,
        media_source: Union[str, Path, bytes],
        content_type: Optional[str] = None,
        use_gemini: bool = False
    ) -> Dict:
        """
        Analyzes media and returns structured description

        Args:
            media_source: URL, file path or bytes media
            content_type: MIME type (optional)
            use_gemini: Use Gemini instead of OpenAI

        Returns:
            Dict with structured scene description
        """
        # Determine source type and load data
        if isinstance(media_source, bytes):
            media_data = media_source
            is_video = self._is_video(content_type or "")
        elif isinstance(media_source, (str, Path)):
            source_str = str(media_source)
            # Check if it's URL or file path
            if source_str.startswith(('http://', 'https://')):
                media_data, detected_content_type = self._download_media(source_str)
                content_type = content_type or detected_content_type
            else:
                # Local file
                file_path = Path(source_str)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                media_data = file_path.read_bytes()
                content_type = content_type or mimetypes.guess_type(str(file_path))[0]

            is_video = self._is_video(content_type or "", str(media_source))
        else:
            raise ValueError(f"Unsupported source type: {type(media_source)}")

        # Choose model for analysis
        analysis_result = None
        model_used = None
        if use_gemini and self.gemini_api_key:
            try:
                analysis_result = self._analyze_with_gemini(media_data, content_type or "", is_video)
                model_used = "gemini-1.5-pro"
            except Exception as gemini_error:
                print(f"[!] Gemini analysis failed: {gemini_error}")
                if self.openai_api_key:
                    print("[i] Switching to OpenAI for analysis")
                    analysis_result = self._analyze_with_openai(media_data, content_type or "", is_video)
                    model_used = self.preferred_model
                else:
                    raise
        if analysis_result is None:
            if self.openai_api_key:
                analysis_result = self._analyze_with_openai(media_data, content_type or "", is_video)
                model_used = self.preferred_model
            else:
                raise ValueError("No API key found for analysis (need OpenAI or Gemini)")

        # Add metadata
        analysis_result['metadata'] = {
            'is_video': is_video,
            'content_type': content_type,
            'model_used': model_used or ("gemini-1.5-pro" if (use_gemini and self.gemini_api_key) else self.preferred_model)
        }

        return analysis_result


if __name__ == "__main__":
    # Test example
    analyzer = VisualAnalyzer()

    # Usage example
    # result = analyzer.analyze("https://example.com/jewelry.jpg")
    # print(json.dumps(result, indent=2, ensure_ascii=False))

