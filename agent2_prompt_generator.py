"""
Agent 2: Visual prompt engineer
Converts structured scene description into optimized prompt for Sora 2 / Veo 3
"""

import os
import json
import time
from typing import Dict, Optional
import requests


class PromptGenerator:
    """Prompt generator for AI video platforms"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        perplexity_api_key: Optional[str] = None,
        auto_polish: Optional[bool] = None
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o"  # Or gpt-4-turbo
        self.perplexity_api_key = perplexity_api_key or os.getenv("PPLX_API_KEY")
        self.perplexity_model = os.getenv("PPLX_MODEL", "sonic")
        if auto_polish is None:
            env_flag = os.getenv("PPLX_AUTO_POLISH")
            if env_flag is not None:
                auto_polish = env_flag.strip().lower() in {"1", "true", "yes", "on"}
            else:
                auto_polish = bool(self.perplexity_api_key)
        self.auto_polish = auto_polish and bool(self.perplexity_api_key)

        # Templates and guidelines for each platform
        self.platform_guides = {
            "veo3": {
                "name": "Google Veo 3.1",
                "guide": """
CRITICAL: Prompt MUST be formatted strictly for Veo 3.1 API. Follow these rules EXACTLY:

1. LANGUAGE: English only. No other languages.

2. STRUCTURE: Start with subject and action, then camera movement, then lighting, then technical details.

3. CAMERA MOVEMENT (use exact terminology):
   - pan (horizontal rotation), tilt (vertical rotation)
   - dolly (forward/backward movement), truck (left/right movement)
   - pedestal (up/down movement), crane/jib (vertical boom movement)
   - arc/orbit (curved movement around subject)
   - roll (rotation around lens axis)
   - zoom (focal length change), push-in (dolly + zoom combination)
   - handheld (with micro-movements), steadicam (smooth floating), gimbal (stabilized)
   - Specify easing: linear, ease-in, ease-out, ease-in-out
   - Include speed: slow, medium, fast motion

4. SHOT SIZE (use standard terms):
   - Extreme close-up (ECU), Close-up (CU), Medium close-up (MCU)
   - Medium shot (MS), Medium long shot (MLS)
   - Wide shot (WS), Extreme wide shot (EWS)

5. LIGHTING (be specific):
   - Key light: position (top-left, front, side, back), softness (hard, soft, diffused)
   - Fill light: intensity relative to key
   - Rim/back light: for edge separation
   - Color temperature: warm (3000-4000K), cool (5000-6500K), daylight (5600K)
   - Quality: hard light, soft light, natural light, studio lighting, dramatic lighting

6. OPTICS & FOCUS:
   - Depth of field: shallow (f/1.4-f/2.8), medium (f/4-f/8), deep (f/11+)
   - Focus: sharp focus, soft focus, rack focus/pull focus
   - Bokeh: creamy bokeh, circular bokeh, cat's-eye bokeh
   - Focal length: wide-angle (14-35mm), standard (35-85mm), telephoto (85mm+)

7. COMPOSITION:
   - Rule of thirds, center framing, symmetry
   - Negative space, leading lines, perspective

8. MATERIALS & TEXTURES:
   - Surface finish: glossy, matte, satin, metallic, reflective
   - Texture: smooth, rough, faceted, polished
   - Reflectivity: highly reflective, semi-reflective, matte

9. STYLE & MOOD:
   - Cinematic, documentary, commercial, artistic
   - Luxury, minimalist, dramatic, romantic, professional

10. FORMAT REQUIREMENTS:
    - MINIMUM 200-400 words, preferably 400-800 words for Veo 3.1 (detailed, comprehensive)
    - Use active voice
    - Specify temporal details: duration hints, pacing
    - Technical precision over flowery language
    - No emojis, no special characters beyond standard punctuation
    - Include ALL technical details from analysis: camera motion keyframes, lighting positions, optics specs

EXAMPLE STRUCTURE:
"[Subject] [action/state]. [Camera movement type] [with speed and easing]. [Shot size]. [Lighting description with positions and color temp]. [Depth of field and focus]. [Composition]. [Materials/textures]. [Style/mood]."

BAD: "A beautiful ring spinning slowly in a elegant way"
GOOD: "Platinum engagement ring rotating clockwise on a pedestal. Slow dolly push-in with ease-in-out easing. Close-up framing. Soft key light from top-left at 45 degrees, 3200K warm temperature, minimal fill. Shallow depth of field at f/2.8, sharp focus on ring. Center composition with negative space. Highly reflective metallic surface with polished facets. Cinematic luxury aesthetic."
                """,
                "example_prefix": ""
            },
            "sora2": {
                "name": "OpenAI Sora 2",
                "guide": """
For Sora 2 prompts should:
- Focus on temporal coherence
- Emphasize motion dynamics
- Include duration hints (if needed)
- Describe style and aesthetics
- Use animation and transition descriptions
- For jewelry: emphasis on smooth motion, highlights, rotation
                """,
                "example_prefix": "Cinematic jewelry showcase: "
            },
            "seedream": {
                "name": "Seedream",
                "guide": """
For Seedream prompts should:
- Be detailed with technical terms
- Include composition description
- Specify lighting and camera characteristics
- Describe movements and effects
                """,
                "example_prefix": "Luxury jewelry visual: "
            }
        }

    def _build_system_prompt(self, platform: str, use_case: str = "general") -> str:
        """Builds system prompt for generation"""
        platform_info = self.platform_guides.get(platform, self.platform_guides["veo3"])

        use_case_instructions = {
            "product_video": "Create prompt for product video (15-30 seconds) with emphasis on product demonstration",
            "hero_image": "Create prompt for hero/key visual image with dramatic lighting",
            "gemstone_closeup": "Create prompt for close-up of gemstone with maximum detail",
            "luxury_brand": "Create prompt in luxury brand style (Tiffany, Chaumet) with elegant lighting",
            "general": "Create quality prompt based on description"
        }

        instruction = use_case_instructions.get(use_case, use_case_instructions["general"])

        # For Veo 3.1 add strict format requirements
        veo31_critical = ""
        if platform == "veo3":
            veo31_critical = """

CRITICALLY IMPORTANT for Veo 3.1:
- Prompt MUST be DETAILED and COMPREHENSIVE (minimum 300-500 words, can be more up to 800-1000 words)
- Use ALL data from analysis:
  * If subjects exist - use subjects instead of main_objects (each object with name, class, materials, dominant_colors_hex)
  * If camera_motion.keyframes exist - MUST describe camera movement from camera_motion.keyframes in detail (each keyframe with time, movement type, position, easing)
  * If camera_motion.support exists - specify camera support type (tripod, handheld, steadicam, gimbal, dolly, crane, drone)
  * If lighting.lights exist - describe EACH light source with role, position, color temperature, intensity
  * If lighting.environment exists - describe environment
  * If optics exist - use all data about focal length, aperture, focus, distortion, vignetting
  * If composition exists - use shot_size, framing, depth_of_field, bokeh_notes
  * If color_and_post exists - use white_balance_K, look, grain_noise, halation_bloom, sharpness
  * If media exists - use fps, duration_s, aspect_ratio
- Use ONLY English language
- Structure: [Subject] [action]. [Camera movement] [speed/easing]. [Shot size]. [Lighting with positions/color temp]. [Depth of field/focus]. [Composition]. [Materials/textures]. [Style]
- For video reference: MUST describe camera movement from camera_motion.keyframes in detail (each keyframe with time, movement type, position, easing)
- Technical precision is more important than artistic descriptions
- Use precise terms for camera movement (pan, tilt, dolly, truck, etc.)
- Specify color temperature in Kelvin (e.g.: 3200K warm, 5600K daylight)
- Specify lens parameters (f/2.8, f/4, f/11, focal length 85mm, etc.)
- DO NOT shorten or simplify - Veo 3.1 requires maximum detail for best results
"""

        system_prompt = f"""You are an expert in prompt engineering for AI video generators.

Your task: convert detailed visual scene description into optimized prompt for {platform_info['name']}.

{platform_info['guide']}

{instruction}
{veo31_critical}

Important rules:
- Prompt must be in English (standard for AI video)
- Use professional photography and cinematography terminology
- Be EXTREMELY specific and DETAILED - use ALL data from analysis
- Include all important visual elements from description
- For jewelry: emphasis on materials, lighting, reflections, camera movement
- Prompt must be ready to copy and use
- FOR Veo 3.1: prompt must be LONG and DETAILED (minimum 300-500 words, can be more)
- DO NOT shorten technical details - include all specifications from analysis

Response format:
Return ONLY the final prompt, without additional explanations or metadata.
Prompt must be COMPLETE and DETAILED, using ALL data from analysis."""

        return system_prompt

    @staticmethod
    def _scene_description_to_json(scene_description: Dict) -> str:
        """Safe conversion of scene description to JSON string."""

        def default_serializer(value):
            if isinstance(value, (set, tuple)):
                return list(value)
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='ignore')
            return str(value)

        return json.dumps(
            scene_description,
            indent=2,
            ensure_ascii=False,
            default=default_serializer
        )

    def _generate_with_openai(self, scene_description: Dict, platform: str, use_case: str) -> str:
        """Generates prompt via OpenAI"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found")

        system_prompt = self._build_system_prompt(platform, use_case)

        # Build user prompt
        # Check for detailed analysis schema (new version)
        has_detailed_schema = bool(
            scene_description.get("subjects") or
            scene_description.get("camera_motion") or
            scene_description.get("lighting") or
            scene_description.get("optics")
        )

        # Check for old schema (for backward compatibility)
        has_old_schema = bool(
            scene_description.get("main_objects") or
            scene_description.get("camera_work")
        )

        has_video_reference = bool(scene_description.get("video_analysis") or scene_description.get("video_reference_info") or (has_detailed_schema and scene_description.get("camera_motion", {}).get("keyframes")))
        has_image_references = bool(scene_description.get("image_analysis_summary") or scene_description.get("main_objects") or scene_description.get("subjects"))

        prompt_instructions = ""
        if has_image_references and has_video_reference:
            prompt_instructions = """
IMPORTANT: You have two sources of information:
1. IMAGES (reference images): Use objects, materials, composition, static elements from them
2. VIDEO REFERENCE: Use camera movement, presentation, scene dynamics, rhythm from it

Combine them like this: objects from images + movement and camera from video.
"""
        elif has_video_reference:
            prompt_instructions = """
IMPORTANT: Use video reference for camera movement, presentation, scene dynamics.
"""
        elif has_image_references:
            prompt_instructions = """
IMPORTANT: Use reference images for objects, materials, composition.
"""

        workflow_context = scene_description.get("workflow_context")
        if isinstance(workflow_context, dict):
            stage = workflow_context.get("stage")
            if stage == "video_camera_plus_image_subject":
                prompt_instructions += """

CRITICAL WORKFLOW INSTRUCTION:
- Camera motion MUST follow video_analysis (camera_motion, media metadata).
- Object identity, materials, textures MUST follow reference_image_analysis.subjects.
- Merge lighting logically: prefer video lighting if present, otherwise derive from image context.
"""
            workflow_notes = workflow_context.get("instructions")
            if isinstance(workflow_notes, dict) and workflow_notes:
                serialized_notes = json.dumps(workflow_notes, ensure_ascii=False, indent=2)
                prompt_instructions += f"\nAdditional merge instructions (JSON):\n{serialized_notes}\n"

        # For Veo 3.1 add special detail instructions
        detail_instructions = ""
        if platform == "veo3":
            detail_instructions = """

CRITICALLY IMPORTANT - DETAILING FOR Veo 3.1:
- Prompt MUST be VERY DETAILED (minimum 300-500 words, can be more)
- IMPORTANT: Veo requires temporal dynamics - ALWAYS add object micro-movements:
  * Check subject_motion_detected or camera_motion.subject_motion or metadata.auto_video_insights.auto_motion.subject_motion
  * If object movement exists - describe it in detail (rotation, swaying, etc.)
  * If no movement - STILL add micro-movements: "gently sway", "sparkle glints travel", "micro-reflections dance"
  * NEVER create a static frame - object must have at least minimal movement
- Use ALL data from analysis, especially:
  * subjects - describe each object (name, class, materials, colors)
  * camera_motion.keyframes - describe each keyframe with time (t_s), movement type (type), position (position_m), rotation (rotation_euler_deg), easing, duration
  * camera_motion.support - camera support type (tripod, handheld, steadicam, gimbal, dolly, crane, drone)
  * camera_motion.dominant_motion - if "static_camera_subject_rotation", must describe object movement
  * lighting.lights - describe EACH light source with role, position (position_m), distance (distance_m), size (size_m), modifier, softness, intensity (intensity_rel), color temperature (color_temp_K), angle (angle_deg)
  * lighting.environment - describe environment (type, time_of_day, weather, reflections)
  * optics - use ALL data: focal_length_mm, fov_deg, aperture_T, focus_distance_m, focus_pulls, distortion, vignetting, chromatic_aberration
  * composition - describe composition: shot_size (ECU/CU/MCU/MS/MLS/WS/EWS), framing (rule_of_thirds, symmetry, leading_lines, negative_space), depth_of_field, bokeh_notes
  * color_and_post - describe color grading: white_balance_K, look (contrast, saturation, hue shifts), grain_noise, halation_bloom, sharpness
  * media - use fps, duration_s, aspect_ratio (if 1:1, use 16:9), resolution_px
- If video reference with camera_motion.keyframes exists, MUST include detailed camera movement description OVER TIME - each keyframe must be described separately with indication of what happens at that moment
- DO NOT shorten technical details - Veo 3.1 requires maximum specificity
- Each camera movement must be described with type (pan/tilt/dolly/truck/pedestal/crane/arc/roll/zoom), direction, speed, easing (linear/easeIn/easeOut/easeInOut)
- Each light source must be described with position relative to object/camera, color temperature in Kelvin, intensity
- Use precise technical terms: e.g. "85mm focal length at f/2.8", "shallow depth of field", "3200K warm key light from top-left at 45 degrees"
"""

        user_prompt = f"""Based on this detailed scene description, create a VERY DETAILED and COMPREHENSIVE prompt for {self.platform_guides[platform]['name']}:
{prompt_instructions}
{detail_instructions}

INPUT DATA (use ALL of this):
{self._scene_description_to_json(scene_description)}

Platform: {platform}
Task type: {use_case}

Create a maximally detailed and technically accurate prompt using ALL available data from analysis."""

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "max_tokens": 2000 if platform == "veo3" else 1000,  # Veo 3.1 needs longer prompts
            "temperature": 0.6  # Slightly lower for more accurate technical details
        }

        # Retry logic for OpenAI API with exponential backoff
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Add delay between retries (exponential backoff)
                if attempt > 0:
                    delay = min(2 ** attempt, 10)  # Max 10 seconds
                    print(f"[INFO] Retrying OpenAI API request (attempt {attempt + 1}/{max_retries}) after {delay}s delay...")
                    time.sleep(delay)

                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()

                result = response.json()
                generated_prompt = result['choices'][0]['message']['content'].strip()

                return generated_prompt
            except requests.exceptions.Timeout as e:
                last_error = e
                error_msg = f"OpenAI API timeout (attempt {attempt + 1}/{max_retries}): {str(e)}"
                if attempt < max_retries - 1:
                    print(f"[WARNING] {error_msg}, retrying...")
                    continue
                raise requests.exceptions.RequestException(f"Failed to generate prompt after {max_retries} attempts: {error_msg}")
            except requests.exceptions.ConnectionError as e:
                last_error = e
                error_msg = f"OpenAI API connection error (attempt {attempt + 1}/{max_retries}): {str(e)}"
                if attempt < max_retries - 1:
                    print(f"[WARNING] {error_msg}, retrying...")
                    continue
                raise requests.exceptions.RequestException(f"Failed to generate prompt after {max_retries} attempts: {error_msg}")
            except requests.exceptions.RequestException as e:
                last_error = e
                error_msg = f"OpenAI API request error (attempt {attempt + 1}/{max_retries}): {str(e)}"
                if attempt < max_retries - 1:
                    print(f"[WARNING] {error_msg}, retrying...")
                    continue
                raise

        # If all attempts failed
        if last_error:
            raise requests.exceptions.RequestException(f"Failed to generate prompt after {max_retries} attempts: {str(last_error)}")

    def _polish_with_perplexity(
        self,
        draft_prompt: str,
        scene_description: Dict,
        platform: str,
        use_case: str
    ) -> str:
        """Performs final polish via Perplexity."""
        if not self.perplexity_api_key:
            return draft_prompt

        system_prompt = (
            "You are an elite AI video prompt engineer. Refine the provided draft prompt "
            "for maximum cinematic clarity while preserving every factual detail."
        )

        user_prompt = (
            "Draft prompt (keep technical cues, improve flow):\n"
            f"{draft_prompt}\n\n"
            "Scene JSON (do not contradict):\n"
            f"{self._scene_description_to_json(scene_description)}\n\n"
            f"Platform: {platform}\nUse case: {use_case}\n"
            "Output only the polished prompt in English."
        )

        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.perplexity_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2000 if platform == "veo3" else 1500,  # Veo 3.1 needs longer prompts
        }

        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            polished = data.get("choices", [{}])[0].get("message", {}).get("content")
            if polished:
                return polished.strip()
        except requests.RequestException as err:
            error_msg = str(err)
            if hasattr(err, 'response') and err.response is not None:
                try:
                    error_detail = err.response.text[:200]
                    print(f"[!] Perplexity polish failed: {err.response.status_code} - {error_detail}")
                except:
                    print(f"[!] Perplexity polish failed: {error_msg}")
            else:
                print(f"[!] Perplexity polish failed: {error_msg}")

        return draft_prompt

    def generate(
        self,
        scene_description: Dict,
        platform: str = "veo3",
        use_case: str = "general",
        polish_with_perplexity: Optional[bool] = None
    ) -> Dict:
        """
        Generates prompt for specified platform

        Args:
            scene_description: Structured scene description from Agent 1
            platform: veo3, sora2, seedream
            use_case: product_video, hero_image, gemstone_closeup, luxury_brand, general

        Returns:
            Dict with prompt and metadata
        """
        if platform not in self.platform_guides:
            platform = "veo3"  # Fallback

        generated_prompt = self._generate_with_openai(scene_description, platform, use_case)

        # If description has raw_analysis (unstructured text), use it
        if "raw_analysis" in scene_description and len(scene_description) == 2:
            # Optimize generation for text response
            scene_description_full = {
                "description": scene_description.get("raw_analysis", ""),
                "format": scene_description.get("format", "text")
            }
            generated_prompt = self._generate_with_openai(scene_description_full, platform, use_case)

        should_polish = self.auto_polish if polish_with_perplexity is None else bool(polish_with_perplexity)
        polished_prompt = generated_prompt
        perplexity_used = False
        if should_polish and self.perplexity_api_key:
            polished_prompt = self._polish_with_perplexity(generated_prompt, scene_description, platform, use_case)
            perplexity_used = polished_prompt != generated_prompt

        result = {
            "prompt": polished_prompt,
            "platform": platform,
            "use_case": use_case,
            "platform_name": self.platform_guides[platform]["name"],
            "metadata": {
                "model_used": self.model,
                "scene_format": "structured" if "raw_analysis" not in scene_description else "text",
                "perplexity_polish": perplexity_used
            }
        }

        if perplexity_used:
            result["metadata"]["perplexity_model"] = self.perplexity_model

        return result

    def generate_multiple(
        self,
        scene_description: Dict,
        platforms: list[str] = None,
        use_case: str = "general",
        polish_with_perplexity: Optional[bool] = None
    ) -> Dict[str, Dict]:
        """Generates prompts for multiple platforms simultaneously"""
        if platforms is None:
            platforms = ["veo3", "sora2"]

        results = {}
        for platform in platforms:
            results[platform] = self.generate(
                scene_description,
                platform,
                use_case,
                polish_with_perplexity=polish_with_perplexity
            )

        return results


if __name__ == "__main__":
    # Test example
    generator = PromptGenerator()

    # Usage example
    test_description = {
        "main_objects": ["platinum engagement ring with round brilliant diamond"],
        "materials": {
            "metal": "platinum",
            "gemstone": "diamond, round brilliant cut"
        },
        "lighting": {
            "type": "studio lighting",
            "direction": "top-down with rim light",
            "quality": "soft, diffused"
        },
        "camera_work": {
            "movement": "slow rotation around the ring",
            "framing": "close-up",
            "depth_of_field": "shallow"
        }
    }

    # result = generator.generate(test_description, platform="veo3", use_case="product_video")
    # print(json.dumps(result, indent=2, ensure_ascii=False))

