"""
Агент 2: Графический промт-инженер
Преобразует структурированное описание сцены в оптимизированный промт для Sora 2 / Veo 3
"""

import os
import json
from typing import Dict, Optional
import requests


class PromptGenerator:
    """Генератор промптов для AI-видео платформ"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        perplexity_api_key: Optional[str] = None,
        auto_polish: Optional[bool] = None
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o"  # Или gpt-4-turbo
        self.perplexity_api_key = perplexity_api_key or os.getenv("PPLX_API_KEY")
        self.perplexity_model = os.getenv("PPLX_MODEL", "sonic")
        if auto_polish is None:
            env_flag = os.getenv("PPLX_AUTO_POLISH")
            if env_flag is not None:
                auto_polish = env_flag.strip().lower() in {"1", "true", "yes", "on"}
            else:
                auto_polish = bool(self.perplexity_api_key)
        self.auto_polish = auto_polish and bool(self.perplexity_api_key)

        # Шаблоны и гайдлайны для каждой платформы
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
Для Sora 2 промты должны:
- Фокусироваться на временной связности (temporal coherence)
- Подчеркивать динамику движения
- Включать указания на длительность (если нужно)
- Описывать стиль и эстетику
- Использовать описания анимации и переходов
- Для драгоценностей: акцент на плавность движения, блики, ротация
                """,
                "example_prefix": "Cinematic jewelry showcase: "
            },
            "seedream": {
                "name": "Seedream",
                "guide": """
Для Seedream промты должны:
- Быть детализированными с техническими терминами
- Включать описание композиции
- Указывать характеристики освещения и камеры
- Описывать движения и эффекты
                """,
                "example_prefix": "Luxury jewelry visual: "
            }
        }

    def _build_system_prompt(self, platform: str, use_case: str = "general") -> str:
        """Строит системный промпт для генерации"""
        platform_info = self.platform_guides.get(platform, self.platform_guides["veo3"])

        use_case_instructions = {
            "product_video": "Создай промт для product video (15-30 секунд) с акцентом на демонстрацию продукта",
            "hero_image": "Создай промт для hero/key visual изображения с драматическим освещением",
            "gemstone_closeup": "Создай промт для крупного плана драгоценного камня с максимальной детализацией",
            "luxury_brand": "Создай промт в стиле luxury брендов (Tiffany, Chaumet) с элегантным освещением",
            "general": "Создай качественный промт на основе описания"
        }

        instruction = use_case_instructions.get(use_case, use_case_instructions["general"])

        # Для Veo 3.1 добавляем строгие требования к формату
        veo31_critical = ""
        if platform == "veo3":
            veo31_critical = """

КРИТИЧЕСКИ ВАЖНО для Veo 3.1:
- Промт ДОЛЖЕН быть ДЕТАЛЬНЫМ и ИСЧЕРПЫВАЮЩИМ (минимум 300-500 слов, можно больше до 800-1000 слов)
- Используй ВСЕ данные из анализа:
  * Если есть subjects - используй subjects вместо main_objects (каждый объект с name, class, materials, dominant_colors_hex)
  * Если есть camera_motion.keyframes - ОБЯЗАТЕЛЬНО детально опиши движение камеры из camera_motion.keyframes (каждый keyframe с временем, типом движения, позицией, easing)
  * Если есть camera_motion.support - укажи тип поддержки камеры (tripod, handheld, steadicam, gimbal, dolly, crane, drone)
  * Если есть lighting.lights - опиши КАЖДЫЙ источник света с ролью, позицией, цветовой температурой, интенсивностью
  * Если есть lighting.environment - опиши окружение
  * Если есть optics - используй все данные о фокусном расстоянии, диафрагме, фокусе, distortion, vignetting
  * Если есть composition - используй shot_size, framing, depth_of_field, bokeh_notes
  * Если есть color_and_post - используй white_balance_K, look, grain_noise, halation_bloom, sharpness
  * Если есть media - используй fps, duration_s, aspect_ratio
- Используй ТОЛЬКО английский язык
- Структура: [Subject] [action]. [Camera movement] [speed/easing]. [Shot size]. [Lighting with positions/color temp]. [Depth of field/focus]. [Composition]. [Materials/textures]. [Style]
- Для видео-референса: ОБЯЗАТЕЛЬНО детально опиши движение камеры из camera_motion.keyframes (каждый keyframe с временем, типом движения, позицией, easing)
- Техническая точность важнее художественных описаний
- Используй точные термины для движения камеры (pan, tilt, dolly, truck, etc.)
- Указывай цветовую температуру в Кельвинах (например: 3200K warm, 5600K daylight)
- Указывай параметры объектива (f/2.8, f/4, f/11, focal length 85mm, etc.)
- НЕ сокращай и НЕ упрощай - Veo 3.1 требует максимальной детализации для лучшего результата
"""

        system_prompt = f"""Ты эксперт по промт-инженерству для AI-видео генераторов.

Твоя задача: преобразовать детальное описание визуальной сцены в оптимизированный промт для {platform_info['name']}.

{platform_info['guide']}

{instruction}
{veo31_critical}

Важные правила:
- Промт должен быть на английском языке (стандарт для AI-видео)
- Используй профессиональную терминологию фотографии и кинематографии
- Будь КРАЙНЕ конкретным и ДЕТАЛИЗИРОВАННЫМ - используй ВСЕ данные из анализа
- Включи все важные визуальные элементы из описания
- Для драгоценностей: акцент на materials, lighting, reflections, camera movement
- Промт должен быть готов к копированию и использованию
- ДЛЯ Veo 3.1: промт должен быть ДОЛГИМ и ДЕТАЛЬНЫМ (минимум 300-500 слов, можно больше)
- НЕ сокращай технические детали - включай все спецификации из анализа

Формат ответа:
Верни ТОЛЬКО финальный промт, без дополнительных объяснений или метаданных.
Промт должен быть ПОЛНЫМ и ДЕТАЛЬНЫМ, используя ВСЕ данные из анализа."""

        return system_prompt

    @staticmethod
    def _scene_description_to_json(scene_description: Dict) -> str:
        """Безопасное преобразование описания сцены в JSON строку."""

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
        """Генерирует промт через OpenAI"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key не найден")

        system_prompt = self._build_system_prompt(platform, use_case)

        # Формируем пользовательский промпт
        # Проверяем наличие детальной схемы анализа (новая версия)
        has_detailed_schema = bool(
            scene_description.get("subjects") or
            scene_description.get("camera_motion") or
            scene_description.get("lighting") or
            scene_description.get("optics")
        )

        # Проверяем наличие старой схемы (для обратной совместимости)
        has_old_schema = bool(
            scene_description.get("main_objects") or
            scene_description.get("camera_work")
        )

        has_video_reference = bool(scene_description.get("video_analysis") or scene_description.get("video_reference_info") or (has_detailed_schema and scene_description.get("camera_motion", {}).get("keyframes")))
        has_image_references = bool(scene_description.get("image_analysis_summary") or scene_description.get("main_objects") or scene_description.get("subjects"))

        prompt_instructions = ""
        if has_image_references and has_video_reference:
            prompt_instructions = """
ВАЖНО: У тебя есть два источника информации:
1. ИЗОБРАЖЕНИЯ (reference images): Используй из них объекты, материалы, композицию, статичные элементы
2. ВИДЕО-РЕФЕРЕНС (video reference): Используй из него движение камеры, подачу, динамику сцены, ритм

Объедини их так: объекты из изображений + движение и камера из видео.
"""
        elif has_video_reference:
            prompt_instructions = """
ВАЖНО: Используй видео-референс для движения камеры, подачи, динамики сцены.
"""
        elif has_image_references:
            prompt_instructions = """
ВАЖНО: Используй референсные изображения для объектов, материалов, композиции.
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
                prompt_instructions += f"\nДополнительные инструкции объединения (JSON):\n{serialized_notes}\n"

        # Для Veo 3.1 добавляем специальные инструкции по детализации
        detail_instructions = ""
        if platform == "veo3":
            detail_instructions = """

КРИТИЧЕСКИ ВАЖНО - ДЕТАЛИЗАЦИЯ ДЛЯ Veo 3.1:
- Промт должен быть ОЧЕНЬ ДЕТАЛЬНЫМ (минимум 300-500 слов, можно больше)
- ВАЖНО: Veo требует временную динамику - ВСЕГДА добавляй микродвижения объекта:
  * Проверь subject_motion_detected или camera_motion.subject_motion или metadata.auto_video_insights.auto_motion.subject_motion
  * Если есть движение объекта - опиши его детально (вращение, покачивание, и т.д.)
  * Если движения нет - ВСЕ РАВНО добавь микродвижения: "gently sway", "sparkle glints travel", "micro-reflections dance"
  * НИКОГДА не создавай статичный кадр - объект должен иметь хотя бы минимальное движение
- Используй ВСЕ данные из анализа, особенно:
  * subjects - опиши каждый объект (name, class, materials, colors)
  * camera_motion.keyframes - опиши каждый keyframe с временем (t_s), типом движения (type), позицией (position_m), поворотом (rotation_euler_deg), easing, duration
  * camera_motion.support - тип поддержки камеры (tripod, handheld, steadicam, gimbal, dolly, crane, drone)
  * camera_motion.dominant_motion - если "static_camera_subject_rotation", обязательно опиши движение объекта
  * lighting.lights - опиши КАЖДЫЙ источник света с ролью (role), позицией (position_m), дистанцией (distance_m), размером (size_m), модификатором (modifier), мягкостью (softness), интенсивностью (intensity_rel), цветовой температурой (color_temp_K), углом (angle_deg)
  * lighting.environment - опиши окружение (type, time_of_day, weather, reflections)
  * optics - используй ВСЕ данные: focal_length_mm, fov_deg, aperture_T, focus_distance_m, focus_pulls, distortion, vignetting, chromatic_aberration
  * composition - опиши композицию: shot_size (ECU/CU/MCU/MS/MLS/WS/EWS), framing (rule_of_thirds, symmetry, leading_lines, negative_space), depth_of_field, bokeh_notes
  * color_and_post - опиши цветокоррекцию: white_balance_K, look (contrast, saturation, hue shifts), grain_noise, halation_bloom, sharpness
  * media - используй fps, duration_s, aspect_ratio (если 1:1, используй 16:9), resolution_px
- Если есть видео-референс с camera_motion.keyframes, ОБЯЗАТЕЛЬНО включи детальное описание движения камеры ВО ВРЕМЕНИ - каждый keyframe должен быть описан отдельно с указанием того, что происходит в этот момент времени
- НЕ сокращай технические детали - Veo 3.1 требует максимальной специфичности
- Каждое движение камеры должно быть описано с указанием типа (pan/tilt/dolly/truck/pedestal/crane/arc/roll/zoom), направления, скорости, easing (linear/easeIn/easeOut/easeInOut)
- Каждый источник света должен быть описан с позицией относительно объекта/камеры, цветовой температурой в Кельвинах, интенсивностью
- Используй точные технические термины: например "85mm focal length at f/2.8", "shallow depth of field", "3200K warm key light from top-left at 45 degrees"
"""

        user_prompt = f"""На основе этого детального описания сцены создай ОЧЕНЬ ДЕТАЛЬНЫЙ и ИСЧЕРПЫВАЮЩИЙ промт для {self.platform_guides[platform]['name']}:
{prompt_instructions}
{detail_instructions}

ВХОДНЫЕ ДАННЫЕ (используй ВСЕ это):
{self._scene_description_to_json(scene_description)}

Платформа: {platform}
Тип задачи: {use_case}

Создай максимально детальный и технически точный промт, используя ВСЕ доступные данные из анализа."""

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
            "max_tokens": 2000 if platform == "veo3" else 1000,  # Для Veo 3.1 нужны более длинные промпты
            "temperature": 0.6  # Немного ниже для более точных технических деталей
        }

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

    def _polish_with_perplexity(
        self,
        draft_prompt: str,
        scene_description: Dict,
        platform: str,
        use_case: str
    ) -> str:
        """Делает финальный полиш через Perplexity."""
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
            "max_tokens": 2000 if platform == "veo3" else 1500,  # Для Veo 3.1 нужны более длинные промпты
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
        Генерирует промт для указанной платформы

        Args:
            scene_description: Структурированное описание сцены от Агента 1
            platform: veo3, sora2, seedream
            use_case: product_video, hero_image, gemstone_closeup, luxury_brand, general

        Returns:
            Dict с промтом и метаданными
        """
        if platform not in self.platform_guides:
            platform = "veo3"  # Fallback

        generated_prompt = self._generate_with_openai(scene_description, platform, use_case)

        # Если в описании есть raw_analysis (текст без структуры), используем его
        if "raw_analysis" in scene_description and len(scene_description) == 2:
            # Оптимизируем генерацию для текстового ответа
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
        """Генерирует промты для нескольких платформ одновременно"""
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
    # Тестовый пример
    generator = PromptGenerator()

    # Пример использования
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

