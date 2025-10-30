"""
Main pipeline: combines Agent 1 (visual analyzer) and Agent 2 (prompt generator)
"""

import json
from typing import Dict, Optional, Union
from pathlib import Path

from agent1_visual_analyzer import VisualAnalyzer
from agent2_prompt_generator import PromptGenerator


class VideoPromptPipeline:
    """Two-agent pipeline for automatic prompt generation from media"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        use_gemini_for_analysis: bool = False,
        perplexity_api_key: Optional[str] = None,
        auto_perplexity_polish: Optional[bool] = None
    ):
        """
        Initialize pipeline

        Args:
            openai_api_key: OpenAI API key (for analysis and generation)
            gemini_api_key: Google Gemini API key (optional, for video analysis)
            use_gemini_for_analysis: Use Gemini for analysis (better for video)
        """
        self.analyzer = VisualAnalyzer(
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key
        )
        self.generator = PromptGenerator(
            openai_api_key=openai_api_key,
            perplexity_api_key=perplexity_api_key,
            auto_polish=auto_perplexity_polish
        )
        self.use_gemini = use_gemini_for_analysis

    def process(
        self,
        media_source: Union[str, Path, bytes],
        platform: str = "veo3",
        use_case: str = "general",
        content_type: Optional[str] = None,
        return_intermediate: bool = False,
        polish_with_perplexity: Optional[bool] = None
    ) -> Dict:
        """
        Full processing cycle: media analysis → prompt generation

        Args:
            media_source: URL, file path or bytes media
            platform: veo3, sora2, seedream
            use_case: product_video, hero_image, gemstone_closeup, luxury_brand, general
            content_type: MIME type (optional, auto-detected)
            return_intermediate: Return intermediate analysis result

        Returns:
            Dict with final prompt and metadata
        """
        try:
            # Agent 1: Visual content analysis
            scene_description = self.analyzer.analyze(
                media_source=media_source,
                content_type=content_type,
                use_gemini=self.use_gemini
            )
        except Exception as e:
            # Fallback: if analysis failed, create basic description
            error_msg = str(e)
            scene_description = {
                "error": error_msg,
                "raw_analysis": f"Failed to analyze media: {error_msg}",
                "format": "error_fallback"
            }

        try:
            # Agent 2: Prompt generation
            prompt_result = self.generator.generate(
                scene_description=scene_description,
                platform=platform,
                use_case=use_case,
                polish_with_perplexity=polish_with_perplexity
            )
        except Exception as e:
            # Fallback: if generation failed, create basic prompt
            error_msg = str(e)
            prompt_result = {
                "prompt": f"[Error generating prompt: {error_msg}] Please provide a manual description of your scene.",
                "platform": platform,
                "platform_name": platform,
                "use_case": use_case,
                "metadata": {"error": error_msg, "format": "error_fallback"}
            }

        result = {
            "prompt": prompt_result["prompt"],
            "platform": prompt_result["platform"],
            "platform_name": prompt_result["platform_name"],
            "use_case": prompt_result["use_case"],
            "metadata": prompt_result["metadata"]
        }

        if return_intermediate:
            result["scene_analysis"] = scene_description

        return result

    def process_multiple_platforms(
        self,
        media_source: Union[str, Path, bytes],
        platforms: list[str] = None,
        use_case: str = "general",
        content_type: Optional[str] = None,
        polish_with_perplexity: Optional[bool] = None
    ) -> Dict:
        """
        Generates prompts for multiple platforms simultaneously

        Args:
            media_source: URL, file path or bytes media
            platforms: List of platforms (default ["veo3", "sora2"])
            use_case: Task type
            content_type: MIME type

        Returns:
            Dict with prompts for each platform
        """
        if platforms is None:
            platforms = ["veo3", "sora2"]

        # Agent 1: Analysis (once for all platforms)
        scene_description = self.analyzer.analyze(
            media_source=media_source,
            content_type=content_type,
            use_gemini=self.use_gemini
        )

        # Agent 2: Generation for all platforms
        prompts_by_platform = self.generator.generate_multiple(
            scene_description=scene_description,
            platforms=platforms,
            use_case=use_case,
            polish_with_perplexity=polish_with_perplexity
        )

        return {
            "scene_analysis": scene_description,
            "platforms": prompts_by_platform
        }


if __name__ == "__main__":
    # Usage example
    import os

    pipeline = VideoPromptPipeline(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        use_gemini_for_analysis=False
    )

    # Example 1: Single platform
    # result = pipeline.process(
    #     media_source="https://example.com/jewelry.jpg",
    #     platform="veo3",
    #     use_case="product_video"
    # )
    # print(json.dumps(result, indent=2, ensure_ascii=False))

    # Example 2: Multiple platforms
    # results = pipeline.process_multiple_platforms(
    #     media_source="https://example.com/jewelry_video.mp4",
    #     platforms=["veo3", "sora2"],
    #     use_case="gemstone_closeup"
    # )
    # print(json.dumps(results, indent=2, ensure_ascii=False))

