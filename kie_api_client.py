"""Client for working with Kie.ai Veo 3.1 API"""

import os
from typing import List, Optional, Dict, Any

import requests


class KieVeoClient:
    """Wrapper over REST API https://api.kie.ai/api/v1/veo/generate"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("KIE_API_KEY")
        if not self.api_key:
            raise ValueError("Kie.ai API key not found. Set KIE_API_KEY in .env")

        self.base_url = (base_url or os.getenv("KIE_API_BASE_URL") or "https://api.kie.ai").rstrip("/")

    def generate_video(
        self,
        prompt: str,
        image_urls: Optional[List[str]] = None,
        model: str = "veo3_fast",
        aspect_ratio: str = "16:9",
        generation_type: Optional[str] = None,
        enable_translation: bool = True,
        callback_url: Optional[str] = None,
        seeds: Optional[int] = None,
        watermark: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates video generation task via Kie.ai"""

        # Clean and limit prompt
        # Remove double quotes at start and end if present
        cleaned_prompt = prompt.strip()
        if cleaned_prompt.startswith('"') and cleaned_prompt.endswith('"'):
            cleaned_prompt = cleaned_prompt[1:-1]

        # Limit prompt length (Kie.ai supports longer prompts, but we limit for safety)
        # Based on testing, Kie.ai can handle at least 2000-3000 characters
        MAX_PROMPT_LENGTH = 2500

        if len(cleaned_prompt) > MAX_PROMPT_LENGTH:
            print(f"[WARNING] Prompt too long ({len(cleaned_prompt)} characters), truncating to {MAX_PROMPT_LENGTH}")
            # Try to truncate at paragraph boundary first (double newline)
            truncated = cleaned_prompt[:MAX_PROMPT_LENGTH]
            last_paragraph = truncated.rfind('\n\n')
            if last_paragraph > MAX_PROMPT_LENGTH * 0.5:  # If paragraph break is within reasonable range
                cleaned_prompt = truncated[:last_paragraph].strip()
            else:
                # Fallback: truncate to last sentence within limit
                last_period = truncated.rfind('.')
                if last_period > MAX_PROMPT_LENGTH * 0.7:  # If period is within reasonable range
                    cleaned_prompt = truncated[:last_period + 1]
                else:
                    cleaned_prompt = truncated

        payload: Dict[str, Any] = {
            "prompt": cleaned_prompt,
            "model": model,
            "aspectRatio": aspect_ratio,
            "enableTranslation": enable_translation,
        }

        if image_urls:
            payload["imageUrls"] = image_urls

        if generation_type:
            payload["generationType"] = generation_type
        elif image_urls:
            # Kie requires REFERENCE_2_VIDEO mode for imageUrls
            payload["generationType"] = "REFERENCE_2_VIDEO"

        if callback_url:
            payload["callBackUrl"] = callback_url

        if seeds is not None:
            payload["seeds"] = seeds

        if watermark:
            payload["watermark"] = watermark

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        url = f"{self.base_url}/api/v1/veo/generate"

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            # Improved error handling for debugging
            error_detail = f"HTTP {response.status_code}"
            try:
                error_json = response.json()
                error_detail += f": {error_json}"
            except:
                error_detail += f": {response.text[:500]}"

            print(f"[ERROR] Kie.ai API error: {error_detail}")
            print(f"[DEBUG] Payload sent (prompt length: {len(payload.get('prompt', ''))}): {payload}")

            # Raise exception with details
            raise requests.HTTPError(f"Kie.ai API error: {error_detail}")
        except requests.RequestException as e:
            print(f"[ERROR] Kie.ai request failed: {e}")
            raise

    def check_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Checks video generation task status by Task ID and returns video URL if ready

        According to Kie.ai documentation, there are two ways to get result:
        1. Callback URL (recommended) - system automatically sends result
        2. Polling via "Get Veo3.1 Video Details" endpoint (exact URL not specified in documentation)

        If status check endpoint is unavailable, try checking video availability directly.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # According to documentation, endpoint may be:
        # - /api/v1/veo/detail?taskId={task_id}
        # - /api/v1/veo/detail/{task_id}
        # But exact format not specified, try different variants
        endpoints = [
            # Variant 1: Query parameter (according to "Get Veo3.1 Video Details" documentation)
            f"{self.base_url}/api/v1/veo/detail?taskId={task_id}",
            # Variant 2: Path parameter
            f"{self.base_url}/api/v1/veo/detail/{task_id}",
            # Variant 3: Without /api
            f"{self.base_url}/v1/veo/detail?taskId={task_id}",
            f"{self.base_url}/v1/veo/detail/{task_id}",
            # Variant 4: Alternative variants (old, in case API changed)
            f"{self.base_url}/api/v1/veo/task/{task_id}",
            f"{self.base_url}/v1/video/task/{task_id}",
            f"{self.base_url}/api/v1/video/task/{task_id}",
        ]

        last_error = None
        last_response = None
        tried_endpoints = []

        for url in endpoints:
            tried_endpoints.append(url)
            try:
                print(f"[i] Trying endpoint: {url}")
                response = requests.get(url, headers=headers, timeout=30)
                last_response = response

                print(f"[i] Response: HTTP {response.status_code}")

                if response.status_code == 200:
                    print(f"[OK] Status endpoint works: {url}")
                    data = response.json()

                    # Log full response for debugging (first 500 characters)
                    print(f"[DEBUG] Full API response: {str(data)[:500]}")

                    result = {
                        "status": data.get("status", "unknown"),
                        "task_id": task_id,
                        "raw_response": data
                    }

                    # Extract video_url from different possible locations in response
                    video_url = None
                    if isinstance(data, dict):
                        # Format from callback (according to documentation):
                        # {
                        #   "code": 200,
                        #   "data": {
                        #     "taskId": "...",
                        #     "info": {
                        #       "resultUrls": ["https://tempfile.aiquickdraw.com/v/..."],
                        #       "originUrls": [...],
                        #       "resolution": "1080p"
                        #     }
                        #   }
                        # }

                        # Variant 1: data.data.info.resultUrls (nested data structure)
                        if "data" in data and isinstance(data["data"], dict):
                            nested_data = data["data"].get("data", {})
                            if isinstance(nested_data, dict):
                                info = nested_data.get("info", {})
                                if isinstance(info, dict):
                                    result_urls = info.get("resultUrls", [])
                                    if isinstance(result_urls, list) and len(result_urls) > 0:
                                        video_url = result_urls[0]
                                    elif isinstance(result_urls, str):
                                        video_url = result_urls

                        # Variant 1b: data.data.info.resultUrls (single data level - standard callback format)
                        if not video_url and "data" in data and isinstance(data["data"], dict):
                            info = data["data"].get("info", {})
                            if isinstance(info, dict):
                                result_urls = info.get("resultUrls", [])
                                if isinstance(result_urls, list) and len(result_urls) > 0:
                                    video_url = result_urls[0]
                                elif isinstance(result_urls, str):
                                    video_url = result_urls

                        # Variant 2: data.data.resultUrls (nested data without info)
                        if not video_url and "data" in data and isinstance(data["data"], dict):
                            nested_data = data["data"].get("data", {})
                            if isinstance(nested_data, dict):
                                result_urls = nested_data.get("resultUrls", [])
                                if isinstance(result_urls, list) and len(result_urls) > 0:
                                    video_url = result_urls[0]
                                elif isinstance(result_urls, str):
                                    video_url = result_urls

                        # Variant 2b: data.data.resultUrls (single data level without info)
                        if not video_url and "data" in data and isinstance(data["data"], dict):
                            result_urls = data["data"].get("resultUrls", [])
                            if isinstance(result_urls, list) and len(result_urls) > 0:
                                video_url = result_urls[0]
                            elif isinstance(result_urls, str):
                                video_url = result_urls

                        # Variant 3: data.result.video_url
                        if not video_url and "result" in data:
                            result_obj = data["result"]
                            if isinstance(result_obj, dict):
                                video_url = result_obj.get("video_url") or result_obj.get("videoUrl") or result_obj.get("video")
                            elif isinstance(result_obj, str) and result_obj.startswith("http"):
                                video_url = result_obj

                        # Variant 4: data.video_url (directly in root)
                        if not video_url:
                            video_url = data.get("video_url") or data.get("videoUrl") or data.get("video")

                        # Variant 5: Search for any link to tempfile.aiquickdraw.com via regex
                        if not video_url:
                            import re
                            json_str = str(data)
                            video_patterns = [
                                r'(https?://tempfile\.aiquickdraw\.com/[^\s"\']+\.mp4)',
                                r'(https?://[^\s"\']+tempfile[^\s"\']+\.mp4)',
                                r'(https?://[^\s"\']+\.mp4)',
                            ]
                            for pattern in video_patterns:
                                matches = re.findall(pattern, json_str)
                                if matches:
                                    # Take first found URL and check its availability
                                    candidate_url = matches[0]
                                    try:
                                        # Quick URL availability check
                                        check_response = requests.head(candidate_url, timeout=2, allow_redirects=True)
                                        if check_response.status_code == 200:
                                            video_url = candidate_url
                                            print(f"[i] Found available video link via regex: {video_url}")
                                            break
                                    except:
                                        # If check failed, still take URL (may be temporary issue)
                                        video_url = candidate_url
                                        print(f"[i] Found video link via regex (check unavailable): {video_url}")
                                        break

                    if video_url:
                        # Additional video_url availability check before returning
                        try:
                            verify_response = requests.head(video_url, timeout=2, allow_redirects=True)
                            if verify_response.status_code != 200:
                                print(f"[!] Video URL found but unavailable (HTTP {verify_response.status_code}): {video_url}")
                                # Still return URL as it may become available later
                        except:
                            print(f"[i] Video URL found but availability check failed: {video_url}")
                            # Continue as this may be temporary network issue

                        result["video_url"] = video_url
                        result["status"] = "completed"
                        print(f"[OK] Video ready! URL: {video_url}")
                    else:
                        # Check code in response
                        code = data.get("code")
                        if code == 200:
                            result["status"] = "completed"
                            print(f"[i] Status code 200, but video_url not found in response")
                        elif code:
                            result["status"] = "failed" if code >= 400 else "processing"

                    return result
                elif response.status_code == 404:
                    print(f"[i] Endpoint {url} returned 404, trying next...")
                    continue
                elif response.status_code == 401:
                    error_text = response.text[:200]
                    print(f"[!] Endpoint {url} returned 401 (Unauthorized): {error_text}")
                    continue
                else:
                    error_text = response.text[:500]
                    print(f"[!] Endpoint {url} returned {response.status_code}: {error_text}")
                    # If not 404, endpoint may exist but format is wrong
                    continue
            except requests.RequestException as e:
                last_error = e
                print(f"[!] Request error to {url}: {e}")
                continue

        # If all endpoints returned 404, try alternative approach:
        # Check video availability directly via known URL format
        print(f"[i] All endpoints returned 404, trying to check video availability directly...")

        # URL format: https://tempfile.aiquickdraw.com/v/{task_id}_{timestamp}.mp4
        # Try checking availability with current timestamp and several variants around it
        import time
        current_timestamp = int(time.time())

        # Try several timestamp variants:
        # - Current time and several minutes ago (video may have been generated recently)
        # - Check at 5-minute intervals for last 2 hours
        test_timestamps = []
        for offset in range(0, 7200, 300):  # Every 5 minutes for last 2 hours
            test_timestamps.append(current_timestamp - offset)

        # Also add more precise check for last 10 minutes (every minute)
        for offset in range(0, 600, 60):
            test_timestamps.append(current_timestamp - offset)

        # Remove duplicates and sort
        test_timestamps = sorted(set(test_timestamps))

        for test_timestamp in test_timestamps:
            test_url = f"https://tempfile.aiquickdraw.com/v/{task_id}_{test_timestamp}.mp4"

            try:
                # Try HEAD request
                head_response = requests.head(test_url, timeout=3, allow_redirects=True)
                if head_response.status_code == 200:
                    print(f"[OK] Video found directly: {test_url}")
                    return {
                        "status": "completed",
                        "task_id": task_id,
                        "video_url": test_url,
                        "method": "direct_check"
                    }
                # If HEAD not supported, try GET with minimal reading
                elif head_response.status_code == 405:  # Method Not Allowed
                    get_response = requests.get(test_url, timeout=3, stream=True)
                    if get_response.status_code == 200:
                        # Read only first bytes for check
                        get_response.raw.read(1024)
                        get_response.close()
                        print(f"[OK] Video found via GET: {test_url}")
                        return {
                            "status": "completed",
                            "task_id": task_id,
                            "video_url": test_url,
                            "method": "direct_check_get"
                        }
            except requests.exceptions.RequestException:
                continue
            except Exception as e:
                # Ignore other errors
                continue

        # If not found via direct access, return error
        error_messages = []
        if last_response:
            error_messages.append(f"Last response: HTTP {last_response.status_code}")
            if last_response.text:
                error_messages.append(f"Server response: {last_response.text[:300]}")
        if last_error:
            error_messages.append(f"Request error: {last_error}")

        error_detail = ". ".join(error_messages) if error_messages else "All endpoints returned 404"
        error_detail += f". Tried endpoints: {len(tried_endpoints)} variants. Direct video check also gave no result."

        # Don't raise exception, return "processing" status so interface continues checking
        return {
            "status": "processing",
            "task_id": task_id,
            "error": error_detail,
            "note": "Status check endpoint unavailable. Recommended to use callback URL or check manually at https://app.kie.ai"
        }


