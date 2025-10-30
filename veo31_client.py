"""
Client for working with Google Veo 3.1 API
Support for "Video from samples" with reference images
"""

import os
import base64
import json
import requests
import time
from typing import List, Dict, Optional, Union
from pathlib import Path
from PIL import Image
from io import BytesIO
import mimetypes


class Veo31Client:
    """Client for Veo 3.1 API with reference image support"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Veo 3.1 client

        Args:
            api_key: Google API key (same as used for Gemini)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key for Veo 3.1 not specified. Set GEMINI_API_KEY in .env")

        # Veo 3.1 API endpoint
        # Note: actual endpoint may differ
        # Possible variants:
        # 1. Via Vertex AI: https://{location}-aiplatform.googleapis.com/v1
        # 2. Via Generative AI API (like Gemini)
        # 3. Via separate Veo API endpoint

        # For now using structure based on Gemini API
        # TODO: Update when official Veo 3.1 API documentation is received
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "veo-3.1"  # Veo 3.1 model name

        # Alternative via Vertex AI (if used)
        self.use_vertex_ai = os.getenv("USE_VERTEX_AI", "false").lower() == "true"
        if self.use_vertex_ai:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            self.base_url = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model}"

    def _encode_image(self, image_data: Union[bytes, str, Path]) -> str:
        """Encodes image to base64"""
        if isinstance(image_data, (str, Path)):
            path = Path(image_data)
            if path.exists():
                image_data = path.read_bytes()
            elif str(image_data).startswith(('http://', 'https://')):
                # Download image
                response = requests.get(str(image_data), timeout=30)
                image_data = response.content
            else:
                raise ValueError(f"Failed to load image: {image_data}")

        return base64.b64encode(image_data).decode('utf-8')

    def _prepare_reference_images(self, images: List[Union[str, Path, bytes, Dict]]) -> List[Dict]:
        """Prepares reference images for sending to API"""
        prepared_images = []

        for i, image in enumerate(images):
            try:
                image_data = None
                mime_type = 'image/jpeg'

                if isinstance(image, dict):
                    mime_type = image.get('content_type', mime_type)
                    if image.get('data') is not None:
                        image_data = image['data']
                    elif image.get('url'):
                        response = requests.get(image['url'], timeout=30)
                        response.raise_for_status()
                        image_data = response.content
                        mime_type = response.headers.get('content-type', mime_type)
                    elif image.get('path'):
                        path = Path(image['path'])
                        image_data = path.read_bytes()
                        mime_type = mime_type or mimetypes.guess_type(str(path))[0] or 'image/jpeg'
                elif isinstance(image, bytes):
                    image_data = image
                elif isinstance(image, (str, Path)):
                    if isinstance(image, str) and image.startswith(('http://', 'https://')):
                        response = requests.get(image, timeout=30)
                        response.raise_for_status()
                        image_data = response.content
                        mime_type = response.headers.get('content-type', mime_type)
                    else:
                        path = Path(image)
                        image_data = path.read_bytes()
                        mime_type = mimetypes.guess_type(str(path))[0] or mime_type

                if image_data is None:
                    raise ValueError("Failed to get image data")

                base64_image = base64.b64encode(image_data).decode('utf-8')

                prepared_images.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_image
                    }
                })
            except Exception as e:
                print(f"Error preparing image {i+1}: {e}")
                continue

        return prepared_images

    def generate_video_from_images(
        self,
        prompt: str,
        reference_images: List[Union[str, Path, bytes]],
        video_reference: Optional[Union[str, Path]] = None,
        duration_seconds: int = 5,
        aspect_ratio: str = "16:9",
        quality: str = "high"
    ) -> Dict:
        """
        Generates video in Veo 3.1 based on prompt and reference images

        Args:
            prompt: Text prompt (from analyzer)
            reference_images: List of 3 reference images (URL, path or bytes)
            video_reference: Optional reference video
            duration_seconds: Video duration in seconds (5-60)
            aspect_ratio: Aspect ratio (16:9, 9:16, 1:1)
            quality: Quality (high, standard)

        Returns:
            Dict with information about generated video
        """
        if len(reference_images) < 1:
            raise ValueError("At least 1 reference image required")

        if len(reference_images) > 3:
            print(f"Warning: {len(reference_images)} images specified, using first 3")
            reference_images = reference_images[:3]

        # Prepare references
        prepared_images = self._prepare_reference_images(reference_images)

        # Form request
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "contents": [{
                "parts": [
                    {
                        "text": prompt
                    },
                    *prepared_images  # Add all reference images
                ],
                "generation_config": {
                    "temperature": 0.7,
                    "video_length": f"{duration_seconds}s",
                    "aspect_ratio": aspect_ratio,
                    "quality": quality
                }
            }]
        }

        # If reference video exists, add it
        if video_reference:
            # For video need to use File API (similar to Gemini)
            video_file = self._upload_video_reference(video_reference)
            if video_file:
                payload["contents"][0]["parts"].append({
                    "file_data": {
                        "mime_type": "video/mp4",
                        "file_uri": video_file["uri"]
                    }
                })

        # Send request
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()

            # Veo 3.1 returns information about generation task
            # Need to periodically check status
            if "video" in result or "task" in result:
                return {
                    "status": "processing" if "task" in result else "completed",
                    "task_id": result.get("task", {}).get("id"),
                    "video_url": result.get("video", {}).get("url"),
                    "result": result
                }
            else:
                return {
                    "status": "completed",
                    "result": result,
                    "video_url": result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("video", {}).get("url")
                }

        except requests.RequestException as e:
            raise Exception(f"Error generating video: {e}")

    def _upload_video_reference(self, video_path: Union[str, Path, bytes, Dict]) -> Optional[Dict]:
        """Uploads reference video via File API"""
        try:
            content_type = 'video/mp4'
            if isinstance(video_path, dict):
                content_type = video_path.get('content_type', content_type)
                if video_path.get('data') is not None:
                    video_data = video_path['data']
                elif video_path.get('url'):
                    response = requests.get(str(video_path['url']), timeout=60)
                    response.raise_for_status()
                    video_data = response.content
                    content_type = response.headers.get('content-type', content_type)
                elif video_path.get('path'):
                    path = Path(video_path['path'])
                    video_data = path.read_bytes()
                    content_type = content_type or mimetypes.guess_type(str(path))[0] or 'video/mp4'
                else:
                    return None
            elif isinstance(video_path, bytes):
                video_data = video_path
            elif isinstance(video_path, (str, Path)):
                path = Path(video_path)
                if path.exists():
                    video_data = path.read_bytes()
                    content_type = mimetypes.guess_type(str(path))[0] or content_type
                elif str(video_path).startswith(('http://', 'https://')):
                    response = requests.get(str(video_path), timeout=60)
                    response.raise_for_status()
                    video_data = response.content
                    content_type = response.headers.get('content-type', content_type)
                else:
                    return None
            else:
                return None

            # Upload via File API
            # Use correct format according to Gemini File API
            upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.api_key}"

            # Determine MIME type
            mime_type = content_type or 'video/mp4'

            # Create multipart/related request
            filename = "video_reference.mp4"

            # Create metadata
            metadata = {
                "file": {
                    "display_name": "video_reference"
                }
            }

            # Form request according to documentation
            headers = {
                'X-Goog-Upload-Protocol': 'multipart'
            }

            files = {
                'metadata': ('metadata', json.dumps(metadata), 'application/json; charset=UTF-8'),
                'file': (filename, video_data, mime_type)
            }

            response = requests.post(upload_url, headers=headers, files=files, timeout=120)
            if response.status_code >= 400:
                raise Exception(f"{response.status_code} {response.reason}: {response.text}")

            return response.json().get("file", {})

        except Exception as e:
            print(f"Error uploading video reference: {e}")
            return None

    def check_video_status(self, task_id: str) -> Dict:
        """Checks video generation status"""
        url = f"{self.base_url}/tasks/{task_id}?key={self.api_key}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Error checking status: {e}")


# Alternative via Vertex AI (if used)
class Veo31VertexClient:
    """Client via Vertex AI for Veo 3.1"""

    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location

    def generate_video(self, prompt: str, reference_images: List[str], **kwargs):
        """Generation via Vertex AI"""
        # Implementation via Vertex AI SDK
        pass


if __name__ == "__main__":
    # Test
    client = Veo31Client()

    # Usage example
    result = client.generate_video_from_images(
        prompt="Rose gold ring with pink diamonds rotating slowly",
        reference_images=[
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "https://example.com/image3.jpg"
        ]
    )

    print(result)

