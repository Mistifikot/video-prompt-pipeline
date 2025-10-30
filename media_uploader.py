"""Utilities for uploading images to public storage (e.g., S3)."""

import os
import uuid
from typing import Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - boto3 may be optional
    boto3 = None
    BotoCoreError = ClientError = Exception


class ImageUploader:
    """Upload images to S3-compatible storage to get public URLs."""

    def __init__(self):
        self.bucket = os.getenv("S3_UPLOAD_BUCKET")
        if not self.bucket:
            raise RuntimeError("S3_UPLOAD_BUCKET not set")

        if boto3 is None:
            raise RuntimeError("boto3 not installed. Add 'boto3' to dependencies")

        self.region = os.getenv("S3_UPLOAD_REGION")
        self.prefix = os.getenv("S3_UPLOAD_PREFIX", "")
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix += "/"

        self.public_domain = os.getenv("S3_UPLOAD_PUBLIC_DOMAIN")
        self.force_public_acl = os.getenv("S3_UPLOAD_FORCE_PUBLIC", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.presign_expires = int(os.getenv("S3_UPLOAD_URL_EXPIRES", "900"))

        session_kwargs = {}
        if self.region:
            session_kwargs["region_name"] = self.region
        self.s3 = boto3.client("s3", **session_kwargs)

    def is_available(self) -> bool:
        return self.s3 is not None and self.bucket is not None

    def upload_image(self, data: bytes, filename: Optional[str], content_type: Optional[str] = None) -> str:
        if not data:
            raise ValueError("Empty image data")

        ext = ""
        if filename and "." in filename:
            ext = filename.rsplit(".", 1)[-1].lower()
            ext = f".{ext}"

        object_key = f"{self.prefix}{uuid.uuid4().hex}{ext}"

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        if self.public_domain or self.force_public_acl:
            extra_args["ACL"] = "public-read"

        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=data,
                **extra_args,
            )
        except (BotoCoreError, ClientError) as err:
            raise RuntimeError(f"Error uploading image to S3: {err}")

        if self.public_domain:
            base = self.public_domain.rstrip("/")
            return f"{base}/{object_key}"

        try:
            presigned = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": object_key},
                ExpiresIn=self.presign_expires
            )
        except (BotoCoreError, ClientError) as err:
            raise RuntimeError(f"Failed to create presigned URL: {err}")

        return presigned


def build_image_uploader() -> Optional[ImageUploader]:
    """Tries to create uploader, returns None if configuration is missing."""
    try:
        return ImageUploader()
    except RuntimeError as err:
        # If configuration missing, just return None, leaving logging to caller
        print(f"[i] ImageUploader unavailable: {err}")
        return None


