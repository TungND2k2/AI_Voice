"""S3 upload utility.

Reads config from environment variables:
  S3_ENDPOINT   — e.g. https://s3.xorcloud.net
  S3_REGION     — e.g. us-east-1
  S3_BUCKET     — bucket name
  S3_ACCESS_KEY
  S3_SECRET_KEY
  S3_PUBLIC_URL — optional base URL for public links (defaults to endpoint/bucket)
"""
import os
import mimetypes
from pathlib import Path

from loguru import logger


def _client():
    import boto3
    from botocore.config import Config
    config = Config(
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        config=config,
    )


def is_s3_enabled() -> bool:
    return all(os.environ.get(k) for k in ("S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY", "S3_SECRET_KEY"))


def upload_file(local_path: str, s3_key: str) -> str:
    """Upload a local file to S3 and return its public URL."""
    bucket = os.environ["S3_BUCKET"]
    content_type, _ = mimetypes.guess_type(local_path)
    content_type = content_type or "application/octet-stream"

    client = _client()
    with open(local_path, "rb") as f:
        data = f.read()
    client.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=data,
        ContentLength=len(data),
        ContentType=content_type,
        ACL="public-read",
    )

    base = os.environ.get("S3_PUBLIC_URL", "").rstrip("/")
    if not base:
        endpoint = os.environ["S3_ENDPOINT"].rstrip("/")
        base = f"{endpoint}/{bucket}"
    url = f"{base}/{s3_key}"
    logger.info(f"[S3] Uploaded {Path(local_path).name} → {url}")
    return url
