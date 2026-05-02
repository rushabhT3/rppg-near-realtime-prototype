import os
import uuid
from datetime import timedelta

from app.core.logging import get_logger

logger = get_logger()

BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "vitalis-uploads")
_gcs_available = None


def _check_gcs():
    """Lazy-check if google-cloud-storage is importable."""
    global _gcs_available
    if _gcs_available is None:
        try:
            from google.cloud import storage  # noqa: F401

            _gcs_available = True
        except ImportError:
            _gcs_available = False
            logger.warning("google-cloud-storage not installed; GCS features disabled")
    return _gcs_available


def _get_bucket():
    from google.cloud import storage
    from google.auth import default
    from google.auth.transport.requests import Request

    client = storage.Client()
    return client.bucket(BUCKET_NAME)


def generate_signed_upload_url(filename: str, expiration_min: int = 15) -> dict:
    """Generate a signed URL for the frontend to upload directly to GCS."""
    if not _check_gcs():
        raise RuntimeError(
            "GCS storage is not available. Install google-cloud-storage."
        )

    bucket = _get_bucket()
    session_id = str(uuid.uuid4())
    object_name = f"uploads/{session_id}_{filename}"

    blob = bucket.blob(object_name)

    from google.auth import default
    from google.auth.transport.requests import Request as GCSRequest

    credentials, _ = default()
    credentials.refresh(GCSRequest())

    upload_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_min),
        method="PUT",
        content_type="video/mp4",
        service_account_email=credentials.service_account_email,
        access_token=credentials.token,
    )

    return {
        "session_id": session_id,
        "upload_url": upload_url,
        "object_name": object_name,
    }


def download_to_local(object_name: str) -> str:
    """Download a GCS object to a local temp file and return the path."""
    if not _check_gcs():
        raise RuntimeError("GCS storage is not available.")

    bucket = _get_bucket()
    blob = bucket.blob(object_name)

    local_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp_uploads")
    os.makedirs(local_dir, exist_ok=True)

    filename = object_name.split("/")[-1]
    local_path = os.path.join(local_dir, filename)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded gs://{BUCKET_NAME}/{object_name} -> {local_path}")

    return local_path


def delete_object(object_name: str):
    """Delete a GCS object after processing."""
    if not _check_gcs():
        return

    bucket = _get_bucket()
    blob = bucket.blob(object_name)
    blob.delete()
    logger.info(f"Deleted gs://{BUCKET_NAME}/{object_name}")
