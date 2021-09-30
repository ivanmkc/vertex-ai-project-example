import csv
import os

from google.cloud import storage
from tensorflow.keras import utils
from typing import Any, Optional


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    destination_file_name = os.path.join("gs://", bucket_name, destination_blob_name)

    return destination_file_name


def download_object(bucket_name: str, blob_name: str) -> Optional[Any]:
    """Download a GCS object using GCSFuse"""
    import json

    # Download using GCSFuse
    gcs_fuse_path = f"/gcs/{bucket_name}/{blob_name}"

    with open(gcs_fuse_path) as json_file:
        return json.load(json_file)
