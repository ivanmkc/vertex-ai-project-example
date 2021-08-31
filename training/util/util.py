import csv
import os

from google.cloud import storage
from tensorflow.keras import utils


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    destination_file_name = os.path.join("gs://", bucket_name, destination_blob_name)

    return destination_file_name
