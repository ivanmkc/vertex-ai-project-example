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


def download_data(data_dir):
    """Download data."""

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
    dataset = utils.get_file(
        "stack_overflow_16k.tar.gz",
        url,
        untar=True,
        cache_dir=data_dir,
        cache_subdir="",
    )
    data_dir = os.path.join(os.path.dirname(dataset))

    return data_dir


def upload_train_data_to_gcs(train_data_dir, bucket_name, destination_blob_prefix):
    """Create CSV file using train data content."""

    train_data_dir = os.path.join(data_dir, "train")
    train_data_fn = os.path.join(data_dir, "train.csv")

    fp = open(train_data_fn, "w", encoding="utf8")
    writer = csv.writer(
        fp, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL, lineterminator="\n"
    )

    for root, _, files in os.walk(train_data_dir):
        for file in files:
            if file.endswith(".txt"):
                class_name = root.split("/")[-1]
                file_fn = os.path.join(root, file)
                with open(file_fn, "r") as f:
                    content = f.readlines()
                    lines = [x.strip().strip('"') for x in content]
                    writer.writerow((lines[0], class_name))

    fp.close()

    train_gcs_url = upload_blob(
        bucket_name, train_data_fn, os.path.join(destination_blob_prefix, "train.csv")
    )

    return train_gcs_url
