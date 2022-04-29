import apache_beam as beam
from typing import Dict, List, Tuple
import numpy as np

import logging
import json
import sys

import distinctipy

TRAINING_BASE = "labels_"
FOLDER = "high-res-patches"
STAGING_BUCKET = "ivanmkc-palm-data-2"

pattern = "gs://" + STAGING_BUCKET + "/" + FOLDER + "/" + TRAINING_BASE + "*"
# pattern = "gs://ivanmkc-palm-data/high-res-patches/labels_0.tfrecord.gz"

number_to_label_map = {
    0: "no data",
    1: "water",
    2: "opaque clouds",
    3: "trees and shrubs",
    4: "built surface",
    5: "bridges and dams",
    6: "grass",
    7: "plant ground mix",
    8: "crops",
    9: "palm plantations",
    10: "flooded vegetation",
    11: "bareground and sand",
    12: "snow and ice",
    13: "unknown",
}

colors = [
    np.array(list(color)) * 255
    for color in distinctipy.get_colors(len(number_to_label_map))
]

# Make sure each color is unique
assert len(colors) == len(set([tuple(color) for color in colors]))

number_to_color_map = {
    number: color for number, color in zip(number_to_label_map.keys(), colors)
}


def convert_record_to_list_of_patches(x) -> List[Tuple[np.ndarray, np.ndarray]]:
    import tensorflow as tf

    # Each tile is from 30cm WV3 satellite imagery, is 1024px x 1024px and is labelled twice.
    PATCH_SIZE = 512

    BANDS = ["R", "G", "B"]
    LABELS_NAMES = ["label_1", "label_2"]
    FEATURES = BANDS + LABELS_NAMES

    # Specify the size and shape of patches expected by the model.
    KERNEL_SHAPE = [PATCH_SIZE, PATCH_SIZE]
    COLUMNS = [
        tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for _ in FEATURES
    ]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

    def parse_tfrecord(example_proto):
        """The parsing function.
        Read a serialized example into the structure defined by FEATURES_DICT.
        Args:
          example_proto: a serialized Example.
        Returns:
          A dictionary of tensors, keyed by feature name.
        """
        return tf.io.parse_single_example(example_proto, FEATURES_DICT)

    def to_tuple(tensor):
        """Function to convert a tensor to a tuple of (inputs, outputs).
        Args:
          tensor: A stacked tensor, with label last.
        Returns:
          A tuple of (inputs, outputs).
        """
        return tensor[:, :, : len(BANDS)], tensor[:, :, len(BANDS) :]

    def flatten_patches(inputs):
        """Function to convert a dictionary of tensors to two stacked
          tensors in HWC shape.
        Args:
          inputs: A dictionary of tensors, keyed by feature name.
        Returns:
          A tf.data.Dataset with two examaples in it.
        """
        inputsList = [inputs.get(key) for key in BANDS]
        label_1 = [inputs.get(LABELS_NAMES[0])]
        label_2 = [inputs.get(LABELS_NAMES[1])]
        stack1 = tf.stack(inputsList + label_1, axis=0)
        stack2 = tf.stack(inputsList + label_2, axis=0)
        # Convert from CHW to HWC
        return tf.data.Dataset.from_tensor_slices(
            [
                tf.transpose(stack1, [1, 2, 0]),
                tf.transpose(stack2, [1, 2, 0]),
            ]
        )

    return [to_tuple(x) for x in flatten_patches(parse_tfrecord(x)).as_numpy_iterator()]


output_gcs_blob_folder = "image_and_masks"


def convert_image_and_mask_to_managed_dataset_record(image_and_mask) -> Dict:
    import uuid
    from typing import Dict
    import os
    import numpy as np
    from PIL import Image
    import tensorflow as tf

    from google.cloud import storage

    def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        destination_file_name = os.path.join(bucket_name, destination_blob_name)

        return destination_file_name
        # destination = os.path.join("/bigstore", bucket_name, destination_blob_name)
        # ! fileutil_bs cp -f {source_file_name} {destination}

    def convert_example_to_dict(
        image_np: np.ndarray,
        mask_np: np.ndarray,
        name: str,
        output_gcs_bucket: str,
        output_gcs_blob_folder: str,
        train_val_test: str,
        show_image: bool = True,
    ) -> Dict:
        # Create image
        image = tf.keras.utils.array_to_img(image_np)

        # Write to temp filea
        temp_filename = "temp.png"
        image.save(temp_filename, format="png")

        # Create image mask matrix
        image_mask_matrix = mask_np.reshape(mask_np.shape[:2])
        # Rewrite as RGB
        image_mask_matrix_rgb = np.zeros(
            (image_mask_matrix.shape[0], image_mask_matrix.shape[1], 3),
            dtype=np.uint8,
        )
        numbers_used = set()
        for number in number_to_label_map.keys():
            color = number_to_color_map[number]
            if len(image_mask_matrix_rgb[..., :][image_mask_matrix == number]) > 0:
                image_mask_matrix_rgb[..., :][image_mask_matrix == number] = color
                numbers_used.add(number)

        # Convert back to image
        image_mask_rgb = Image.fromarray(image_mask_matrix_rgb)
        mask_filename = "mask.png"
        image_mask_rgb.save(mask_filename, format="png")

        # if show_image:
        #   image_mask = Image.fromarray(np.uint8(image_mask_matrix>0)*255, mode="L")

        #   image = Image.open(BytesIO(image_encoded))
        #   image.paste(image_mask_rgb, (0, 0), image_mask)
        #   imshow(image)

        # Upload image to GCS
        output_gcs_blob_file = os.path.join(output_gcs_blob_folder, f"{name}.png")
        upload_blob(output_gcs_bucket, temp_filename, output_gcs_blob_file)

        # Upload mask to GCS
        mask_gcs_blob_file = os.path.join(output_gcs_blob_folder, f"{name}_mask.png")
        upload_blob(output_gcs_bucket, mask_filename, mask_gcs_blob_file)

        # Check for duplicates
        # image_checksum = hashlib.md5(image.tobytes()).hexdigest()
        # if image_checksum in self.file_to_checksum_map:
        #   raise ValueError(f"Processing {name}, but duplicate image found in {self.file_to_checksum_map[image_checksum]}")
        # else:
        #   self.file_to_checksum_map[image_checksum] = name

        mask_annotation = {
            "categoryMaskGcsUri": f"gs://{os.path.join(output_gcs_bucket, mask_gcs_blob_file)}",
            "annotationSpecColors": [
                {
                    "color": {
                        "red": color[0] / 255,
                        "green": color[1] / 255,
                        "blue": color[2] / 255,
                    },
                    "displayName": number_to_label_map[number],
                }
                for number, color in number_to_color_map.items()
                if number in numbers_used
            ],
        }

        return {
            "imageGcsUri": f"gs://{os.path.join(output_gcs_bucket, output_gcs_blob_file)}",
            "segmentationAnnotation": mask_annotation,
            "dataItemResourceLabels": {
                "aiplatform.googleapis.com/ml_use": train_val_test
            },
        }

    image_np, mask_np = image_and_mask

    return convert_example_to_dict(
        image_np=image_np,
        mask_np=mask_np,
        name=uuid.uuid4(),
        output_gcs_bucket=STAGING_BUCKET,
        output_gcs_blob_folder=output_gcs_blob_folder,
        train_val_test="train",
    )


def run(argv=None):
    with beam.Pipeline(argv=argv) as pipeline:
        prefix = "train"
        (
            pipeline
            | beam.io.tfrecordio.ReadFromTFRecord(pattern)
            | beam.FlatMap(convert_record_to_list_of_patches)
            | beam.Map(convert_image_and_mask_to_managed_dataset_record)
            # | beam.Map(print)
            # | beam.combiners.ToList()
            | beam.Map(json.dumps)
            # | beam.io.WriteToText(file_path_prefix=TEMP_FILE, file_name_suffix="jsonl")
            | beam.io.WriteToText(
                file_path_prefix=f"gs://{STAGING_BUCKET}/jsonl/{prefix}/{prefix}",
                file_name_suffix=".jsonl",
            )
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run(sys.argv)
