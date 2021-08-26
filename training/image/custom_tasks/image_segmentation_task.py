# Single, Mirror and Multi-Machine Distributed Training for CIFAR-10

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow_examples.models.pix2pix import pix2pix

from tensorflow.python.client import device_lib
import argparse
import os
import sys
import json
from typing import List


print("Python Version = {}".format(sys.version))
print("TensorFlow Version = {}".format(tf.__version__))
print("TF_CONFIG = {}".format(os.environ.get("TF_CONFIG", "Not found")))
print("DEVICES", device_lib.list_local_devices())


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Keras Image Segmentation")
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of training epochs"
    )
    # parser.add_argument("--image-width", default=32, type=int, help="image width")
    # parser.add_argument("--image-height", default=32, type=int, help="image height")
    parser.add_argument("--batch-size", default=16, type=int, help="mini-batch size")
    parser.add_argument(
        "--model-dir",
        default=os.getenv("AIP_MODEL_DIR"),
        type=str,
        help="model directory",
    )
    parser.add_argument("--data-dir", default="./data", type=str, help="data directory")
    # parser.add_argument(
    #     "--test-run",
    #     default=False,
    #     type=str2bool,
    #     help="test run the training application, i.e. 1 epoch for training using sample dataset",
    # )
    parser.add_argument("--model-version", default=1, type=int, help="model version")
    parser.add_argument(
        "--lr", dest="lr", default=0.01, type=float, help="Learning rate."
    )
    parser.add_argument(
        "--steps",
        dest="steps",
        default=200,
        type=int,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--distribute",
        dest="distribute",
        type=str,
        default="single",
        help="distributed training strategy",
    )

    args = parser.parse_args()
    return args


args = parse_args()


import numpy as np
import json
from typing import Any, Dict, List, Tuple
from PIL import Image
from io import BytesIO


def create_dataset_from_uri_pattern(dataset_uri_pattern: str) -> List[str]:
    instances = []
    for aip_data_uri in tf.io.gfile.glob(pattern=dataset_uri_pattern):
        with tf.io.gfile.GFile(name=aip_data_uri, mode="r") as gfile:
            for line in gfile.readlines():
                instance = json.loads(line)
                instances.append(instance)
    return instances


# Get strategy
# Single Machine, single compute device
if args.distribute == "single":
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
# Single Machine, multiple compute device
elif args.distribute == "mirror":
    strategy = tf.distribute.MirroredStrategy()
# Multiple Machine, multiple compute device
elif args.distribute == "multi":
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
else:
    raise ValueError(f"Unknown distrubute argument provided: {args.distribute}")

# Multi-worker configuration
print("num_replicas_in_sync = {}".format(strategy.num_replicas_in_sync))

NUM_WORKERS = strategy.num_replicas_in_sync
# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size.
GLOBAL_BATCH_SIZE = args.batch_size * NUM_WORKERS

aip_model_dir = os.environ.get("AIP_MODEL_DIR")
aip_data_format = os.environ.get("AIP_DATA_FORMAT")
aip_training_data_uri = os.environ.get("AIP_TRAINING_DATA_URI")
aip_validation_data_uri = os.environ.get("AIP_VALIDATION_DATA_URI")
# aip_test_data_uri = os.environ.get("AIP_TEST_DATA_URI")

print(f"aip_model_dir: {aip_model_dir}")
print(f"aip_data_format: {aip_data_format}")
print(f"aip_training_data_uri: {aip_training_data_uri}")
print(f"aip_validation_data_uri: {aip_validation_data_uri}")
# print(f"aip_test_data_uri: {aip_test_data_uri}")

print("Loading AIP datasets")
train_instances, validation_instances = (
    create_dataset_from_uri_pattern(dataset_uri_pattern)
    for dataset_uri_pattern in [
        aip_training_data_uri,
        aip_validation_data_uri,
    ]
)
print("AIP test dataset is loaded")

from functools import partial

# Extract color labels from training instances
color_labels = {
    annotation_spec_color["displayName"]
    for instance in train_instances
    for annotation_spec_color in instance["maskAnnotation"]["annotationSpecColors"]
}
color_labels_to_indices = {
    label: index for index, label in enumerate(color_labels, start=1)
}


def convert_instance_to_features(
    raw_instance: Dict, color_labels_to_indices: Dict[str, int]
) -> Tuple[str, List, List]:
    image_uri = raw_instance["imageGcsUri"]

    # Download image
    image_data = tf.io.gfile.GFile(image_uri, "rb").read()
    image_matrix = np.array(Image.open(BytesIO(image_data)))

    # Gather colors
    label_to_color_map = {}
    for annotation_spec_color in raw_instance["maskAnnotation"]["annotationSpecColors"]:
        color = annotation_spec_color["color"]
        color_array = np.array([color["red"], color["green"], color["blue"]]) * 255
        label_to_color_map[annotation_spec_color["displayName"]] = color_array.astype(
            np.uint8
        )

    # Download mask
    mask_uri = raw_instance["maskAnnotation"]["categoryMaskGcsUri"]
    mask_data = tf.io.gfile.GFile(mask_uri, "rb").read()
    mask_matrix = np.array(Image.open(BytesIO(mask_data)))

    # Recolor mask
    mask_recolored = np.zeros(
        (mask_matrix.shape[0], mask_matrix.shape[1], 1), dtype=np.uint8
    )

    # print(f"label_to_color_map: {label_to_color_map}")
    for label, color in label_to_color_map.items():
        red, green, blue = mask_matrix.T  # Temporarily unpack the bands for readability

        target_areas = (red == color[0]) & (green == color[1]) & (blue == color[2])

        # print(f"target_areas: {np.sum(target_areas)}")
        mask_recolored[target_areas.T] = color_labels_to_indices[label]

    # print(f"image_uri: {image_uri}")
    # print(f"mask_uri: {mask_uri}")
    return image_uri, image_matrix, mask_recolored


def instance_generator(
    raw_instances: List[Dict], color_labels_to_indices: Dict[str, int]
) -> Dict:
    for raw_instance in raw_instances:
        image_uri, image_matrix, mask_recolored = convert_instance_to_features(
            raw_instance, color_labels_to_indices
        )
        yield {
            "file_name": image_uri,
            "image": image_matrix,
            "segmentation_mask": mask_recolored,
        }


# Define output signature for generator
output_signature = {
    "file_name": tf.TensorSpec(shape=(), dtype=tf.string, name=None),
    "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None),
    "segmentation_mask": tf.TensorSpec(
        shape=(None, None, 1), dtype=tf.uint8, name=None
    ),
}

dataset_train, dataset_validation = (
    tf.data.Dataset.from_generator(
        partial(instance_generator, instances, color_labels_to_indices),
        output_signature=output_signature,
    )
    for instances in [train_instances, validation_instances]
)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    # input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint["image"], (128, 128))
    input_mask = tf.image.resize(datapoint["segmentation_mask"], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


train_images, validation_images = (
    dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    for dataset in [dataset_train, dataset_validation]
)


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same randomn changes.
        self.augment_inputs = preprocessing.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = preprocessing.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


SHUFFLE_SEED = 42

train_batches = (
    train_images.cache()
    .shuffle(buffer_size=GLOBAL_BATCH_SIZE * 8, seed=SHUFFLE_SEED)
    .batch(GLOBAL_BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_batches = validation_images.batch(GLOBAL_BATCH_SIZE)


def unet_model(output_channels: int):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[128, 128, 3], include_top=False
    )

    # Use the activations of these layers
    layer_names = [
        "block_1_expand_relu",  # 64x64
        "block_3_expand_relu",  # 32x32
        "block_6_expand_relu",  # 16x16
        "block_13_expand_relu",  # 8x8
        "block_16_project",  # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding="same"
    )  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


TRAIN_LENGTH = len(train_instances)
STEPS_PER_EPOCH = TRAIN_LENGTH // GLOBAL_BATCH_SIZE

OUTPUT_CLASSES = len(color_labels) + 1  # Add one for background color of 0
VALIDATION_STEPS = len(validation_instances) // GLOBAL_BATCH_SIZE

with strategy.scope():
    model = unet_model(output_channels=OUTPUT_CLASSES)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

# Train the model
model_history = model.fit(
    train_batches,
    epochs=args.epochs,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    validation_data=validation_batches,
    callbacks=[],
)

model_dir = os.getenv("AIP_MODEL_DIR")

if model_dir:
    model.save(model_dir)
