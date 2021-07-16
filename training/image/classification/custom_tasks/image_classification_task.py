# Single, Mirror and Multi-Machine Distributed Training for CIFAR-10

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import os
import sys
import json
import tqdm
from typing import List

# TODO: Switch to arg
IMG_SIZE = 32


def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image


# Scaling image data from (0, 255] to (0., 1.]
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label


def load_aip_dataset(
    aip_data_uri_pattern: str,
    batch_size: int,
    class_names: List[str],
    test_run: bool,
    shuffle=True,
    repeat=False,
    seed=42,
):

    data_file_urls = list()
    labels = list()

    class_indices = dict(zip(class_names, range(len(class_names))))
    num_classes = len(class_names)

    for aip_data_uri in tqdm.tqdm(tf.io.gfile.glob(pattern=aip_data_uri_pattern)):
        with tf.io.gfile.GFile(name=aip_data_uri, mode="r") as gfile:
            for line in gfile.readlines():
                line = json.loads(line)
                data_file_urls.append(line["imageGcsUri"])
                classification_annotation = line["classificationAnnotations"][0]
                label = classification_annotation["displayName"]
                labels.append(class_indices[label])
                if test_run:
                    break

    filenames_ds = tf.data.Dataset.from_tensor_slices(data_file_urls)
    dataset = filenames_ds.map(
        parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    print(f" data files count: {len(data_file_urls)}")
    print(f" labels count: {len(labels)}")

    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    label_ds = label_ds.map(lambda x: tf.one_hot(x, num_classes))

    dataset = tf.data.Dataset.zip((dataset, label_ds)).map(scale).cache()

    if shuffle:
        # Shuffle locally at each iteration
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    # Users may need to reference `class_names`.
    dataset.class_names = class_names

    return dataset


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
    parser = argparse.ArgumentParser(description="Keras Image Classification")
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of training epochs"
    )
    parser.add_argument("--batch-size", default=16, type=int, help="mini-batch size")
    parser.add_argument(
        "--model-dir",
        default=os.getenv("AIP_MODEL_DIR"),
        type=str,
        help="model directory",
    )
    parser.add_argument("--data-dir", default="./data", type=str, help="data directory")
    parser.add_argument(
        "--test-run",
        default=False,
        type=str2bool,
        help="test run the training application, i.e. 1 epoch for training using sample dataset",
    )
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

class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
class_indices = dict(zip(class_names, range(len(class_names))))
num_classes = len(class_names)
print(f" class names: {class_names}")
print(f" class indices: {class_indices}")
print(f" num classes: {num_classes}")


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

print("Loading AIP dataset")
train_ds = load_aip_dataset(
    aip_training_data_uri,
    GLOBAL_BATCH_SIZE,
    class_names,
    args.test_run,
    shuffle=True,
    repeat=True,
)
print("AIP training dataset is loaded")
val_ds = load_aip_dataset(aip_validation_data_uri, 1, class_names, args.test_run)
print("AIP validation dataset is loaded")
# test_ds = load_aip_dataset(aip_test_data_uri, 1, class_names, args.test_run)
# print("AIP test dataset is loaded")

tfds.disable_progress_bar()

print("Python Version = {}".format(sys.version))
print("TensorFlow Version = {}".format(tf.__version__))
print("TF_CONFIG = {}".format(os.environ.get("TF_CONFIG", "Not found")))
print("DEVICES", device_lib.list_local_devices())

# Build the Keras model
def build_and_compile_cnn_model(num_classes: int, image_size: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, 3, activation="relu", input_shape=(image_size, image_size, 3)
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr),
        metrics=["accuracy"],
    )
    return model


# Train the model
model_dir = os.getenv("AIP_MODEL_DIR")

with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within
    # `strategy.scope()`.
    model = build_and_compile_cnn_model(num_classes=num_classes, image_size=IMG_SIZE)

model.fit(
    x=train_ds, epochs=args.epochs, validation_data=val_ds, steps_per_epoch=args.steps
)

if model_dir:
    model.save(model_dir)
