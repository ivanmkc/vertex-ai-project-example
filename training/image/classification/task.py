import os
import argparse

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import json
import tqdm

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def build_model(num_classes, loss, optimizer, metrics, vectorize_layer):
    # vocab_size is VOCAB_SIZE + 1 since 0 is used additionally for padding.
    model = tf.keras.Sequential(
        [
            vectorize_layer,
            layers.Embedding(VOCAB_SIZE + 1, 64, mask_zero=True),
            layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
            layers.GlobalMaxPooling1D(),
            layers.Dense(num_classes),
            layers.Activation("softmax"),
        ]
    )
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def get_string_labels(predicted_scores_batch, class_names):
    predicted_labels = tf.argmax(predicted_scores_batch, axis=1)
    predicted_labels = tf.gather(class_names, predicted_labels)
    return predicted_labels


def predict(export_model, class_names, inputs):
    predicted_scores = export_model.predict(inputs)
    predicted_labels = get_string_labels(predicted_scores, class_names)
    return predicted_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keras Text Classification on Stack Overflow Questions"
    )
    parser.add_argument(
        "--epochs", default=25, type=int, help="number of training epochs"
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
    args = parser.parse_args()
    return args


def load_aip_dataset(
    aip_data_uri_pattern, batch_size, class_names, test_run, shuffle=True, seed=42
):

    data_file_urls = list()
    labels = list()

    class_indices = dict(zip(class_names, range(len(class_names))))
    num_classes = len(class_names)

    for aip_data_uri in tqdm.tqdm(tf.io.gfile.glob(pattern=aip_data_uri_pattern)):
        with tf.io.gfile.GFile(name=aip_data_uri, mode="r") as gfile:
            for line in gfile.readlines():
                line = json.loads(line)
                data_file_urls.append(line["textContent"])
                classification_annotation = line["classificationAnnotations"][0]
                label = classification_annotation["displayName"]
                labels.append(class_indices[label])
                if test_run:
                    break

    data = list()
    for data_file_url in tqdm.tqdm(data_file_urls):
        with tf.io.gfile.GFile(name=data_file_url, mode="r") as gf:
            txt = gf.read()
            data.append(txt)

    print(f" data files count: {len(data_file_urls)}")
    print(f" data count: {len(data)}")
    print(f" labels count: {len(labels)}")

    dataset = tf.data.Dataset.from_tensor_slices(data)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    label_ds = label_ds.map(lambda x: tf.one_hot(x, num_classes))

    dataset = tf.data.Dataset.zip((dataset, label_ds))

    if shuffle:
        # Shuffle locally at each iteration
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    dataset = dataset.batch(batch_size)
    # Users may need to reference `class_names`.
    dataset.class_names = class_names

    return dataset


def main():

    args = parse_args()

    class_names = ["csharp", "java", "javascript", "python"]
    class_indices = dict(zip(class_names, range(len(class_names))))
    num_classes = len(class_names)
    print(f" class names: {class_names}")
    print(f" class indices: {class_indices}")
    print(f" num classes: {num_classes}")

    epochs = 1 if args.test_run else args.epochs

    aip_model_dir = os.environ.get("AIP_MODEL_DIR")
    aip_data_format = os.environ.get("AIP_DATA_FORMAT")
    aip_training_data_uri = os.environ.get("AIP_TRAINING_DATA_URI")
    aip_validation_data_uri = os.environ.get("AIP_VALIDATION_DATA_URI")
    aip_test_data_uri = os.environ.get("AIP_TEST_DATA_URI")

    print(f"aip_model_dir: {aip_model_dir}")
    print(f"aip_data_format: {aip_data_format}")
    print(f"aip_training_data_uri: {aip_training_data_uri}")
    print(f"aip_validation_data_uri: {aip_validation_data_uri}")
    print(f"aip_test_data_uri: {aip_test_data_uri}")

    print("Loading AIP dataset")
    train_ds = load_aip_dataset(
        aip_training_data_uri, args.batch_size, class_names, args.test_run
    )
    print("AIP training dataset is loaded")
    val_ds = load_aip_dataset(aip_validation_data_uri, 1, class_names, args.test_run)
    print("AIP validation dataset is loaded")
    test_ds = load_aip_dataset(aip_test_data_uri, 1, class_names, args.test_run)
    print("AIP test dataset is loaded")

    vectorize_layer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    train_text = train_ds.map(lambda text, labels: text)
    vectorize_layer.adapt(train_text)
    print("The vectorize_layer is adapted")

    print("Build model")
    optimizer = "adam"
    metrics = ["accuracy"]

    model = build_model(
        num_classes,
        losses.CategoricalCrossentropy(from_logits=True),
        optimizer,
        metrics,
        vectorize_layer,
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    history = history.history

    print(
        "Training accuracy: {acc}, loss: {loss}".format(
            acc=history["accuracy"][-1], loss=history["loss"][-1]
        )
    )
    print(
        "Validation accuracy: {acc}, loss: {loss}".format(
            acc=history["val_accuracy"][-1], loss=history["val_loss"][-1]
        )
    )

    loss, accuracy = model.evaluate(test_ds)
    print("Test accuracy: {acc}, loss: {loss}".format(acc=accuracy, loss=loss))

    inputs = [
        "how do I extract keys from a dict into a list?",  # python
        "debug public static void main(string[] args) {...}",  # java
    ]
    predicted_labels = predict(model, class_names, inputs)
    for input, label in zip(inputs, predicted_labels):
        print(f"Question: {input}")
        print(f"Predicted label: {label.numpy()}")

    model_export_path = os.path.join(args.model_dir, str(args.model_version))
    model.save(model_export_path)
    print(f"Model version {args.model_version} is exported to {args.model_dir}")

    loaded = tf.saved_model.load(model_export_path)
    input_name = list(
        loaded.signatures["serving_default"].structured_input_signature[1].keys()
    )[0]
    print(f"Serving function input: {input_name}")

    return


if __name__ == "__main__":
    main()
