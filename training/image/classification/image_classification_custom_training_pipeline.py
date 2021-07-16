from training.image.classification.image_classification_training_pipeline import (
    ImageClassificationManagedDatasetPipeline,
)
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import Callable
from kfp.v2.dsl import component, Dataset, Input, Output, OutputPath, Model


class ImageClassificationCustomManagedDatasetPipeline(
    ImageClassificationManagedDatasetPipeline
):
    id = "Image Classification Custom"
    managed_dataset_uri: str = "aiplatform://v1/projects/1012616486416/locations/us-central1/datasets/7601275726536376320"

    # @component
    # def training_op(
    #     dataset: Input[Dataset],
    #     model: Output[Model],
    # ):
    #     print("training task: {}".format(dataset.id))

    @component(packages_to_install=["google-cloud-storage", "google-cloud-aiplatform"])
    def train(
        dataset: Input[Dataset],
        model: Output[Model],
        artifact_uri: OutputPath(str),
        pipeline_root: str,
        experiment_name: str,
        run_name: str,
        num_epochs: int,
    ):
        import json
        from google.cloud import storage
        from google.cloud.aiplatform.datasets import ImageDataset

        def download_blob(bucket_name, source_blob_name, destination_file_name):
            """Downloads a blob from the bucket."""
            # bucket_name = "your-bucket-name"
            # source_blob_name = "storage-object-name"
            # destination_file_name = "local/path/to/file"

            storage_client = storage.Client()

            bucket = storage_client.bucket(bucket_name)

            # Construct a client side representation of a blob.
            # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
            # any content from Google Cloud Storage. As we don't need additional data,
            # using `Bucket.blob` is preferred here.
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)

            print(
                "Blob {} downloaded to {}.".format(
                    source_blob_name, destination_file_name
                )
            )

        def extract_bucket_and_prefix_from_gcs_path(gcs_path: str):
            """Given a complete GCS path, return the bucket name and prefix as a tuple.

            Example Usage:

                bucket, prefix = extract_bucket_and_prefix_from_gcs_path(
                    "gs://example-bucket/path/to/folder"
                )

                # bucket = "example-bucket"
                # prefix = "path/to/folder"

            Args:
                gcs_path (str):
                    Required. A full path to a Google Cloud Storage folder or resource.
                    Can optionally include "gs://" prefix or end in a trailing slash "/".

            Returns:
                Tuple[str, Optional[str]]
                    A (bucket, prefix) pair from provided GCS path. If a prefix is not
                    present, a None will be returned in its place.
            """
            if gcs_path.startswith("gs://"):
                gcs_path = gcs_path[5:]
            if gcs_path.endswith("/"):
                gcs_path = gcs_path[:-1]

            gcs_parts = gcs_path.split("/", 1)
            gcs_bucket = gcs_parts[0]
            gcs_blob_prefix = None if len(gcs_parts) == 1 else gcs_parts[1]

            return (gcs_bucket, gcs_blob_prefix)

        dataset_output_dir = f"{pipeline_root}/dataset_output"
        print(f"dataset.__dict__: {dataset.__dict__}")
        print(f"dataset.path: {dataset.path}")
        print(f"dataset.__dict__['name']: {dataset.__dict__['name']}")
        print(f"dataset.name: {dataset.name}")
        print(f"dataset.uri: {dataset.uri}")

        managed_dataset = ImageDataset(
            dataset_name=dataset.uri.replace("aiplatform://v1/", "")
        )

        print(f"Exporting to: {dataset_output_dir}")
        managed_dataset.export_data(output_dir=dataset_output_dir)

        bucket, file_path = extract_bucket_and_prefix_from_gcs_path(dataset_output_dir)
        download_blob(
            bucket_name=bucket,
            source_blob_name=file_path,
            destination_file_name=file_path,
        )

        with open(file_path, "r") as file:
            contents = json.loads(file.read())
            print(f"contents: {contents}")

        # from google.cloud import aiplatform
        # from typing import NamedTuple

        # aiplatform.init(
        #     project="kubeflow-demos",
        #     location="us-central1",
        #     staging_bucket="gs://user-group-demo/pipeline_root",
        #     experiment=experiment_name,
        # )

        # aiplatform.start_run(run_name)
        # parameters = {"epochs": num_epochs}
        # aiplatform.log_params(parameters)

        # 1
        # import pandas as pd
        # import os
        # from imblearn.under_sampling import RandomUnderSampler

        # training_data_uri = os.getenv("AIP_TRAINING_DATA_URI")
        # validation_data_uri = os.getenv("AIP_VALIDATION_DATA_URI")
        # testing_data_uri = os.getenv("AIP_TEST_DATA_URI")

        # print(f"training_data_uri: {training_data_uri}")
        # print(f"validation_data_uri: {validation_data_uri}")
        # print(f"testing_data_uri: {testing_data_uri}")

        # print(f"dataset: {dataset}")
        # print(f"dataset.metadata: {dataset.metadata}")

        # df_review = pd.read_csv("")
        # print(len(df_review))

        # df_positive = df_review[df_review["sentiment"] == "positive"][:9000]
        # df_negative = df_review[df_review["sentiment"] == "negative"][:1000]

        # df_review_imb = pd.concat([df_positive, df_negative])
        # df_review_imb.value_counts(["sentiment"])

        # print(len(df_review_imb))
        # rus = RandomUnderSampler(random_state=0)
        # df_review_bal, df_review_bal["sentiment"] = rus.fit_resample(
        #     df_review_imb[["review"]], df_review_imb["sentiment"]
        # )

        # print(len(df_review_bal))
        # from sklearn.model_selection import train_test_split

        # train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)
        # train_x, train_y = train["review"], train["sentiment"]
        # test_x, test_y = test["review"], test["sentiment"]

        # print("train x values count")
        # print(len(train_x))
        # print("train y values count")
        # print(train_y.value_counts())

        # from sklearn.feature_extraction.text import TfidfVectorizer

        # tfidf = TfidfVectorizer(stop_words="english")
        # train_x_vector = tfidf.fit_transform(train_x)
        # test_x_vector = tfidf.transform(test_x)

        # print(train_x_vector)
        # from sklearn.svm import SVC

        # svc = SVC(kernel="linear")
        # svc.fit(train_x_vector, train_y)

        # print(svc.score(test_x_vector, test_y))

        # # aiplatform.log_metrics({"accuracy": accu})
        # import joblib

        # joblib.dump(
        #     svc,
        #     os.path.join(model.path.replace("saved_model", ""), "model.joblib"),
        # )
        # print(" saved_model.path: " + model.path)
        # print(" saved_model.uri: " + model.uri)
        # with open(artifact_uri, "w") as f:
        #     f.write(model.uri.replace("saved_model", ""))

        # print(model.uri)

    def create_training_op(
        self, project: str, pipeline_root: str, dataset: Dataset
    ) -> Callable:
        from datetime import datetime

        TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
        return ImageClassificationCustomManagedDatasetPipeline.train(
            dataset,
            pipeline_root=pipeline_root,
            experiment_name="test-" + TIMESTAMP,
            run_name="test-run-" + TIMESTAMP,
            num_epochs=1,
        )

        # train_task = ImageClassificationCustomManagedDatasetPipeline.training_op(
        #     dataset=dataset
        # )

        # hp_dict: str = '{"num_hidden_layers": 3, "hidden_size": 32, "learning_rate": 0.01, "epochs": 1, "steps_per_epoch": -1}'
        # data_dir: str = "gs://aju-dev-demos-codelabs/bikes_weather/"
        # TRAINER_ARGS = ["--data-dir", data_dir, "--hptune-dict", hp_dict]

        # # create working dir to pass to job spec
        # import time

        # ts = int(time.time())
        # WORKING_DIR = f"{pipeline_root}/{ts}"

        # experimental.run_as_aiplatform_custom_job(
        #     train_task,
        #     worker_pool_specs=[
        #         {
        #             "containerSpec": {
        #                 "args": TRAINER_ARGS,
        #                 "env": [{"name": "AIP_MODEL_DIR", "value": WORKING_DIR}],
        #                 "imageUri": "gcr.io/google-samples/bw-cc-train:latest",
        #             },
        #             "replicaCount": "1",
        #             "machineSpec": {
        #                 "machineType": "n1-standard-16",
        #                 "accelerator_type": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_K80,
        #                 "accelerator_count": 2,
        #             },
        #         }
        #     ],
        # )

        # return train_task

    # return gcc_aip.CustomPythonPackageTrainingJobRunOp
    # return gcc_aip.AutoMLImageTrainingJobRunOp(
    #     project=project,
    #     display_name="train-iris-automl-mbsdk-custom",
    #     prediction_type="classification",
    #     model_type="CLOUD",
    #     base_model=None,
    #     dataset=dataset,
    #     model_display_name="iris-classification-model-mbsdk",
    #     training_fraction_split=0.6,
    #     validation_fraction_split=0.2,
    #     test_fraction_split=0.2,
    #     budget_milli_node_hours=8000,
    # )
