from training.image.classification.image_classification_automl_training_pipeline import (
    ImageClassificationAutoMLManagedDatasetPipeline,
)
from training.image.classification.image_classification_custom_training_pipeline import (
    ImageClassificationCustomManagedDatasetPipeline,
)

from training.image.classification.image_classification_custom_python_package_training_pipeline import (
    ImageClassificationCustomPythonPackageManagedDatasetPipeline,
)

from training.image.classification.image_classification_automl_training_pipeline import (
    ImageClassificationAutoMLManagedDatasetPipeline,
)

from training.image.object_detection.object_detection_automl_training_pipeline import (
    ObjectDetectionAutoMLManagedDatasetPipeline,
)

from google.cloud import aiplatform
from pipeline import Pipeline
from kfp.v2 import compiler  # noqa: F811
from kfp.v2.google.client import AIPlatformClient  # noqa: F811
from typing import List, Set


def check_if_dataset_changed(
    dataset_id: str,
) -> bool:
    # TODO: Check if dataset_id was already trained
    # Perhaps modify the dataset metadata somehow
    return True


def get_changed_datasets() -> Set[str]:
    # TODO: Get all datasets

    return []


def retrain_for_changed_datasets(project_id: str, region: str):
    # Get last modified dates
    datasets = aiplatform.Dataset.list()

    # Get all datasets
    changed_datasets = [
        dataset
        for dataset in datasets
        if check_if_dataset_changed(dataset_id=dataset.id)
    ]

    # Get all pipelines for changed datasets
    all_pipelines: List[Pipeline] = []  # TODO

    changed_dataset_uris = set([changed_datasets.uri for dataset in changed_datasets])

    # Run all pipelines that use a modified dataseta
    for pipeline in all_pipelines:
        if pipeline.managed_dataset_uri in changed_dataset_uris:
            run_pipeline(
                project_id=project_id,
                region=region,
                pipeline_root=pipeline_root,
                pipeline=pipeline,
            )


JOB_SPEC_PATH = "package.json"


def run_pipeline(
    project_id: str,
    region: str,
    pipeline_root: str,
    pipeline: Pipeline,
):
    compiler.Compiler().compile(
        pipeline_func=pipeline.create_pipeline(
            project=project_id, pipeline_root=pipeline_root
        ),
        package_path=JOB_SPEC_PATH,
    )

    api_client = AIPlatformClient(project_id=project_id, region=region)

    # TODO: Replace with something that blocks and throws an error
    response = api_client.create_run_from_job_spec(
        JOB_SPEC_PATH,
        pipeline_root=pipeline_root,
        parameter_values={},
    )

    # print(response)


BUCKET_NAME = "gs://ivanmkc-mineral/training"
pipeline_root = "{}/pipeline_root".format(BUCKET_NAME)

run_pipeline(
    project_id="python-docs-samples-tests",
    region="us-central1",
    pipeline_root=pipeline_root,
    pipeline=ObjectDetectionAutoMLManagedDatasetPipeline(),
)
