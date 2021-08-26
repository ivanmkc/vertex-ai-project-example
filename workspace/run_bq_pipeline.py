from workspace.pipelines import pipelines

from google.cloud import aiplatform
from pipelines_folder.pipeline import Pipeline
from kfp.v2 import compiler  # noqa: F811
from kfp.v2.google.client import AIPlatformClient  # noqa: F811
from typing import List, Set

from training.tabular.bq_training_pipelines import (
    BQMLTrainingPipeline,
    BQQueryAutoMLPipeline,
)


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


BUCKET_NAME = "gs://ivanmkc-test2/pipeline_staging"
pipeline_root = "{}/pipeline_root".format(BUCKET_NAME)

# Perform a query.
query = (
    "SELECT name FROM `bigquery-public-data.usa_names.usa_1910_2013` "
    'WHERE state = "TX" '
    "LIMIT 100"
)

for pipeline in [
    # BQMLTrainingPipeline(name="bqml-training"),
    BQQueryAutoMLPipeline(
        "bq-automl",
        query=query,
        bq_output_table_id="python-docs-samples-tests.ivan_test.output",
    ),
]:
    print(f"Running pipeline: {pipeline.name}")
    run_pipeline(
        project_id="python-docs-samples-tests",
        region="us-central1",
        pipeline_root=pipeline_root,
        pipeline=pipeline,
    )
