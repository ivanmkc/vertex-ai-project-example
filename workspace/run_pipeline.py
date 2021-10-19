from workspace.pipelines import pipelines
from pipelines_folder.pipeline import Pipeline
import workspace.util as util
import os
from typing import List
from google.cloud import aiplatform


PROJECT = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

if not PROJECT:
    raise RuntimeError("Please set the GCP_PROJECT_ID environmental variable.")

if not REGION:
    raise RuntimeError("Please set the GCP_REGION environmental variable.")

if not BUCKET_NAME:
    raise RuntimeError("Please set the GCP_BUCKET_NAME environmental variable.")

pipeline_root = "{}/pipeline_root".format(BUCKET_NAME)

# TODO: Run in parallel

pipeline_jobs: List[aiplatform.PipelineJob] = []

for pipeline in [
    pipelines.classification.custom_pipeline,
    # pipelines.classification.automl_pipeline,
    # pipelines.object_detection.automl_pipeline,
    # pipelines.image_segmentation.custom_pipeline,
]:
    print(f"Running pipeline: {pipeline.name}")
    job = util.create_pipeline_job(
        project_id=PROJECT,
        location=REGION,
        pipeline_root=pipeline_root,
        pipeline=pipeline,
    )

    job.run(sync=False)

    pipeline_jobs.append(job)

for job in pipeline_jobs:
    job.wait()
