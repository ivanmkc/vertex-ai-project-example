from workspace.pipelines import pipelines

from google.cloud import aiplatform
from pipelines_folder.pipeline import Pipeline
from kfp.v2 import compiler
import os

JOB_SPEC_PATH = "package.json"


def run_pipeline(
    project_id: str,
    location: str,
    pipeline_root: str,
    pipeline: Pipeline,
):
    compiler.Compiler().compile(
        pipeline_func=pipeline.create_pipeline(
            project=project_id,
            location=location,
            pipeline_root=pipeline_root,
        ),
        package_path=JOB_SPEC_PATH,
    )

    job = aiplatform.PipelineJob(
        display_name=pipeline.name,
        template_path=JOB_SPEC_PATH,
        pipeline_root=pipeline_root,
    )

    job.run()


PROJECT = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

if not PROJECT:
    raise RuntimeError("Please set the GCP_PROJECT_ID environmental variable.")

if not BUCKET_NAME:
    raise RuntimeError("Please set the GCP_BUCKET_NAME environmental variable.")

pipeline_root = "{}/pipeline_root".format(BUCKET_NAME)

for pipeline in [
    # pipelines.classification.custom_pipeline,
    pipelines.object_detection.custom_pipeline,
    # pipelines.image_segmentation.custom_pipeline,
    # pipelines.object_detection.custom_pipeline,
]:
    print(f"Running pipeline: {pipeline.name}")
    run_pipeline(
        project_id=PROJECT,
        location="us-central1",
        pipeline_root=pipeline_root,
        pipeline=pipeline,
    )
