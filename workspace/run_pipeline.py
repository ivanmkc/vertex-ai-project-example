from workspace.pipelines import pipelines
from pipelines_folder.pipeline import Pipeline
import workspace.util as util
import os


PROJECT = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

if not PROJECT:
    raise RuntimeError("Please set the GCP_PROJECT_ID environmental variable.")

if not BUCKET_NAME:
    raise RuntimeError("Please set the GCP_BUCKET_NAME environmental variable.")

pipeline_root = "{}/pipeline_root".format(BUCKET_NAME)

# TODO: Run in parallel
for pipeline in [
    pipelines.classification.custom_pipeline,
    # pipelines.object_detection.custom_pipeline,
    # pipelines.image_segmentation.custom_pipeline,
]:
    print(f"Running pipeline: {pipeline.name}")
    util.run_pipeline(
        project_id=PROJECT,
        location="us-central1",
        pipeline_root=pipeline_root,
        pipeline=pipeline,
    )
