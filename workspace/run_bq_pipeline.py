from workspace.pipelines import pipelines
import workspace.util as util

JOB_SPEC_PATH = "package.json"


BUCKET_NAME = "gs://ivanmkc-test2/pipeline_staging"
pipeline_root = "{}/pipeline_root".format(BUCKET_NAME)

# TODO: Run in parallel
for pipeline in [
    # pipelines.tabular.bqml_custom_predict,
    pipelines.tabular.bq_automl,
    # pipelines.tabular.bq_custom,
    # pipelines.tabular.bqml_export_vertexai,
]:
    print(f"Running pipeline: {pipeline.name}")
    util.run_pipeline(
        project_id="python-docs-samples-tests",
        location="us-central1",
        pipeline_root=pipeline_root,
        pipeline=pipeline,
    )
