from workspace.pipelines import pipelines

from google.cloud import aiplatform
from pipelines_folder.pipeline import Pipeline
from kfp.v2 import compiler


JOB_SPEC_PATH = "package.json"


def run_pipeline(
    project_id: str,
    location: str,
    pipeline_root: str,
    pipeline: Pipeline,
):
    compiler.Compiler().compile(
        pipeline_func=pipeline.create_pipeline(
            project=project_id, pipeline_root=pipeline_root, location=location
        ),
        package_path=JOB_SPEC_PATH,
    )

    job = aiplatform.PipelineJob(
        display_name=pipeline.name,
        template_path=JOB_SPEC_PATH,
        pipeline_root=pipeline_root,
        # parameter_values={"project": project_id, "display_name": pipeline.name},
    )

    job.run()


BUCKET_NAME = "gs://ivanmkc-test2/pipeline_staging"
pipeline_root = "{}/pipeline_root".format(BUCKET_NAME)

# TODO: Run in parallel
for pipeline in [
    # pipelines.tabular.bqml_custom_predict,
    pipelines.tabular.bq_automl,
]:
    print(f"Running pipeline: {pipeline.name}")
    run_pipeline(
        project_id="python-docs-samples-tests",
        location="us-east1",
        pipeline_root=pipeline_root,
        pipeline=pipeline,
    )
