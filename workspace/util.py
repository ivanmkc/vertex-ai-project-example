import datetime
from google.cloud import aiplatform
from pipelines_folder.pipeline import Pipeline
from kfp.v2 import compiler
import re

JOB_SPEC_PATH = "package.json"


def generate_pipeline_job_id(pipeline_name: str) -> str:
    current_time = datetime.datetime.now()

    return "{pipeline_name}-{timestamp}".format(
        pipeline_name=re.sub("[^-0-9a-z]+", "-", pipeline_name.lower())
        .lstrip("-")
        .rstrip("-"),
        timestamp=current_time.strftime("%Y%m%d%H%M%S"),
    )


def run_pipeline(
    project_id: str,
    location: str,
    pipeline_root: str,
    pipeline: Pipeline,
):
    pipeline_run_name = generate_pipeline_job_id(pipeline.name)

    compiler.Compiler().compile(
        pipeline_func=pipeline.create_pipeline(
            project=project_id,
            pipeline_root=pipeline_root,
            pipeline_run_name=pipeline_run_name,
            location=location,
        ),
        package_path=JOB_SPEC_PATH,
    )

    job = aiplatform.PipelineJob(
        display_name=pipeline.name,
        template_path=JOB_SPEC_PATH,
        job_id=pipeline_run_name,
        pipeline_root=pipeline_root,
        # parameter_values={"project": project_id, "display_name": pipeline.name},
    )

    job.run(sync=True)
