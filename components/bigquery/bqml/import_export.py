from kfp.v2.dsl import component
from typing import Optional, NamedTuple


@component(
    packages_to_install=[
        "google-cloud-bigquery[all]",
        "protobuf",
        "google-cloud-pipeline-components",
    ]
)
def bqml_import_model(
    project: str,
    location: str,
    model_source_path: str,
    should_replace: bool,
) -> NamedTuple("Outputs", [("gcp_resources", str), ("model_destination", str)]):
    """Import model

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-tensorflow
    """

    from google.cloud import bigquery
    from google_cloud_pipeline_components.experimental.proto.gcp_resources_pb2 import (
        GcpResources,
    )
    from google.protobuf import json_format

    # Build query
    def build_query(model_source_path: str, should_replace: bool) -> str:
        create_statement = "CREATE MODEL"

        if should_replace:
            create_statement = "CREATE OR REPLACE MODEL"

        return f"{create_statement} OPTIONS(MODEL_TYPE = 'TENSORFLOW', MODEL_PATH = {model_source_path})"

    query = build_query(
        model_source_path=model_source_path,
        should_replace=should_replace,
    )

    client = bigquery.Client(project=project, location=location)

    query_job = client.query(query)
    _ = query_job.result()

    # Retrieve model name and model
    table: bigquery.table.TableReference = query_job.ddl_target_table
    model_name = f"{table.project}.{table.dataset_id}.{table.table_id}"
    # model = client.get_model(model_name)

    # Instantiate GCPResources Proto
    query_job_resources = GcpResources()
    query_job_resource = query_job_resources.resources.add()

    # Write the job proto to output
    query_job_resource.resource_type = "BigQueryJob"
    query_job_resource.resource_uri = query_job.self_link

    query_job_resources_serialized = json_format.MessageToJson(query_job_resources)

    from collections import namedtuple

    output = namedtuple("Outputs", ["gcp_resources", "model_name"])
    return output(query_job_resources_serialized, model_name)


@component(
    packages_to_install=[
        "google-cloud-bigquery[all]",
        "protobuf",
        "google-cloud-pipeline-components",
    ]
)
def bqml_export_model(
    project: str,
    location: str,
    model: str,  # TODO: Change to Input[BQMLModel
    model_destination: str,
    trial_id: Optional[str],
) -> NamedTuple("Outputs", [("gcp_resources", str), ("model_destination", str)]):
    """Export model

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-export-model
    """

    from google.cloud import bigquery
    from google_cloud_pipeline_components.experimental.proto.gcp_resources_pb2 import (
        GcpResources,
    )
    from google.protobuf import json_format

    # Build query
    def build_query(
        model_name: str, model_destination: str, trial_id: Optional[str]
    ) -> str:
        if trial_id:
            return f"EXPORT MODEL `{model_name}` OPTIONS(URI = {model_destination}, TRIAL_ID = {trial_id})"
        else:
            return f"EXPORT MODEL `{model_name}` OPTIONS(URI = {model_destination})"

    query = build_query(
        model_name=model,
        model_destination=model_destination,
        trial_id=trial_id,
    )

    client = bigquery.Client(project=project, location=location)

    query_job = client.query(query)
    _ = query_job.result()  # Waits for query to finish

    # Instantiate GCPResources Proto
    query_job_resources = GcpResources()
    query_job_resource = query_job_resources.resources.add()

    # Write the job proto to output
    query_job_resource.resource_type = "BigQueryJob"
    query_job_resource.resource_uri = query_job.self_link

    query_job_resources_serialized = json_format.MessageToJson(query_job_resources)

    from collections import namedtuple

    output = namedtuple("Outputs", ["gcp_resources", "model_destination"])
    return output(query_job_resources_serialized, model_destination)
