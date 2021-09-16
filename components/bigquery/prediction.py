from kfp.v2.dsl import component
from typing import Optional, NamedTuple


@component(
    packages_to_install=[
        "google-cloud-bigquery[all]",
        "protobuf",
        "google-cloud-pipeline-components",
    ]
)
def bqml_predict(
    project: str,
    location: str,
    model: str,  # TODO: Change to Input[BQMLModel
    query_statement: Optional[str] = None,
    table_name: Optional[str] = None,
    threshold: Optional[float] = None,
    keep_original_columns: Optional[bool] = None,
    destination_table_id: Optional[str] = None,  # TODO: Change to Output[BQTable],
) -> NamedTuple("Outputs", [("gcp_resources", str), ("destination_table_id", str)]):
    """Get prediction

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-predict
    """

    from google.cloud import bigquery
    from google_cloud_pipeline_components.experimental.proto.gcp_resources_pb2 import (
        GcpResources,
    )
    from google.protobuf import json_format

    # Build query
    def build_query(
        model_name: str,
        table_name: Optional[str] = None,
        query_statement: Optional[str] = None,
        threshold: Optional[float] = None,
        keep_original_columns: Optional[bool] = None,
    ) -> str:
        parameters = [
            f"MODEL `{model_name}`",
        ]

        if table_name and query_statement:
            raise ValueError(
                "Only one of 'table_name' or 'query_statement' can be set, but not both."
            )
        elif all(x is None for x in [table_name, query_statement]):
            raise ValueError("One of 'table_name' or 'query_statement' must be set.")
        elif table_name:
            parameters.append(f"TABLE `{table_name}`")
        elif query_statement:
            parameters.append(f"({query_statement})")

        settings = []
        if threshold is not None:
            settings.append(f"{threshold} AS threshold")

        if keep_original_columns is not None:
            settings.append(
                f"{'TRUE' if keep_original_columns else 'FALSE'} AS keep_original_columns"
            )

        if len(settings) > 0:
            parameters.append(f"STRUCT({', '.join(settings)})")

        return f"SELECT * FROM ML.PREDICT({', '.join(parameters)})"

    query = build_query(
        model_name=model,
        table_name=table_name,
        query_statement=query_statement,
        threshold=threshold,
        keep_original_columns=keep_original_columns,
    )

    client = bigquery.Client(project=project, location=location)

    job_config = None
    if destination_table_id:
        job_config = bigquery.QueryJobConfig(destination=destination_table_id)

    query_job = client.query(query, job_config=job_config)  # API request

    _ = query_job.result()  # Waits for query to finish

    destination = query_job.destination

    destination_table_id = (
        f"{destination.project}.{destination.dataset_id}.{destination.table_id}"
    )

    # Instantiate GCPResources Proto
    query_job_resources = GcpResources()
    query_job_resource = query_job_resources.resources.add()

    # Write the job proto to output
    query_job_resource.resource_type = "BigQueryJob"
    query_job_resource.resource_uri = query_job.self_link

    query_job_resources_serialized = json_format.MessageToJson(query_job_resources)

    from collections import namedtuple

    output = namedtuple("Outputs", ["gcp_resources", "destination_table_id"])
    return output(query_job_resources_serialized, destination_table_id)
