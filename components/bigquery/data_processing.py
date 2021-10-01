from typing import Dict, List
from kfp.v2.dsl import component
from typing import NamedTuple, Optional


@component(
    packages_to_install=[
        "google-cloud-bigquery[all]",
        "protobuf",
        "google-cloud-pipeline-components",
    ]
)
def bq_query(
    project: str,
    location: str,
    query: str,
    # destination_table: Output[BQTable],
    destination_table_id: Optional[str] = None,
) -> NamedTuple("Outputs", [("gcp_resources", str), ("destination_table_id", str)]):
    """Query

    https://cloud.google.com/bigquery/docs/writing-results?hl=en
    """

    from google.cloud import bigquery
    from google_cloud_pipeline_components.experimental.proto.gcp_resources_pb2 import (
        GcpResources,
    )
    from google.protobuf import json_format

    client = bigquery.Client(project=project, location=location)

    job_config = None
    if destination_table_id:
        job_config = bigquery.QueryJobConfig(destination=destination_table_id)

    query_job = client.query(query, job_config=job_config)  # API request
    query_job.result()  # Waits for query to finish

    # Extract destination table
    destination = query_job.destination
    output_destination_table_id = (
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
    return output(query_job_resources_serialized, output_destination_table_id)


@component(
    packages_to_install=[
        "google-cloud-bigquery[all]",
        "protobuf",
        "google-cloud-pipeline-components",
    ]
)
def export(
    project: str,
    location: str,
    source: str,
    destination_uri: str,
    destination_format: Optional[str] = None,
    field_delimiter: Optional[str] = None,
    compression: Optional[str] = None,
    print_header: Optional[bool] = None,
    use_avro_logical_types: Optional[str] = None,
) -> NamedTuple("Outputs", [("gcp_resources", str), ("destination_uri", str)]):
    """Export BigQuery table to CSV at GCS destination

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-predict
    """

    from google.cloud import bigquery
    from google_cloud_pipeline_components.experimental.proto.gcp_resources_pb2 import (
        GcpResources,
    )
    from google.protobuf import json_format

    client = bigquery.Client(project=project, location=location)

    job_config = bigquery.ExtractJobConfig(
        compression=compression,
        destination_format=destination_format,
        field_delimiter=field_delimiter,
        print_header=print_header,
        use_avro_logical_types=use_avro_logical_types,
    )

    extract_job = client.extract_table(
        source=source,
        destination_uris=destination_uri,
        location=location,
        job_config=job_config,
    )  # API request
    extract_job.result()  # Waits for job to complete.

    # Instantiate GCPResources Proto
    query_job_resources = GcpResources()
    query_job_resource = query_job_resources.resources.add()

    # Write the job proto to output
    query_job_resource.resource_type = "BigQueryJob"
    query_job_resource.resource_uri = extract_job.self_link

    query_job_resources_serialized = json_format.MessageToJson(query_job_resources)

    from collections import namedtuple

    output = namedtuple("Outputs", ["gcp_resources", "destination_uri"])
    return output(query_job_resources_serialized, destination_uri)
