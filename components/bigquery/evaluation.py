from kfp.v2.dsl import (
    Output,
    component,
    ClassificationMetrics,
    Metrics,
)
from typing import List, NamedTuple, Optional
from kfp.v2.components.types.artifact_types import Artifact


@component(
    packages_to_install=[
        "google-cloud-bigquery[all]",
        "protobuf",
        "google-cloud-pipeline-components",
    ]
)
def bqml_create_evaluation_op(
    project: str,
    location: str,
    model: str,  # TODO: Change to Input[BQMLModel]
    metrics: Output[Metrics],
    query_statement: Optional[str] = None,
    table_name: Optional[str] = None,
    thresholds: Optional[float] = None,
) -> NamedTuple("Outputs", [("gcp_resources", str)]):
    """Get the evaluation curve

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-evaluate
    """
    import collections
    from google.cloud import bigquery
    from google_cloud_pipeline_components.experimental.proto.gcp_resources_pb2 import (
        GcpResources,
    )
    from google.protobuf import json_format

    # Build query
    def build_query(
        action_name: str,
        model_name: str,
        table_name: Optional[str],
        query_statement: Optional[str],
        thresholds: List[float],
    ) -> str:
        parameters = [
            f"MODEL `{model_name}`",
        ]

        if table_name and query_statement:
            raise ValueError(
                "Only one of 'table_name' or 'query_statement' can be set, but not both."
            )
        elif table_name:
            parameters.append(f"TABLE `{table_name}`")
        elif query_statement:
            parameters.append(f"({query_statement})")

        if thresholds:
            parameters.append(f"GENERATE_ARRAY({', '.join(thresholds)})")

        return f"SELECT * FROM ML.{action_name}({', '.join(parameters)})"

    def log_evaluations(
        model_name: str,
        table_name: Optional[str],
        query_statement: Optional[str],
        thresholds: List[float],
    ) -> GcpResources:
        query = build_query(
            action_name="EVALUATE",
            model_name=model_name,
            table_name=table_name,
            query_statement=query_statement,
            thresholds=thresholds,
        )

        client = bigquery.Client(project=project, location=location)

        query_job = client.query(query)  # API request

        df = query_job.to_dataframe()  # Waits for query to finish

        # Log matrix
        for name, value in df.to_dict(orient="list").items():
            if not isinstance(value, collections.Sequence):
                metrics.log_metric(name, value)
            else:
                metrics.log_metric(name, value[-1])

        # Instantiate GCPResources Proto
        query_job_resources = GcpResources()
        query_job_resource = query_job_resources.resources.add()

        # Write the job proto to output
        query_job_resource.resource_type = "BigQueryJob"
        query_job_resource.resource_uri = query_job.self_link
        return query_job_resources

    query_job_resources = log_evaluations(
        model_name=model,
        table_name=table_name,
        query_statement=query_statement,
        thresholds=thresholds,
    )

    query_job_resources_serialized = json_format.MessageToJson(query_job_resources)

    from collections import namedtuple

    output = namedtuple("Outputs", ["gcp_resources"])
    return output(query_job_resources_serialized)


@component(
    packages_to_install=[
        "google-cloud-bigquery[all]",
        "protobuf",
        "google-cloud-pipeline-components",
    ]
)
def create_confusion_matrix(
    project: str,
    location: str,
    model: str,  # TODO: Change to Input[BQMLModel]
    classification_metrics: Output[ClassificationMetrics],
    query_statement: Optional[str] = None,
    table_name: Optional[str] = None,
    thresholds: Optional[float] = None,
) -> NamedTuple("Outputs", [("gcp_resources", str)]):
    """Get the confusion matrix

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-confusion
    """

    from google.cloud import bigquery
    from google_cloud_pipeline_components.experimental.proto.gcp_resources_pb2 import (
        GcpResources,
    )
    from google.protobuf import json_format

    # Build query
    def build_query(
        action_name: str,
        model_name: str,
        table_name: Optional[str],
        query_statement: Optional[str],
        thresholds: List[float],
    ) -> str:
        parameters = [
            f"MODEL `{model_name}`",
        ]

        if table_name and query_statement:
            raise ValueError(
                "Only one of 'table_name' or 'query_statement' can be set, but not both."
            )
        elif table_name:
            parameters.append(f"TABLE `{table_name}`")
        elif query_statement:
            parameters.append(f"({query_statement})")

        if thresholds:
            parameters.append(f"GENERATE_ARRAY({', '.join(thresholds)})")

        return f"SELECT * FROM ML.{action_name}({', '.join(parameters)})"

    def log_confusion_matrix(
        model_name: str,
        table_name: Optional[str],
        query_statement: Optional[str],
        thresholds: List[float],
    ) -> GcpResources:
        query = build_query(
            action_name="CONFUSION_MATRIX",
            model_name=model_name,
            table_name=table_name,
            query_statement=query_statement,
            thresholds=thresholds,
        )

        client = bigquery.Client(project=project, location=location)

        query_job = client.query(query)  # API request

        df = query_job.to_dataframe()  # Waits for query to finish
        df = df.drop("expected_label", 1)

        categories = [column for column in df.columns]
        matrix = df.values.tolist()
        print(matrix)

        classification_metrics.log_confusion_matrix(
            categories=categories, matrix=matrix
        )

        # Instantiate GCPResources Proto
        query_job_resources = GcpResources()
        query_job_resource = query_job_resources.resources.add()

        # Write the job proto to output
        query_job_resource.resource_type = "BigQueryJob"
        query_job_resource.resource_uri = query_job.self_link

        return query_job_resource

    query_job_resources = log_confusion_matrix(
        model_name=model,
        table_name=table_name,
        query_statement=query_statement,
        thresholds=thresholds,
    )

    query_job_resources_serialized = json_format.MessageToJson(query_job_resources)

    from collections import namedtuple

    output = namedtuple("Outputs", ["gcp_resources"])
    return output(query_job_resources_serialized)


@component(
    packages_to_install=[
        "google-cloud-bigquery[all]",
        "protobuf",
        "google-cloud-pipeline-components",
    ]
)
def create_roc_curve(
    project: str,
    location: str,
    model: str,  # TODO: Change to Input[BQMLModel]
    classification_metrics: Output[ClassificationMetrics],
    query_statement: Optional[str] = None,
    table_name: Optional[str] = None,
    thresholds: Optional[float] = None,
) -> NamedTuple("Outputs", [("gcp_resources", str)]):
    """Get the ROC curve

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-roc
    """

    from google.cloud import bigquery
    from google_cloud_pipeline_components.experimental.proto.gcp_resources_pb2 import (
        GcpResources,
    )
    from google.protobuf import json_format

    # Build query
    def build_query(
        action_name: str,
        model_name: str,
        table_name: Optional[str],
        query_statement: Optional[str],
        thresholds: List[float],
    ) -> str:
        parameters = [
            f"MODEL `{model_name}`",
        ]

        if table_name and query_statement:
            raise ValueError(
                "Only one of 'table_name' or 'query_statement' can be set, but not both."
            )
        elif table_name:
            parameters.append(f"TABLE `{table_name}`")
        elif query_statement:
            parameters.append(f"({query_statement})")

        if thresholds:
            parameters.append(f"GENERATE_ARRAY({', '.join(thresholds)})")

        return f"SELECT * FROM ML.{action_name}({', '.join(parameters)})"

    def log_roc_curve(
        model_name: str,
        table_name: Optional[str],
        query_statement: Optional[str],
        thresholds: List[float],
    ) -> GcpResources:
        query = build_query(
            action_name="ROC_CURVE",
            model_name=model_name,  # TODO: Pass name
            table_name=table_name,
            query_statement=query_statement,
            thresholds=thresholds,
        )

        client = bigquery.Client(project=project, location=location)

        query_job = client.query(query)  # API request

        df = query_job.to_dataframe()  # Waits for query to finish

        df_dict = df.to_dict(orient="list")

        classification_metrics.log_roc_curve(
            fpr=df_dict["false_positive_rate"],
            tpr=df_dict["recall"],
            threshold=df_dict["threshold"],
        )

        # Instantiate GCPResources Proto
        query_job_resources = GcpResources()
        query_job_resource = query_job_resources.resources.add()

        # Write the job proto to output
        query_job_resource.resource_type = "BigQueryJob"
        query_job_resource.resource_uri = query_job.self_link

        return query_job_resources

    query_job_resources = log_roc_curve(
        model_name=model,
        table_name=table_name,
        query_statement=query_statement,
        thresholds=thresholds,
    )

    query_job_resources_serialized = json_format.MessageToJson(query_job_resources)

    from collections import namedtuple

    output = namedtuple("Outputs", ["gcp_resources"])
    return output(query_job_resources_serialized)


@component(
    packages_to_install=[
        "google-cloud-bigquery[all]",
        "protobuf",
        "google-cloud-pipeline-components",
    ]
)
def query(
    # An input parameter of type string.
    query: str,
    bq_output_table_id: str,
    project: str,
) -> str:
    """Query

    https://cloud.google.com/bigquery/docs/writing-results?hl=en
    """

    from google.cloud import bigquery

    client = bigquery.Client(project=project)

    job_config = bigquery.QueryJobConfig(destination=bq_output_table_id)

    query_job = client.query(query, job_config=job_config)  # API request
    query_job.result()  # Waits for query to finish

    # TODO: Save job ID in metadata

    return bq_output_table_id
