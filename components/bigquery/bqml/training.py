from kfp.v2.components.types.artifact_types import Artifact
from kfp.v2.dsl import (
    Input,
    Output,
    OutputPath,
    component,
    ClassificationMetrics,
    Metrics,
)
from google.cloud.bigquery import Model
from typing import NamedTuple, Optional


@component(
    packages_to_install=[
        "google-cloud-bigquery[all]",
        "google-cloud-pipeline-components",
    ]
)
def bqml_create_model_op(
    project: str,
    location: str,
    query: str,
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    encryption_spec_key_name: Optional[str] = None,
) -> NamedTuple("Outputs", [("gcp_resources", str), ("model", str)]):

    """Create a BQML model

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create
    """

    import collections
    from google.cloud import bigquery
    from google_cloud_pipeline_components.experimental.proto.gcp_resources_pb2 import (
        GcpResources,
    )
    from google.protobuf import json_format
    from typing import List, Optional

    client = bigquery.Client(project=project, location=location)

    # TODO: Add labels: https://cloud.google.com/bigquery/docs/adding-labels#job-label
    job_config = bigquery.QueryJobConfig()
    if encryption_spec_key_name:
        encryption_config = bigquery.EncryptionConfiguration(
            encryption_spec_key_name=encryption_spec_key_name
        )
        job_config.destination_encryption_configuration = encryption_config

    query_job = client.query(query, job_config=job_config)  # API request

    _ = query_job.result()  # Waits for query to finish

    # Retrieve model name and model
    table: bigquery.table.TableReference = query_job.ddl_target_table
    model_name = f"{table.project}.{table.dataset_id}.{table.table_id}"
    model = client.get_model(model_name)

    def get_is_classification(model: Model) -> bool:
        if model.training_runs:
            last_training_run = model.training_runs[-1]

            return (
                last_training_run.evaluation_metrics.binary_classification_metrics
                is not None
            ) or (
                last_training_run.evaluation_metrics.multi_class_classification_metrics
                is not None
            )
        else:
            return False

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
        encryption_spec_key_name: Optional[str],
    ) -> GcpResources:
        query = build_query(
            action_name="EVALUATE",
            model_name=model_name,
            table_name=table_name,
            query_statement=query_statement,
            thresholds=thresholds,
        )

        client = bigquery.Client(project=project, location=location)

        job_config = bigquery.QueryJobConfig()
        if encryption_spec_key_name:
            encryption_config = bigquery.EncryptionConfiguration(
                kms_key_name=encryption_spec_key_name
            )
            job_config.destination_encryption_configuration = encryption_config

        query_job = client.query(query, job_config=job_config)  # API request

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

    def log_confusion_matrix(
        model_name: str,
        table_name: Optional[str],
        query_statement: Optional[str],
        thresholds: List[float],
        encryption_spec_key_name: Optional[str],
    ) -> GcpResources:
        query = build_query(
            action_name="CONFUSION_MATRIX",
            model_name=model_name,
            table_name=table_name,
            query_statement=query_statement,
            thresholds=thresholds,
        )

        client = bigquery.Client(project=project, location=location)

        job_config = bigquery.QueryJobConfig()
        if encryption_spec_key_name:
            encryption_config = bigquery.EncryptionConfiguration(
                kms_key_name=encryption_spec_key_name
            )
            job_config.destination_encryption_configuration = encryption_config

        query_job = client.query(query, job_config=job_config)  # API request

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

    def log_roc_curve(
        model_name: str,
        table_name: Optional[str],
        query_statement: Optional[str],
        thresholds: List[float],
        encryption_spec_key_name: Optional[str],
    ) -> GcpResources:
        query = build_query(
            action_name="ROC_CURVE",
            model_name=model_name,
            table_name=table_name,
            query_statement=query_statement,
            thresholds=thresholds,
        )

        client = bigquery.Client(project=project, location=location)

        job_config = bigquery.QueryJobConfig()
        if encryption_spec_key_name:
            encryption_config = bigquery.EncryptionConfiguration(
                kms_key_name=encryption_spec_key_name
            )
            job_config.destination_encryption_configuration = encryption_config

        query_job = client.query(query, job_config=job_config)  # API request

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

    log_evaluations(
        model_name=model_name,
        table_name=None,
        query_statement=None,
        thresholds=[],
        encryption_spec_key_name=encryption_spec_key_name,
    )

    if get_is_classification(model=model):
        # Log confusion matric
        log_confusion_matrix(
            model_name=model_name,
            table_name=None,
            query_statement=None,
            thresholds=[],
            encryption_spec_key_name=encryption_spec_key_name,
        )

        # Log roc curve
        log_roc_curve(
            model_name=model_name,
            table_name=None,
            query_statement=None,
            thresholds=[],
            encryption_spec_key_name=encryption_spec_key_name,
        )

    # Instantiate GCPResources Proto
    query_job_resources = GcpResources()
    query_job_resource = query_job_resources.resources.add()

    # Write the job proto to output
    query_job_resource.resource_type = "BigQueryJob"
    query_job_resource.resource_uri = query_job.self_link

    query_job_resources_serialized = json_format.MessageToJson(query_job_resources)

    from collections import namedtuple

    output = namedtuple("Outputs", ["gcp_resources", "model"])
    return output(query_job_resources_serialized, model_name)
