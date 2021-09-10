from kfp.v2.dsl import Input, Output, component, ClassificationMetrics, Metrics
from typing import Optional
from kfp.v2.components.types.artifact_types import Artifact


@component(packages_to_install=["google-cloud-bigquery[all]"])
def create_evaluation(
    project: str,
    location: str,
    model_name: Input[Artifact],  # Model name
    metrics: Output[Metrics],
    query_statement: str = "",  # Optional: Blocked by b/198790426
    table_name: str = "",  # Optional: Blocked by b/198790426
    thresholds_str: str = "",  # Optional: Blocked by b/198790426
):
    """Get the evaluation curve

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-evaluate
    """
    import collections
    from google.cloud import bigquery

    # Build query
    def build_query(
        action_name: str,
        model_name: str,
        table_name: str,
        query_statement: str,
        thresholds_str: str,
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

        if thresholds_str:
            parameters.append(f"GENERATE_ARRAY({thresholds_str})")

        return f"SELECT * FROM ML.{action_name}({', '.join(parameters)})"

    query = build_query(
        action_name="EVALUATE",
        model_name=model_name,
        table_name=table_name,
        query_statement=query_statement,
        thresholds_str=thresholds_str,
    )

    client = bigquery.Client(project=project, location=location)

    query_job = client.query(query)  # API request

    df = query_job.to_dataframe()  # Waits for query to finish

    print(df.to_dict(orient="list"))

    # Log matrix
    for name, value in df.to_dict(orient="list").items():
        if not isinstance(value, collections.Sequence):
            metrics.log_metric(name, value)
        else:
            metrics.log_metric(name, value[-1])


@component(packages_to_install=["google-cloud-bigquery[all]"])
def create_confusion_matrix(
    project: str,
    location: str,
    model_name: str,  # Model name
    classification_metrics: Output[ClassificationMetrics],
    query_statement: str = "",  # Optional: Blocked by b/198790426
    table_name: str = "",  # Optional: Blocked by b/198790426
    thresholds_str: str = "",  # Optional: Blocked by b/198790426
):
    """Get the confusion matrix

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-confusion
    """

    from google.cloud import bigquery

    # Build query
    def build_query(
        action_name: str,
        model_name: str,
        table_name: str,
        query_statement: str,
        thresholds_str: str,
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

        if thresholds_str:
            parameters.append(f"GENERATE_ARRAY({thresholds_str})")

        return f"SELECT * FROM ML.{action_name}({', '.join(parameters)})"

    query = build_query(
        action_name="CONFUSION_MATRIX",
        model_name=model_name,
        table_name=table_name,
        query_statement=query_statement,
        thresholds_str=thresholds_str,
    )

    client = bigquery.Client(project=project, location=location)

    query_job = client.query(query)  # API request

    df = query_job.to_dataframe()  # Waits for query to finish
    df = df.drop("expected_label", 1)

    categories = [column for column in df.columns]
    matrix = df.values.tolist()
    print(matrix)

    classification_metrics.log_confusion_matrix(categories=categories, matrix=matrix)


@component(packages_to_install=["google-cloud-bigquery[all]"])
def create_roc_curve(
    project: str,
    location: str,
    model_name: str,  # Model name
    classification_metrics: Output[ClassificationMetrics],
    query_statement: str = "",  # Optional: Blocked by b/198790426
    table_name: str = "",  # Optional: Blocked by b/198790426
    thresholds_str: str = "",  # Optional: Blocked by b/198790426
):
    """Get the ROC curve

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-roc
    """

    from google.cloud import bigquery

    # Build query
    def build_query(
        action_name: str,
        model_name: str,
        table_name: str,
        query_statement: str,
        thresholds_str: str,
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

        if thresholds_str:
            parameters.append(f"GENERATE_ARRAY({thresholds_str})")

        return f"SELECT * FROM ML.{action_name}({', '.join(parameters)})"

    query = build_query(
        action_name="ROC_CURVE",
        model_name=model_name,
        table_name=table_name,
        query_statement=query_statement,
        thresholds_str=thresholds_str,
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


@component(packages_to_install=["google-cloud-bigquery[all]"])
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
