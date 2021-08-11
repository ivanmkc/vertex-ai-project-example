from typing import Dict, List
from enum import Enum
from kfp.v2.dsl import Input, Output, component, ClassificationMetrics
from typing import Any, Callable, Optional


class BQMLModelCreateMode(Enum):
    CREATE_MODEL = "CREATE MODEL"
    CREATE_MODEL_IF_NOT_EXISTS = "CREATE MODEL IF NOT EXISTS"
    CREATE_OR_REPLACE_MODEL = "CREATE OR REPLACE MODEL"


class BQMLCreateModelOptions:
    pass


@component(packages_to_install=["google-cloud-bigquery"])
def create_model(
    # Create mode
    create_mode: str,  # BQMLModelCreateMode
    # Model name
    model_name: str,
    # Options
    create_model_options: str,  # BQMLCreateModelOptions
    # SELECT query
    select_query: str,
) -> str:
    """Create a BQML model

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create
    """

    from google.cloud import bigquery

    # TODO: Build query
    query = ""

    client = bigquery.Client()

    query_job = client.query(query)  # API request
    rows = query_job.result()  # Waits for query to finish

    return model_name


@component(packages_to_install=["google-cloud-bigquery"])
def create_roc_curve(
    # Model name
    model_name: str,
    # Table name
    table_name: str,
    # Thresholds
    thresholds_str: str,
    metrics_classification: Output[ClassificationMetrics],
):
    """Get the ROC curve

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-roc
    """

    from google.cloud import bigquery

    # Build query
    if len(thresholds_str) > 0:
        query = f"""
        SELECT
        *
        FROM
        ML.ROC_CURVE(MODEL `{model_name}`,
            TABLE `{table_name}`,
            GENERATE_ARRAY({thresholds_str}))
        """
    else:
        query = f"""
        SELECT
        *
        FROM
        ML.ROC_CURVE(MODEL `{model_name}`,
            TABLE `{table_name}`
        """

    client = bigquery.Client()

    query_job = client.query(query)  # API request
    # rows = query_job.result()  # Waits for query to finish


@component(packages_to_install=["google-cloud-bigquery"])
def query(
    # An input parameter of type string.
    query: str,
):
    """Query"""

    from google.cloud import bigquery

    client = bigquery.Client()

    query_job = client.query(query)  # API request
    rows = query_job.result()  # Waits for query to finish

    return
