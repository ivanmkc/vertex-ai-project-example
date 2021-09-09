from typing import Dict, List
from enum import Enum
from kfp.v2.dsl import Input, Output, component, ClassificationMetrics, Metrics
from typing import Any, Callable, Optional, Sequence, Union


class BQMLModelCreateMode(Enum):
    CREATE_MODEL = "CREATE MODEL"
    CREATE_MODEL_IF_NOT_EXISTS = "CREATE MODEL IF NOT EXISTS"
    CREATE_OR_REPLACE_MODEL = "CREATE OR REPLACE MODEL"


class BQMLCreateModelOptions:
    def to_sql(self) -> Dict:
        options_dict = {"model_type": "logistic_reg"}

        # TODO: Fix after https://b.corp.google.com/issues/198506675
        create_model_options_str = ",".join(
            [f"{key}='{value}'" for key, value in options_dict.items()]
        )

        return create_model_options_str


@component(packages_to_install=["google-cloud-bigquery[all]"])
def create_model(
    project: str,
    location: str,
    create_mode_str: str,  # BQMLModelCreateMode
    model_name: str,  # Model name
    create_model_options_str: str,  # Model OPTIONS, Convert to Dict after after https://b.corp.google.com/issues/198506675
    # query_statement: str,  # SELECT query used to generate the training data
    transform_statement: str = "",
    select_statement: str = "",
) -> str:
    """Create a BQML model

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create
    """

    from google.cloud import bigquery

    # create_model_options_str = ",".join(
    #     [f"{key}={value}" for key, value in create_model_options.items()]
    # )

    create_model_options_str = create_model_options_str

    # Build query
    all_statements = [f"{create_mode_str} `{model_name}`"]

    if transform_statement:
        all_statements.append(transform_statement)

    if create_model_options_str:
        all_statements.append(f"OPTIONS({create_model_options_str})")

    all_statements.append(f"AS {select_statement}")

    query = " ".join(all_statements)

    client = bigquery.Client(project=project, location=location)

    query_job = client.query(query)  # API request
    _ = query_job.result()  # Waits for query to finish

    return model_name
