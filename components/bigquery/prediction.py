from typing import Dict, List
from kfp.v2.dsl import component
from typing import Optional


@component(packages_to_install=["google-cloud-bigquery[all]"])
def predict(
    project: str,
    location: str,
    model_name: str,  # Model name
    query_statement: str = "",  # Optional: Blocked by b/198790426
    table_name: str = "",  # Optional: Blocked by b/198790426
    threshold: Optional[float] = None,  # Optional: Blocked by b/198790426
    keep_original_columns: Optional[bool] = None,  # Optional: Blocked by b/198790426
    destination_table_id: str = "",  # Optional: Blocked by b/198790426
) -> str:
    """Get prediction

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-predict
    """

    from google.cloud import bigquery

    # Build query
    def build_query(
        model_name: str,
        table_name: str = None,
        query_statement: str = None,
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
        model_name=model_name,
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

    return f"{destination.project}.{destination.dataset_id}.{destination.table_id}"
