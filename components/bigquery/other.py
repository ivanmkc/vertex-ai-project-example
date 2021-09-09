from typing import Dict, List
from kfp.v2.dsl import component
from typing import Optional


@component(packages_to_install=["google-cloud-bigquery[all]"])
def export_to_csv(
    project: str,
    location: str,
    source_table_id: str,
    source_table_location: str,  # Optional
    destination_csv_uri: str,
    # query_statement: str = "",  # Optional: Blocked by b/198790426
) -> str:
    """Get prediction

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-predict
    """

    from google.cloud import bigquery

    client = bigquery.Client(project=project, location=location)

    extract_job = client.extract_table(
        source=source_table_id,
        destination_uris=destination_csv_uri,
        # Location must match that of the source table.
        location=source_table_location,
    )  # API request
    extract_job.result()  # Waits for job to complete.

    return destination_csv_uri
    # # Build query
    # def build_query(
    #     model_name: str,
    #     table_name: str = None,
    #     query_statement: str = None,
    #     threshold: Optional[float] = None,
    #     keep_original_columns: Optional[bool] = None,
    # ) -> str:
    #     parameters = [
    #         f"MODEL `{model_name}`",
    #     ]

    #     if table_name and query_statement:
    #         raise ValueError(
    #             "Only one of 'table_name' or 'query_statement' can be set, but not both."
    #         )
    #     elif all(x is None for x in [table_name, query_statement]):
    #         raise ValueError("One of 'table_name' or 'query_statement' must be set.")
    #     elif table_name:
    #         parameters.append(f"TABLE `{table_name}`")
    #     elif query_statement:
    #         parameters.append(f"({query_statement})")

    #     settings = []
    #     if threshold is not None:
    #         settings.append(f"{threshold} AS threshold")

    #     if keep_original_columns is not None:
    #         settings.append(
    #             f"{'TRUE' if keep_original_columns else 'FALSE'} AS keep_original_columns"
    #         )

    #     if len(settings) > 0:
    #         parameters.append(f"STRUCT({', '.join(settings)})")

    #     return f"SELECT * FROM ML.PREDICT({', '.join(parameters)})"

    # query = build_query(
    #     model_name=model_name,
    #     table_name=table_name,
    #     query_statement=query_statement,
    #     threshold=threshold,
    #     keep_original_columns=keep_original_columns,
    # )

    # client = bigquery.Client(project=project, location=location)

    # job_config = None
    # if destination_table_id:
    #     job_config = bigquery.QueryJobConfig(destination=destination_table_id)

    # query_job = client.query(query, job_config=job_config)  # API request

    # _ = query_job.result()  # Waits for query to finish

    # return destination_table_id


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
