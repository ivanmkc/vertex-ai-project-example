def create_model(
    project: str,
    location: str,  # TODO: Check if service_account or CMEK are required
    model_name: str,
    # model_name_output: Output[Artifact],
    sql_statement: str,
    # metrics: Output[Metrics],
    # classification_metrics: Output[ClassificationMetrics],
):  # TODO: Decide on returning BQModel or str (i.e. uri)
    """Create a BQML model

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create
    """
    from google.protobuf import json_format
    from google.cloud import bigquery

    def process_binary_classification_metrics(
        binary_classification_metrics: "google.cloud.bigquery_v2.types.Model.BinaryClassificationMetrics",
        # metrics: Output[Metrics],
        # classification_metrics: Output[ClassificationMetrics],
    ):
        print(
            f"binary_classification_metrics.aggregate_classification_metrics: {binary_classification_metrics.aggregate_classification_metrics}"
        )
        print(
            f"type(binary_classification_metrics.aggregate_classification_metrics: {binary_classification_metrics.aggregate_classification_metrics})"
        )
        # for key, value in json_format.MessageToDict(
        #     last_training_run.evaluation_metrics.binary_classification_metrics.aggregate_classification_metrics
        # ).items():
        #     metrics.log_metric(key, value)

        # TODO
        # multi_class_classification_metrics.confusion_matrix_list
        print(
            f"binary_classification_metrics.binary_confusion_matrix_list: {binary_classification_metrics.binary_confusion_matrix_list}"
        )

    def process_multiclass_classification_metrics(
        multi_class_classification_metrics: "google.cloud.bigquery_v2.types.model.Model.MultiClassClassificationMetrics",
        # metrics: Output[Metrics],
        # classification_metrics: Output[ClassificationMetrics],
    ):
        #     for key, value in vars(
        #         multi_class_classification_metrics.aggregate_classification_metrics
        #     ):
        #         metrics.log_metric(key, value)

        # TODO
        # multi_class_classification_metrics.confusion_matrix_list
        print(
            f"multi_class_classification_metrics.confusion_matrix_list: {multi_class_classification_metrics.confusion_matrix_list}"
        )

    client = bigquery.Client(project=project, location=location)

    # TODO: Add labels: https://cloud.google.com/bigquery/docs/adding-labels#job-label
    query_job = client.query(sql_statement)  # API request
    _ = query_job.result()  # Waits for query to finish
    model = client.get_model(model_name)

    # TODO: Get evaluation metrics and store in MLMD
    last_training_run = model.training_runs[-1]
    if last_training_run:
        # process_regression_metrics(
        #     regression_metrics=last_training_run.evaluation_metrics.regression_metrics,
        #     metrics=metrics,
        #     classification_metrics=classification_metrics,
        # )

        if last_training_run.evaluation_metrics.binary_classification_metrics:
            process_binary_classification_metrics(
                binary_classification_metrics=last_training_run.evaluation_metrics.binary_classification_metrics,
                # metrics=metrics,
                # classification_metrics=classification_metrics,
            )

        if last_training_run.evaluation_metrics.multi_class_classification_metrics:
            process_multiclass_classification_metrics(
                multi_class_classification_metrics=last_training_run.evaluation_metrics.multi_class_classification_metrics,
                # metrics=metrics,
                # classification_metrics=classification_metrics,
            )

    model_name_output = model_name


create_model(
    project="python-docs-samples-tests",
    location="us",
    model_name="bqml_tutorial_ivan.sample_model",
    sql_statement="""
            CREATE OR REPLACE MODEL `bqml_tutorial_ivan.sample_model2`
            OPTIONS(model_type='logistic_reg') AS
            SELECT
            IF(totals.transactions IS NULL, 0, 1) AS label,
            IFNULL(device.operatingSystem, "") AS os,
            device.isMobile AS is_mobile,
            IFNULL(geoNetwork.country, "") AS country,
            IFNULL(totals.pageviews, 0) AS pageviews
            FROM
            `bigquery-public-data.google_analytics_sample.ga_sessions_*`
            WHERE
            _TABLE_SUFFIX BETWEEN '20160801' AND '20170630'
        """,
)
