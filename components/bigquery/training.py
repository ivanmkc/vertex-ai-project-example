from kfp.v2.components.types.artifact_types import Artifact
from kfp.v2.dsl import Input, Output, component, ClassificationMetrics, Metrics
from google.cloud.bigquery import Model
import google.cloud.bigquery_v2


@component(packages_to_install=["google-cloud-bigquery[all]"])
def create_model(
    project: str,
    location: str,  # TODO: Check if service_account or CMEK are required
    model_name: str,
    model_name_output: Output[Artifact],
    sql_statement: str,
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
):  # TODO: Decide on returning BQModel or str (i.e. uri)
    """Create a BQML model

    https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create
    """

    from google.cloud import bigquery

    def process_binary_classification_metrics(
        binary_classification_metrics: "google.cloud.bigquery_v2.types.Model.BinaryClassificationMetrics",
        metrics: Output[Metrics],
        classification_metrics: Output[ClassificationMetrics],
    ):
        print(
            f"binary_classification_metrics.aggregate_classification_metrics: {binary_classification_metrics.aggregate_classification_metrics}"
        )
        print(
            f"type(binary_classification_metrics.aggregate_classification_metrics: {binary_classification_metrics.aggregate_classification_metrics})"
        )
        for key, value in vars(
            binary_classification_metrics.aggregate_classification_metrics
        ):
            metrics.log_metric(key, value)

        # TODO
        # multi_class_classification_metrics.confusion_matrix_list
        print(
            f"binary_classification_metrics.binary_confusion_matrix_list: {binary_classification_metrics.binary_confusion_matrix_list}"
        )

    def process_multiclass_classification_metrics(
        multi_class_classification_metrics: google.cloud.bigquery_v2.types.model.Model.MultiClassClassificationMetrics,
        metrics: Output[Metrics],
        classification_metrics: Output[ClassificationMetrics],
    ):
        for key, value in vars(
            multi_class_classification_metrics.aggregate_classification_metrics
        ):
            metrics.log_metric(key, value)

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

        process_binary_classification_metrics(
            binary_classification_metrics=last_training_run.evaluation_metrics.binary_classification_metrics,
            metrics=metrics,
            classification_metrics=classification_metrics,
        )

        process_multiclass_classification_metrics(
            multi_class_classification_metrics=last_training_run.evaluation_metrics.multi_class_classification_metrics,
            metrics=metrics,
            classification_metrics=classification_metrics,
        )

    model_name_output = model_name
