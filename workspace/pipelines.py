from training.tabular.bq_training_pipelines import (
    BQMLTrainingPipeline,
    BQQueryAutoMLPipeline,
    BQQueryCustomPipeline,
    BQMLExportToVertexAI,
)
from training.common.custom_python_package_training_pipeline import (
    CustomPythonPackageTrainingInfo,
    CustomPythonPackageManagedDatasetPipeline,
)

from training.common.dataset_training_deploy_pipeline import DeployInfo, ExportInfo
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip
from workspace.datasets import datasets

# Custom training constants
TRAIN_VERSION = "tf-gpu.2-1"
DEPLOY_VERSION = "tf2-gpu.2-1"

TRAIN_IMAGE = "gcr.io/cloud-aiplatform/training/{}:latest".format(TRAIN_VERSION)
DEPLOY_IMAGE = "gcr.io/cloud-aiplatform/prediction/{}:latest".format(DEPLOY_VERSION)

TRAIN_GPU = aip.AcceleratorType.NVIDIA_TESLA_K80
TRAIN_NGPU = 1
TRAIN_COMPUTE = "n1-standard-4"
DEPLOY_COMPUTE = "n1-standard-4"


class pipelines:
    class classification:
        # automl_pipeline = AutoMLImageManagedDatasetPipeline(
        #     name="image-classification-automl",
        #     managed_dataset=datasets.classification.flowers,
        #     training_info=AutoMLImageTrainingInfo(
        #         prediction_type="classification",
        #         model_type="CLOUD",
        #         training_fraction_split=0.6,
        #         validation_fraction_split=0.2,
        #         test_fraction_split=0.2,
        #         budget_milli_node_hours=8000,
        #     ),
        # )

        # automl_pipeline_car_damage = AutoMLImageManagedDatasetPipeline(
        #     name="image-classification-automl-car-damage",
        #     managed_dataset=datasets.classification.car_damage,
        #     training_info=AutoMLImageTrainingInfo(
        #         prediction_type="classification",
        #         model_type="CLOUD",
        #         training_fraction_split=0.6,
        #         validation_fraction_split=0.2,
        #         test_fraction_split=0.2,
        #         budget_milli_node_hours=8000,
        #     ),
        # )

        custom_pipeline = CustomPythonPackageManagedDatasetPipeline(
            name="image-classification-custom",
            managed_dataset=datasets.classification.flowers,
            training_script_path=(
                "training/image/custom_tasks/image_classification_task.py"
            ),
            requirements=["tqdm", "tensorflow_datasets==1.3.0"],
            training_info=CustomPythonPackageTrainingInfo(
                container_uri=TRAIN_IMAGE,
                model_serving_container_image_uri=DEPLOY_IMAGE,
                annotation_schema_uri=aiplatform.schema.dataset.annotation.image.classification,
                args=["--epochs", "50", "--image-width", "32", "--image-height", "32"],
                replica_count=1,
                machine_type=TRAIN_COMPUTE,
                accelerator_type=TRAIN_GPU.name,
                accelerator_count=TRAIN_NGPU,
            ),
            deploy_info=DeployInfo(
                machine_type=DEPLOY_COMPUTE,
                accelerator_count=1,
                accelerator_type="NVIDIA_TESLA_K80",
            ),
        )

    class object_detection:
        # automl_pipeline = AutoMLImageManagedDatasetPipeline(
        #     name="object-detection-automl",
        #     managed_dataset=datasets.object_detection.mineral_plants,
        #     training_info=AutoMLImageTrainingInfo(
        #         prediction_type="object_detection",
        #         model_type="MOBILE_TF_LOW_LATENCY_1",
        #         budget_milli_node_hours=20000,
        #     ),
        #     export_info=ExportInfo(
        #         export_format_id="tf-saved-model",
        #         artifact_destination="gs://mineral-cloud-data/exported_models",
        #     ),
        # )

        custom_pipeline = CustomPythonPackageManagedDatasetPipeline(
            name="object-detection-custom",
            managed_dataset=datasets.object_detection.salads,
            training_script_path=(
                "training/image/custom_tasks/image_object_detection_task.py"
            ),
            requirements=["tqdm", "tensorflow_datasets==1.3.0"],
            training_info=CustomPythonPackageTrainingInfo(
                container_uri=TRAIN_IMAGE,
                model_serving_container_image_uri=DEPLOY_IMAGE,
                annotation_schema_uri=aiplatform.schema.dataset.annotation.image.bounding_box,
                args=["--epochs", "50", "--image-width", "32", "--image-height", "32"],
                replica_count=1,
                machine_type=TRAIN_COMPUTE,
                accelerator_type=TRAIN_GPU.name,
                accelerator_count=TRAIN_NGPU,
            ),
            deploy_info=DeployInfo(
                machine_type=DEPLOY_COMPUTE,
                accelerator_count=1,
                accelerator_type="NVIDIA_TESLA_K80",
            ),
            export_info=ExportInfo(
                export_format_id="custom_trained",
                artifact_destination="gs://mineral-cloud-data/exported_models",
            ),
        )

    class image_segmentation:

        custom_pipeline = CustomPythonPackageManagedDatasetPipeline(
            name="image-segmentation-custom",
            managed_dataset=datasets.image_segmentation.mineral_leaves,
            training_script_path=(
                "training/image/custom_tasks/image_segmentation_task.py"
            ),
            requirements=[
                "tensorflow_examples @ git+https://github.com/tensorflow/examples.git@master#egg=corepkg",
                "Pillow",
                "tensorflow",
                "google-cloud-aiplatform",
            ],
            training_info=CustomPythonPackageTrainingInfo(
                container_uri=TRAIN_IMAGE,
                model_serving_container_image_uri=DEPLOY_IMAGE,
                annotation_schema_uri=aiplatform.schema.dataset.annotation.image.segmentation,
                args=[
                    "--epochs",
                    "1",
                    "--distribute",
                    "multi",
                ],
                replica_count=1,
                machine_type=TRAIN_COMPUTE,
                accelerator_type=TRAIN_GPU.name,
                accelerator_count=TRAIN_NGPU,
            ),
            metric_key_for_comparison="accuracy",
            deploy_info=DeployInfo(
                machine_type=DEPLOY_COMPUTE,
                accelerator_count=1,
                accelerator_type="NVIDIA_TESLA_K80",
            ),
            export_info=ExportInfo(
                export_format_id="custom-trained",
                artifact_destination="gs://mineral-cloud-data/exported_models",
            ),
        )

    class tabular:
        bqml_custom_predict = BQMLTrainingPipeline(
            name="bqml-custom-predict",
            query_training="""
                CREATE OR REPLACE MODEL `bqml_tutorial_ivan.sample_model3`
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
            query_statement_evaluation="""
                SELECT
                    IF(totals.transactions IS NULL,  0, 1) AS label,
                    IFNULL(device.operatingSystem, "") AS os,
                    device.isMobile AS is_mobile,
                    IFNULL(geoNetwork.country, "") AS country,
                    IFNULL(totals.pageviews, 0) AS pageviews
                FROM
                    `bigquery-public-data.google_analytics_sample.ga_sessions_*`
                WHERE
                    _TABLE_SUFFIX BETWEEN '20170701' AND '20170801'        
            """,
            query_statement_prediction="""
                SELECT
                    IFNULL(device.operatingSystem, "") AS os,
                    device.isMobile AS is_mobile,
                    IFNULL(geoNetwork.country, "") AS country,
                    IFNULL(totals.pageviews, 0) AS pageviews
                FROM
                    `bigquery-public-data.google_analytics_sample.ga_sessions_*`
                WHERE
                    _TABLE_SUFFIX BETWEEN '20170801' AND '20170901'        
            """,
        )

        bq_automl = BQQueryAutoMLPipeline(
            name="bq-automl",
            query="SELECT * FROM `bigquery-public-data.ml_datasets.penguins`",
            training_location="us-central1",
            optimization_prediction_type="classification",
            target_column="species",
        )

        bq_custom = BQQueryCustomPipeline(
            name="bq-query-custom",
            query="SELECT * FROM `bigquery-public-data.ml_datasets.penguins`",
            training_image_uri=TRAIN_IMAGE,
            export_format="CSV",
            training_location="us-central1",
        )

        bqml_export_vertexai = BQMLExportToVertexAI(
            name="bqml-export-vertexai",
            query_training="""
                CREATE OR REPLACE MODEL `bqml_tutorial_ivan.sample_model3`
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
