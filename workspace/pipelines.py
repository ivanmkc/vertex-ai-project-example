from training.image.automl_image_training_pipeline import (
    AutoMLImageTrainingInfo,
    AutoMLImageManagedDatasetPipeline,
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
        automl_pipeline = AutoMLImageManagedDatasetPipeline(
            name="image-classification-automl",
            managed_dataset=datasets.classification.flowers,
            training_info=AutoMLImageTrainingInfo(
                prediction_type="classification",
                model_type="CLOUD",
                training_fraction_split=0.6,
                validation_fraction_split=0.2,
                test_fraction_split=0.2,
                budget_milli_node_hours=8000,
            ),
        )

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
        automl_pipeline = AutoMLImageManagedDatasetPipeline(
            name="object-detection-automl",
            managed_dataset=datasets.object_detection.mineral_plants,
            training_info=AutoMLImageTrainingInfo(
                prediction_type="object_detection",
                model_type="MOBILE_TF_LOW_LATENCY_1",
                budget_milli_node_hours=20000,
            ),
            export_info=ExportInfo(
                export_format_id="tf-saved-model",
                artifact_destination="gs://mineral-cloud-data/exported_models",
            ),
        )

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
