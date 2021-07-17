from training.image.classification.automl_image_training_pipeline import (
    AutoMLImageTrainingInfo,
    AutoMLImageManagedDatasetPipeline,
)

from training.common.custom_python_package_training_pipeline import (
    CustomPythonPackageTrainingInfo,
    CustomPythonPackageManagedDatasetPipeline,
)
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip
from datasets import datasets

# Custom training constants
TRAIN_VERSION = "tf-gpu.2-1"
DEPLOY_VERSION = "tf2-gpu.2-1"

TRAIN_IMAGE = "gcr.io/cloud-aiplatform/training/{}:latest".format(TRAIN_VERSION)
DEPLOY_IMAGE = "gcr.io/cloud-aiplatform/prediction/{}:latest".format(DEPLOY_VERSION)

TRAIN_GPU = aip.AcceleratorType.NVIDIA_TESLA_K80
TRAIN_NGPU = 1
TRAIN_COMPUTE = "n1-standard-4"


class pipelines:
    class classification:
        automl_pipeline = AutoMLImageManagedDatasetPipeline(
            name="image-classification-automl",
            managed_dataset=datasets.classification.flowers,
            should_deploy=True,
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
            should_deploy=True,
            training_script_path=(
                "training/image/classification/custom_tasks/image_classification_task.py"
            ),
            requirements=["tqdm", "tensorflow_datasets==1.3.0"],
            training_info=CustomPythonPackageTrainingInfo(
                container_uri=TRAIN_IMAGE,
                model_serving_container_image_uri=DEPLOY_IMAGE,
                annotation_schema_uri=aiplatform.schema.dataset.annotation.image.classification,
                args=["--epochs", "50", "--image_width", "32", "--image_height", "32"],
                replica_count=1,
                machine_type=TRAIN_COMPUTE,
                accelerator_type=TRAIN_GPU.name,
                accelerator_count=TRAIN_NGPU,
            ),
        )

    class object_detection:
        automl_pipeline = AutoMLImageManagedDatasetPipeline(
            name="object-detection-automl",
            managed_dataset=datasets.object_detection.salads,
            should_deploy=True,
            training_info=AutoMLImageTrainingInfo(
                prediction_type="object_detection",
                model_type="CLOUD",
                training_fraction_split=0.6,
                validation_fraction_split=0.2,
                test_fraction_split=0.2,
                budget_milli_node_hours=20000,
            ),
        )

        custom_pipeline = CustomPythonPackageManagedDatasetPipeline(
            name="object-detection-custom",
            managed_dataset=datasets.object_detection.salads,
            should_deploy=True,
            training_script_path=(
                "training/image/object_detection/custom_tasks/object_detection_task.py"
            ),
            requirements=["tqdm", "tensorflow_datasets==1.3.0"],
            training_info=CustomPythonPackageTrainingInfo(
                container_uri=TRAIN_IMAGE,
                model_serving_container_image_uri=DEPLOY_IMAGE,
                annotation_schema_uri=aiplatform.schema.dataset.annotation.image.bounding_box,
                args=["--epochs", "50", "--image_width", "32", "--image_height", "32"],
                replica_count=1,
                machine_type=TRAIN_COMPUTE,
                accelerator_type=TRAIN_GPU.name,
                accelerator_count=TRAIN_NGPU,
            ),
        )
