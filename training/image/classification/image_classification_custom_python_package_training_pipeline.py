from training.image.classification.image_classification_training_pipeline import (
    ImageClassificationTrainingPipeline,
)
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import Callable, List
from kfp.v2.dsl import Dataset
from google.cloud import aiplatform
from google.cloud.aiplatform.utils import source_utils
from google.cloud.aiplatform import gapic as aip

TRAIN_VERSION = "tf-gpu.2-1"
DEPLOY_VERSION = "tf2-gpu.2-1"

TRAIN_IMAGE = "gcr.io/cloud-aiplatform/training/{}:latest".format(TRAIN_VERSION)
DEPLOY_IMAGE = "gcr.io/cloud-aiplatform/prediction/{}:latest".format(DEPLOY_VERSION)

TRAIN_GPU, TRAIN_NGPU = (aip.AcceleratorType.NVIDIA_TESLA_K80, 1)
# DEPLOY_GPU, DEPLOY_NGPU = (aip.AcceleratorType.NVIDIA_TESLA_K80, 1)

MACHINE_TYPE = "n1-standard"

VCPU = "4"
TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU

ANNOTATION_SCHEMA_URI = aiplatform.schema.dataset.annotation.image.classification

# Script
SCRIPT_PATH: str = "training/image/classification/custom_tasks/task.py"
REQUIREMENTS: List[str] = ["tensorflow_datasets==1.3.0"]

PYTHON_MODULE_NAME = f"{source_utils._TrainingScriptPythonPackager._ROOT_MODULE}.{source_utils._TrainingScriptPythonPackager._TASK_MODULE_NAME}"


class ImageClassificationCustomPythonPackageTrainingPipeline(
    ImageClassificationTrainingPipeline
):
    id = "Image Classification Custom Python Package"
    annotation_dataset_uri: str = "aiplatform://v1/projects/1012616486416/locations/us-central1/datasets/7601275726536376320"

    def create_training_op(
        self, project: str, pipeline_root: str, dataset: Dataset
    ) -> Callable:
        # Create packager
        python_packager = source_utils._TrainingScriptPythonPackager(
            script_path=SCRIPT_PATH, requirements=REQUIREMENTS
        )

        # Package and upload to GCS
        package_gcs_uri = python_packager.package_and_copy_to_gcs(
            gcs_staging_dir=pipeline_root,
            project=project,
        )

        print(f"Custom Training Python Package is uploaded to: {package_gcs_uri}")

        return gcc_aip.CustomPythonPackageTrainingJobRunOp(
            display_name=self.pipeline_name,
            python_package_gcs_uri=package_gcs_uri,
            python_module=PYTHON_MODULE_NAME,
            container_uri=TRAIN_IMAGE,
            model_serving_container_image_uri=DEPLOY_IMAGE,
            # model_serving_container_command=["/usr/bin/tensorflow_model_server"],
            # model_serving_container_args=[
            #     f"--model_name={self.pipeline_name}",
            #     "--model_base_path=$(AIP_STORAGE_URI)",
            #     "--rest_api_port=8080",
            #     "--port=8500",
            #     "--file_system_poll_wait_seconds=31540000",
            # ],
            # model_serving_container_predict_route=f"/v1/models/{self.pipeline_name}:predict",
            # model_serving_container_health_route=f"/v1/models/{self.pipeline_name}",
            project=project,
            staging_bucket=pipeline_root,
            dataset=dataset,
            annotation_schema_uri=ANNOTATION_SCHEMA_URI,
            args=["--epochs", "50"],
            replica_count=1,
            machine_type=TRAIN_COMPUTE,
            accelerator_type=TRAIN_GPU.name,
            accelerator_count=TRAIN_NGPU,
            model_display_name=self.pipeline_name,
        )
