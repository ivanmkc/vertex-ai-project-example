from training.common.custom_python_package_training_pipeline import (
    CustomPythonPackageManagedDatasetPipeline,
)
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import Callable
from kfp.v2.dsl import Dataset
from google.cloud import aiplatform
from google.cloud.aiplatform.utils import source_utils
from google.cloud.aiplatform import gapic as aip

TRAIN_VERSION = "tf-gpu.2-1"
DEPLOY_VERSION = "tf2-gpu.2-1"

TRAIN_IMAGE = "gcr.io/cloud-aiplatform/training/{}:latest".format(TRAIN_VERSION)
DEPLOY_IMAGE = "gcr.io/cloud-aiplatform/prediction/{}:latest".format(DEPLOY_VERSION)

TRAIN_GPU, TRAIN_NGPU = (aip.AcceleratorType.NVIDIA_TESLA_K80, 1)

MACHINE_TYPE = "n1-standard"

VCPU = "4"
TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU

ANNOTATION_SCHEMA_URI = aiplatform.schema.dataset.annotation.image.classification


class ImageClassificationCustomPythonPackageManagedDatasetPipeline(
    CustomPythonPackageManagedDatasetPipeline
):
    id = "Image Classification Custom Python Package"
    managed_dataset_uri: str = "aiplatform://v1/projects/1012616486416/locations/us-central1/datasets/7601275726536376320"
    should_deploy = True

    training_script_path = (
        "training/image/classification/custom_tasks/image_classification_task.py"
    )
    requirements = ["tqdm", "tensorflow_datasets==1.3.0"]

    def create_training_op_for_package(
        self,
        project: str,
        pipeline_root: str,
        dataset: Dataset,
        package_gcs_uri: str,
        python_module_name: str,
    ) -> Callable:
        return gcc_aip.CustomPythonPackageTrainingJobRunOp(
            display_name=self.pipeline_name,
            python_package_gcs_uri=package_gcs_uri,
            python_module=python_module_name,
            container_uri=TRAIN_IMAGE,
            model_serving_container_image_uri=DEPLOY_IMAGE,
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
