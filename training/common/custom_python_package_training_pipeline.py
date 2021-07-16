from abc import abstractmethod
from training.common.import_training_deploy_pipeline import (
    ImportTrainingDeployPipeline,
)
from typing import Callable, List
from kfp.v2.dsl import Dataset
from google.cloud.aiplatform.utils import source_utils

PYTHON_MODULE_NAME = f"{source_utils._TrainingScriptPythonPackager._ROOT_MODULE}.{source_utils._TrainingScriptPythonPackager._TASK_MODULE_NAME}"


class CustomPythonPackageManagedDatasetPipeline(ImportTrainingDeployPipeline):
    @property
    @abstractmethod
    def training_script_path(self) -> str:
        """
        Path to the training script. Relative to project root.

        e.g. "training/image/classification/custom_tasks/image_classification_task.py"

        """
        pass

    @property
    @abstractmethod
    def requirements(self) -> List[str]:
        """
        List of Python dependencies to add to package
        """
        pass

    @abstractmethod
    def create_training_op_for_package(
        self,
        project: str,
        pipeline_root: str,
        dataset: Dataset,
        package_gcs_uri: str,
        python_module_name: str,
    ) -> Callable:
        pass

    def _package_and_upload_module(self, project: str, pipeline_root: str) -> str:
        # Create packager
        python_packager = source_utils._TrainingScriptPythonPackager(
            script_path=self.training_script_path, requirements=self.requirements
        )

        # Package and upload to GCS
        package_gcs_uri = python_packager.package_and_copy_to_gcs(
            gcs_staging_dir=pipeline_root,
            project=project,
        )

        print(f"Custom Training Python Package is uploaded to: {package_gcs_uri}")

        return package_gcs_uri

    def create_training_op(
        self, project: str, pipeline_root: str, dataset: Dataset
    ) -> Callable:
        package_gcs_uri = self._package_and_upload_module(
            project=project, pipeline_root=pipeline_root
        )

        return self.create_training_op_for_package(
            project=project,
            pipeline_root=pipeline_root,
            dataset=dataset,
            package_gcs_uri=package_gcs_uri,
            python_module_name=PYTHON_MODULE_NAME,
        )
