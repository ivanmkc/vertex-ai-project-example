import dataclasses
from training.common.dataset_training_deploy_pipeline import (
    DatasetTrainingDeployPipeline,
)
from typing import Callable, List
from kfp.v2.dsl import Dataset
from google.cloud.aiplatform.utils import source_utils
from training.common.managed_dataset_pipeline import ManagedDataset
from typing import Callable, Dict, List, Optional, Sequence, Union
from google_cloud_pipeline_components import aiplatform as gcc_aip

PYTHON_MODULE_NAME = f"{source_utils._TrainingScriptPythonPackager._ROOT_MODULE}.{source_utils._TrainingScriptPythonPackager._TASK_MODULE_NAME}"


@dataclasses.dataclass
class CustomPythonPackageTrainingInfo:
    # display_name: str
    # python_module_name: str
    container_uri: str
    model_serving_container_image_uri: Optional[str] = None
    model_serving_container_predict_route: Optional[str] = None
    model_serving_container_health_route: Optional[str] = None
    model_serving_container_command: Optional[Sequence[str]] = None
    model_serving_container_args: Optional[Sequence[str]] = None
    model_serving_container_environment_variables: Optional[Dict[str, str]] = None
    model_serving_container_ports: Optional[Sequence[int]] = None
    model_description: Optional[str] = None
    model_instance_schema_uri: Optional[str] = None
    model_parameters_schema_uri: Optional[str] = None
    model_prediction_schema_uri: Optional[str] = None
    # project: Optional[str] = None
    # location: Optional[str] = None
    # credentials: Optional[auth_credentials.Credentials] = None
    training_encryption_spec_key_name: Optional[str] = None
    model_encryption_spec_key_name: Optional[str] = None
    # staging_bucket: Optional[str] = None
    annotation_schema_uri: Optional[str] = None
    model_display_name: Optional[str] = None
    base_output_dir: Optional[str] = None
    service_account: Optional[str] = None
    network: Optional[str] = None
    bigquery_destination: Optional[str] = None
    args: Optional[List[Union[str, float, int]]] = None
    environment_variables: Optional[Dict[str, str]] = None
    replica_count: int = 0
    machine_type: str = "n1-standard-4"
    accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED"
    accelerator_count: int = 0
    training_fraction_split: float = 0.8
    validation_fraction_split: float = 0.1
    test_fraction_split: float = 0.1
    predefined_split_column_name: Optional[str] = None
    tensorboard: Optional[str] = None


class CustomPythonPackageManagedDatasetPipeline(DatasetTrainingDeployPipeline):
    def __init__(
        self,
        name: str,
        managed_dataset: ManagedDataset,
        should_deploy: bool,
        training_script_path: str,
        requirements: List[str],
        training_info: CustomPythonPackageTrainingInfo,
    ):
        super().__init__(
            name=name,
            managed_dataset=managed_dataset,
            should_deploy=should_deploy,
        )

        self.training_script_path = training_script_path
        self.requirements = requirements
        self.training_info = training_info

    def _create_training_op_for_package(
        self,
        project: str,
        pipeline_root: str,
        dataset: Dataset,
        package_gcs_uri: str,
        python_module_name: str,
    ) -> Callable:
        return gcc_aip.CustomPythonPackageTrainingJobRunOp(
            display_name=self.name,
            python_package_gcs_uri=package_gcs_uri,
            python_module=python_module_name,
            container_uri=self.training_info.container_uri,
            model_serving_container_image_uri=self.training_info.model_serving_container_image_uri,
            model_serving_container_predict_route=self.training_info.model_serving_container_predict_route,
            model_serving_container_health_route=self.training_info.model_serving_container_health_route,
            model_serving_container_command=self.training_info.model_serving_container_command,
            model_serving_container_args=self.training_info.model_serving_container_args,
            model_serving_container_environment_variables=self.training_info.model_serving_container_environment_variables,
            model_serving_container_ports=self.training_info.model_serving_container_ports,
            model_description=self.training_info.model_description,
            model_instance_schema_uri=self.training_info.model_instance_schema_uri,
            model_parameters_schema_uri=self.training_info.model_parameters_schema_uri,
            model_prediction_schema_uri=self.training_info.model_prediction_schema_uri,
            project=project,
            # location=self.training_info.location,
            # credentials=self.training_info.credentials,
            training_encryption_spec_key_name=self.training_info.training_encryption_spec_key_name,
            model_encryption_spec_key_name=self.training_info.model_encryption_spec_key_name,
            staging_bucket=pipeline_root,
            dataset=dataset,
            annotation_schema_uri=self.training_info.annotation_schema_uri,
            model_display_name=self.training_info.model_display_name,
            base_output_dir=self.training_info.base_output_dir,
            service_account=self.training_info.service_account,
            network=self.training_info.network,
            bigquery_destination=self.training_info.bigquery_destination,
            args=self.training_info.args,
            environment_variables=self.training_info.environment_variables,
            replica_count=self.training_info.replica_count,
            machine_type=self.training_info.machine_type,
            accelerator_type=self.training_info.accelerator_type,
            accelerator_count=self.training_info.accelerator_count,
            training_fraction_split=self.training_info.training_fraction_split,
            validation_fraction_split=self.training_info.validation_fraction_split,
            test_fraction_split=self.training_info.test_fraction_split,
            predefined_split_column_name=self.training_info.predefined_split_column_name,
            tensorboard=self.training_info.tensorboard,
        )

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

        return self._create_training_op_for_package(
            project=project,
            pipeline_root=pipeline_root,
            dataset=dataset,
            package_gcs_uri=package_gcs_uri,
            python_module_name=PYTHON_MODULE_NAME,
        )
