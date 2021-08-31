import dataclasses
from training.common.dataset_training_deploy_pipeline import (
    DatasetTrainingDeployPipeline,
    DeployInfo,
    ExportInfo,
)
from typing import Callable, List
from kfp.v2.dsl import Dataset, Model
from kfp.v2.dsl import (
    ClassificationMetrics,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
    OutputPath,
    component,
)

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
        training_script_path: str,
        requirements: List[str],
        training_info: CustomPythonPackageTrainingInfo,
        deploy_info: Optional[DeployInfo] = None,
        export_info: Optional[ExportInfo] = None,
    ):
        super().__init__(
            name=name,
            managed_dataset=managed_dataset,
            deploy_info=deploy_info,
            export_info=export_info,
        )

        self.training_script_path = training_script_path
        self.requirements = requirements
        self.training_info = training_info


    def _get_confusion_matrix_uri(self, pipeline_root: str) -> str:
        return f"{pipeline_root}/confusion_matrix.json"
    
    def create_confusion_matrix_op(self, project: str, pipeline_root: str, model: Model) -> Optional[Callable]:
        @component(packages_to_install=["google-cloud-storage"])
        def confusion_matrix_op(
            project: str,
            confusion_matrix_uri: str,
            model: Input[Model],
            classification_metrics: Output[ClassificationMetrics],
        ):
            from typing import Any, Tuple

            def extract_bucket_and_prefix_from_gcs_path(gcs_path: str) -> Tuple[str, Optional[str]]:
                """Given a complete GCS path, return the bucket name and prefix as a tuple.

                Example Usage:

                    bucket, prefix = extract_bucket_and_prefix_from_gcs_path(
                        "gs://example-bucket/path/to/folder"
                    )

                    # bucket = "example-bucket"
                    # prefix = "path/to/folder"

                Args:
                    gcs_path (str):
                        Required. A full path to a Google Cloud Storage folder or resource.
                        Can optionally include "gs://" prefix or end in a trailing slash "/".

                Returns:
                    Tuple[str, Optional[str]]
                        A (bucket, prefix) pair from provided GCS path. If a prefix is not
                        present, a None will be returned in its place.
                """
                if gcs_path.startswith("gs://"):
                    gcs_path = gcs_path[5:]
                if gcs_path.endswith("/"):
                    gcs_path = gcs_path[:-1]

                gcs_parts = gcs_path.split("/", 1)
                gcs_bucket = gcs_parts[0]
                gcs_blob_prefix = None if len(gcs_parts) == 1 else gcs_parts[1]

                return (gcs_bucket, gcs_blob_prefix)    

            def download_object(bucket_name: str, blob_name: str) -> Any:
                from google.cloud import storage
                import json

                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                object_as_string = blob.download_as_string()
                return json.loads(object_as_string)

            # Extract uri components
            bucket_name, prefix = extract_bucket_and_prefix_from_gcs_path(confusion_matrix_uri)
            
            # Download confusion matrix
            confusion_matrix = download_object(bucket_name=bucket_name, blob_name=prefix)
            
            # Log matrix
            classification_metrics.log_confusion_matrix(confusion_matrix["labels"], confusion_matrix["matrix"])
        
        confusion_matrix_uri = self._get_confusion_matrix_uri(pipeline_root)

        return confusion_matrix_op(project=project, confusion_matrix_uri=confusion_matrix_uri, model=model)

    def _get_classification_report_uri(self, pipeline_root: str) -> str:
        return f"{pipeline_root}/classification_report.json"

    def create_classification_report_op(self, project: str, pipeline_root: str, model: Model) -> Optional[Callable]:
        @component(packages_to_install=["google-cloud-storage", "pandas"])
        def classification_report_op(
            project: str,
            classification_report_uri: str,
            model: Input[Model],
            mlpipeline_ui_metadata: OutputPath()
        ):
            import json
            from typing import Any, Tuple

            def extract_bucket_and_prefix_from_gcs_path(gcs_path: str) -> Tuple[str, Optional[str]]:
                """Given a complete GCS path, return the bucket name and prefix as a tuple.

                Example Usage:

                    bucket, prefix = extract_bucket_and_prefix_from_gcs_path(
                        "gs://example-bucket/path/to/folder"
                    )

                    # bucket = "example-bucket"
                    # prefix = "path/to/folder"

                Args:
                    gcs_path (str):
                        Required. A full path to a Google Cloud Storage folder or resource.
                        Can optionally include "gs://" prefix or end in a trailing slash "/".

                Returns:
                    Tuple[str, Optional[str]]
                        A (bucket, prefix) pair from provided GCS path. If a prefix is not
                        present, a None will be returned in its place.
                """
                if gcs_path.startswith("gs://"):
                    gcs_path = gcs_path[5:]
                if gcs_path.endswith("/"):
                    gcs_path = gcs_path[:-1]

                gcs_parts = gcs_path.split("/", 1)
                gcs_bucket = gcs_parts[0]
                gcs_blob_prefix = None if len(gcs_parts) == 1 else gcs_parts[1]

                return (gcs_bucket, gcs_blob_prefix)    

            def download_object(bucket_name: str, blob_name: str) -> Any:
                from google.cloud import storage
                import json

                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                object_as_string = blob.download_as_string()
                return json.loads(object_as_string)

            # Extract uri components
            bucket_name, prefix = extract_bucket_and_prefix_from_gcs_path(classification_report_uri)
            
            # Download classification report
            classification_report = download_object(bucket_name=bucket_name, blob_name=prefix)
            
            import pandas as pd

            df = pd.DataFrame(columns=["key", "precision", "recall", "f1-score", "support"])

            for key, value in classification_report.items():
                if isinstance(value, Dict):
                    value["key"] = key
                    df = df.append(value, ignore_index=True)

            metadata = {
                'outputs' : [{
                'type': 'table',
                'storage': 'inline',
                'format': 'csv',
                'header': list(df.columns),
                'source': df.to_csv()
                }]
            }

            with open(mlpipeline_ui_metadata, 'w') as metadata_file:
                json.dump(metadata, metadata_file)
        
        classification_report_uri = self._get_classification_report_uri(pipeline_root)
        return classification_report_op(project=project, classification_report_uri=classification_report_uri, model=model)

    def _get_model_history_uri(self, pipeline_root: str) -> str:
        return f"{pipeline_root}/model_history.json"
    
    def create_model_history_op(self, project: str, pipeline_root: str, model: Model) -> Optional[Callable]:
        @component(packages_to_install=["google-cloud-storage"])
        def model_history_op(
            project: str,
            model_history_uri: str,
            model: Input[Model],
            metrics: Output[Metrics],
        ):
            import collections
            from typing import Any, Tuple

            def extract_bucket_and_prefix_from_gcs_path(gcs_path: str) -> Tuple[str, Optional[str]]:
                """Given a complete GCS path, return the bucket name and prefix as a tuple.

                Example Usage:

                    bucket, prefix = extract_bucket_and_prefix_from_gcs_path(
                        "gs://example-bucket/path/to/folder"
                    )

                    # bucket = "example-bucket"
                    # prefix = "path/to/folder"

                Args:
                    gcs_path (str):
                        Required. A full path to a Google Cloud Storage folder or resource.
                        Can optionally include "gs://" prefix or end in a trailing slash "/".

                Returns:
                    Tuple[str, Optional[str]]
                        A (bucket, prefix) pair from provided GCS path. If a prefix is not
                        present, a None will be returned in its place.
                """
                if gcs_path.startswith("gs://"):
                    gcs_path = gcs_path[5:]
                if gcs_path.endswith("/"):
                    gcs_path = gcs_path[:-1]

                gcs_parts = gcs_path.split("/", 1)
                gcs_bucket = gcs_parts[0]
                gcs_blob_prefix = None if len(gcs_parts) == 1 else gcs_parts[1]

                return (gcs_bucket, gcs_blob_prefix)    

            def download_object(bucket_name: str, blob_name: str) -> Any:
                from google.cloud import storage
                import json

                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                object_as_string = blob.download_as_string()
                return json.loads(object_as_string)

            # Extract uri components
            bucket_name, prefix = extract_bucket_and_prefix_from_gcs_path(model_history_uri)
            
            # Download confusion matrix
            model_history = download_object(bucket_name=bucket_name, blob_name=prefix)
            
            # Log matrix
            for name, value in model_history.items():
                if not isinstance(value, collections.Sequence):
                    metrics.log_metric(name, value)
                else:
                    metrics.log_metric(name, value[-1])
        
        model_history_uri = self._get_model_history_uri(pipeline_root)

        return model_history_op(project=project, model_history_uri=model_history_uri, model=model)

    def _create_training_op_for_package(
        self,
        project: str,
        pipeline_root: str,
        dataset: Dataset,
        package_gcs_uri: str,
        python_module_name: str,
    ) -> Callable:
        training_args = self.training_info.args

        # TODO: Check if training task supports this arg
        training_args.extend(["--confusion_matrix_destination_uri", self._get_confusion_matrix_uri(pipeline_root)])
        training_args.extend(["--classification_report_destination_uri", self._get_classification_report_uri(pipeline_root)])
        training_args.extend(["--model_history_destination_uri", self._get_model_history_uri(pipeline_root)])

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
            args=training_args,
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
