from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import abc
import dataclasses

from google_cloud_pipeline_components import aiplatform as gcc_aip
import kfp
from kfp.dsl.io_types import Model
from kfp.v2.dsl import component, importer, Condition, Dataset, Input
import training.common.managed_dataset_pipeline as managed_dataset_pipeline
from google.cloud.aiplatform import explain


@dataclasses.dataclass
class ExportInfo:
    """Info for exporting a trained, exportable Model to a location specified by the user.
    A Model is considered to be exportable if it has at least one `supported_export_formats`.
    Either `artifact_destination` or `image_destination` must be provided.

    Usage:
        ExportInfo(
            export_format_id='tf-saved-model'
            artifact_destination='gs://my-bucket/models/'
        )

        or

        ExportInfo(
            export_format_id='custom-model'
            image_destination='us-central1-docker.pkg.dev/projectId/repo/image'
        )

    Args:
        export_format_id (str):
            Required. The ID of the format in which the Model must be exported.
            The list of export formats that this Model supports can be found
            by calling `Model.supported_export_formats`.
        artifact_destination (str):
            The Cloud Storage location where the Model artifact is to be
            written to. Under the directory given as the destination a
            new one with name
            "``model-export-<model-display-name>-<timestamp-of-export-call>``",
            where timestamp is in YYYY-MM-DDThh:mm:ss.sssZ ISO-8601
            format, will be created. Inside, the Model and any of its
            supporting files will be written.

            This field should only be set when, in [Model.supported_export_formats],
            the value for the key given in `export_format_id` contains ``ARTIFACT``.
        image_destination (str):
            The Google Container Registry or Artifact Registry URI where
            the Model container image will be copied to. Accepted forms:

            -  Google Container Registry path. For example:
            ``gcr.io/projectId/imageName:tag``.

            -  Artifact Registry path. For example:
            ``us-central1-docker.pkg.dev/projectId/repoName/imageName:tag``.

            This field should only be set when, in [Model.supported_export_formats],
            the value for the key given in `export_format_id` contains ``IMAGE``.
        sync (bool):
            Whether to execute this export synchronously. If False, this method
            will be executed in concurrent Future and any downstream object will
            be immediately returned and synced when the Future has completed.
    Returns:
        output_info (Dict[str, str]):
            Details of the completed export with output destination paths to
            the artifacts or container image.
    Raises:
        ValueError if model does not support exporting.

        ValueError if invalid arguments or export formats are provided.
    """

    export_format_id: str
    artifact_destination: Optional[str] = None
    image_destination: Optional[str] = None


@dataclasses.dataclass
class DeployInfo:
    """Info for deploying a model to endpoint. Endpoint will be created if unspecified.

    Args:
        endpoint ("Endpoint"):
            Optional. Endpoint to deploy model to. If not specified, endpoint
            display name will be model display name+'_endpoint'.
        deployed_model_display_name (str):
            Optional. The display name of the DeployedModel. If not provided
            upon creation, the Model's display_name is used.
        traffic_percentage (int):
            Optional. Desired traffic to newly deployed model. Defaults to
            0 if there are pre-existing deployed models. Defaults to 100 if
            there are no pre-existing deployed models. Negative values should
            not be provided. Traffic of previously deployed models at the endpoint
            will be scaled down to accommodate new deployed model's traffic.
            Should not be provided if traffic_split is provided.
        traffic_split (Dict[str, int]):
            Optional. A map from a DeployedModel's ID to the percentage of
            this Endpoint's traffic that should be forwarded to that DeployedModel.
            If a DeployedModel's ID is not listed in this map, then it receives
            no traffic. The traffic percentage values must add up to 100, or
            map must be empty if the Endpoint is to not accept any traffic at
            the moment. Key for model being deployed is "0". Should not be
            provided if traffic_percentage is provided.
        machine_type (str):
            Optional. The type of machine. Not specifying machine type will
            result in model to be deployed with automatic resources.
        min_replica_count (int):
            Optional. The minimum number of machine replicas this deployed
            model will be always deployed on. If traffic against it increases,
            it may dynamically be deployed onto more replicas, and as traffic
            decreases, some of these extra replicas may be freed.
        max_replica_count (int):
            Optional. The maximum number of replicas this deployed model may
            be deployed on when the traffic against it increases. If requested
            value is too large, the deployment will error, but if deployment
            succeeds then the ability to scale the model to that many replicas
            is guaranteed (barring service outages). If traffic against the
            deployed model increases beyond what its replicas at maximum may
            handle, a portion of the traffic will be dropped. If this value
            is not provided, the smaller value of min_replica_count or 1 will
            be used.
        accelerator_type (str):
            Optional. Hardware accelerator type. Must also set accelerator_count if used.
            One of ACCELERATOR_TYPE_UNSPECIFIED, NVIDIA_TESLA_K80, NVIDIA_TESLA_P100,
            NVIDIA_TESLA_V100, NVIDIA_TESLA_P4, NVIDIA_TESLA_T4
        accelerator_count (int):
            Optional. The number of accelerators to attach to a worker replica.
        service_account (str):
            The service account that the DeployedModel's container runs as. Specify the
            email address of the service account. If this service account is not
            specified, the container runs as a service account that doesn't have access
            to the resource project.
            Users deploying the Model must have the `iam.serviceAccounts.actAs`
            permission on this service account.
        explanation_metadata (explain.ExplanationMetadata):
            Optional. Metadata describing the Model's input and output for explanation.
            Both `explanation_metadata` and `explanation_parameters` must be
            passed together when used. For more details, see
            `Ref docs <http://tinyurl.com/1igh60kt>`
        explanation_parameters (explain.ExplanationParameters):
            Optional. Parameters to configure explaining for Model's predictions.
            For more details, see `Ref docs <http://tinyurl.com/1an4zake>`
        metadata (Sequence[Tuple[str, str]]):
            Optional. Strings which should be sent along with the request as
            metadata.
        encryption_spec_key_name (Optional[str]):
            Optional. The Cloud KMS resource identifier of the customer
            managed encryption key used to protect the model. Has the
            form:
            ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
            The key needs to be in the same region as where the compute
            resource is created.

            If set, this Model and all sub-resources of this Model will be secured by this key.

            Overrides encryption_spec_key_name set in aiplatform.init
        sync (bool):
            Whether to execute this method synchronously. If False, this method
            will be executed in concurrent Future and any downstream object will
            be immediately returned and synced when the Future has completed.
    Returns:
        endpoint ("Endpoint"):
            Endpoint with the deployed model.
    """

    # endpoint: Optional["Endpoint"] = (None,)
    deployed_model_display_name: Optional[str] = None
    traffic_percentage: Optional[int] = 0
    traffic_split: Optional[Dict[str, int]] = None
    machine_type: Optional[str] = None
    min_replica_count: int = 1
    max_replica_count: int = 1
    accelerator_type: Optional[str] = None
    accelerator_count: Optional[int] = None
    service_account: Optional[str] = None
    explanation_metadata: Optional[explain.ExplanationMetadata] = None
    explanation_parameters: Optional[explain.ExplanationParameters] = None
    metadata: Optional[Sequence[Tuple[str, str]]] = ()
    encryption_spec_key_name: Optional[str] = None


class DatasetTrainingDeployPipeline(managed_dataset_pipeline.ManagedDatasetPipeline):
    """
    Create a new Vertex AI managed dataset and trains an arbitrary AutoML or custom model
    """

    def __init__(
        self,
        name: str,
        managed_dataset: managed_dataset_pipeline.ManagedDataset,
        metric_key_for_comparison: Optional[str],
        deploy_info: Optional[DeployInfo],
        export_info: Optional[ExportInfo],
    ):
        super().__init__(name=name, managed_dataset=managed_dataset)
        self.metric_key_for_comparison = metric_key_for_comparison
        self.deploy_info = deploy_info
        self.export_info = export_info

    @abc.abstractmethod
    def create_training_op(self, project: str, dataset: Dataset) -> Callable:
        pass

    def get_metric_key_for_comparison(self) -> Optional[str]:
        return self.metric_key_for_comparison

    def create_pipeline_metric_comparison_op(
        self, project: str, location: str, pipeline_run_name: str, model: Model
    ) -> Callable:
        @component(
            packages_to_install=[
                "google-cloud-storage",
                "google-cloud-aiplatform",
                "pandas",
            ]
        )
        def pipeline_metric_comparison_op(
            project: str,
            location: str,
            pipeline_run_name: str,
            pipeline_name: str,
            model: Input[Model],
            pipeline_job_id_injected: str = "{{$.pipeline_job_uuid}}",
            metric_key: Optional[str] = None,
            is_greater_better: bool = True,
        ) -> bool:  # Return parameter.
            print(f"metric_key={metric_key}")
            print(f"pipeline_job_id_injected={pipeline_job_id_injected}")
            # If no target metric key is provided, return True
            if not metric_key:
                return True

            from google.cloud import aiplatform
            import pandas as pd

            pd.set_option("max_columns", None)  # show all cols
            pd.set_option("max_colwidth", None)  # show full width of showing cols
            pd.set_option(
                "expand_frame_repr", False
            )  # print cols side by side as it's supposed to be

            aiplatform.init(project=project, location=location)

            # Get all metrics from all pipeline runs
            df_metrics = aiplatform.get_pipeline_df(pipeline_name)

            # Check if pipeline_run_name is even in the data
            if pipeline_run_name not in df_metrics["run_name"].values:
                raise RuntimeError(
                    f"Current run id ({pipeline_run_name}) not found in metrics."
                )

            # Get current metric
            print(f"pipeline_run_name: {pipeline_run_name}")
            # print(f"metrics_for_all_pipelines: {df_metrics}")

            metrics_for_current_run = df_metrics[
                df_metrics["run_name"] == pipeline_run_name
            ]
            metric_value_for_current_run = metrics_for_current_run.to_dict("records")[
                0
            ].get(metric_key)

            print(f"metric_value_for_current_run: {metric_value_for_current_run}")

            if pd.isna(metric_value_for_current_run):
                raise RuntimeError(f"Metric value is not available.")

            # Get row for max metric
            if is_greater_better:
                metrics_for_best_run = df_metrics[
                    df_metrics[metric_key] == df_metrics[metric_key].max()
                ]

                metric_value_for_best_run = metrics_for_best_run.to_dict("records")[
                    0
                ].get(metric_key)

                return metric_value_for_current_run >= metric_value_for_best_run
            else:
                metrics_for_best_run = df_metrics[
                    df_metrics[metric_key] == df_metrics[metric_key].min()
                ]

                metric_value_for_best_run = metrics_for_best_run.to_dict("records")[
                    0
                ].get(metric_key)

                return metric_value_for_current_run <= metric_value_for_best_run

        metric_key = self.get_metric_key_for_comparison()

        return pipeline_metric_comparison_op(
            project=project,
            location=location,
            pipeline_run_name=pipeline_run_name,
            pipeline_name=self.name,
            model=model,
            metric_key=metric_key,
            is_greater_better=True,
        )

    def create_confusion_matrix_op(
        self, project: str, pipeline_root: str, model: Model
    ) -> Optional[Callable]:
        return None

    def create_classification_report_op(
        self, project: str, pipeline_root: str, model: Model
    ) -> Optional[Callable]:
        return None

    def create_model_history_op(
        self, project: str, pipeline_root: str, model: Model
    ) -> Optional[Callable]:
        return None

    def create_pipeline(
        self, project: str, location: str, pipeline_run_name: str, pipeline_root: str
    ) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.name, pipeline_root=pipeline_root)
        def pipeline():
            dataset_op = self.managed_dataset.as_kfp_op(project=project)

            # training_op = self.create_training_op(
            #     project=project, pipeline_root=pipeline_root, dataset=dataset_op.output
            # )
            training_op = importer(
                artifact_uri="807754018322382848",
                artifact_class=Model,
                reimport=False,
            )

            confusion_matrix_op = self.create_confusion_matrix_op(
                project=project,
                pipeline_root=pipeline_root,
                model=training_op.output,
            )

            classification_report_op = self.create_classification_report_op(
                project=project,
                pipeline_root=pipeline_root,
                model=training_op.output,
            )

            model_history_op = self.create_model_history_op(
                project=project,
                pipeline_root=pipeline_root,
                model=training_op.output,
            )

            pipeline_metric_comparison_op = self.create_pipeline_metric_comparison_op(
                project=project,
                location=location,
                pipeline_run_name=pipeline_run_name,
                model=training_op.output,
            ).after(confusion_matrix_op, classification_report_op, model_history_op)

            if self.deploy_info or self.export_info:
                with Condition(
                    pipeline_metric_comparison_op.output == "true",
                    name="post_train_decision",
                ):
                    if self.deploy_info:
                        deploy_op = gcc_aip.ModelDeployOp(
                            model=training_op.output,
                            # endpoint=self.deploy_info.endpoint,
                            deployed_model_display_name=self.deploy_info.deployed_model_display_name,
                            traffic_percentage=self.deploy_info.traffic_percentage,
                            traffic_split=self.deploy_info.traffic_split,
                            machine_type=self.deploy_info.machine_type,
                            min_replica_count=self.deploy_info.min_replica_count,
                            max_replica_count=self.deploy_info.max_replica_count,
                            accelerator_type=self.deploy_info.accelerator_type,
                            accelerator_count=self.deploy_info.accelerator_count,
                            service_account=self.deploy_info.service_account,
                            explanation_metadata=self.deploy_info.explanation_metadata,
                            explanation_parameters=self.deploy_info.explanation_parameters,
                            metadata=self.deploy_info.metadata,
                            encryption_spec_key_name=self.deploy_info.encryption_spec_key_name,
                        )

                    if self.export_info:
                        export_op = gcc_aip.ModelExportOp(
                            model=training_op.output,
                            export_format_id=self.export_info.export_format_id,
                            artifact_destination=self.export_info.artifact_destination,
                            image_destination=self.export_info.image_destination,
                        )

        return pipeline
