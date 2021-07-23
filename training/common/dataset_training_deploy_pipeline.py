from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import abc
import dataclasses

from google_cloud_pipeline_components import aiplatform as gcc_aip
import kfp
from kfp.v2.dsl import (
    Dataset,
)
import training.common.managed_dataset_pipeline as managed_dataset_pipeline
from google.cloud.aiplatform import explain


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
        should_deploy: bool,
        deploy_info: DeployInfo,
    ):
        super().__init__(name=name, managed_dataset=managed_dataset)
        self.should_deploy = should_deploy
        self.deploy_info = deploy_info

    @abc.abstractmethod
    def create_training_op(self, project: str, dataset: Dataset) -> Callable:
        pass

    def create_pipeline(self, project: str, pipeline_root: str) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.name, pipeline_root=pipeline_root)
        def pipeline():
            dataset_op = self.managed_dataset.as_kfp_op(project=project)

            training_op = self.create_training_op(
                project=project, pipeline_root=pipeline_root, dataset=dataset_op.output
            )

            # TODO: Add optional evaluation task

            if self.should_deploy:
                deploy_op = gcc_aip.ModelDeployOp(
                    model=training_op.outputs["model"],
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

        return pipeline
