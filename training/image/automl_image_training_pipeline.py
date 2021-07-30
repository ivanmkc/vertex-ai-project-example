import dataclasses
import training.common.managed_dataset_pipeline as managed_dataset_pipeline

from training.common.dataset_training_deploy_pipeline import (
    DatasetTrainingDeployPipeline,
    DeployInfo,
    ExportInfo,
)
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import Callable
from kfp.v2.dsl import Dataset
from typing import Optional


@dataclasses.dataclass
class AutoMLImageTrainingInfo:
    # display_name: str
    prediction_type: str = "classification"
    multi_label: bool = False
    model_type: str = "CLOUD"
    # base_model: Optional[models.Model] = None
    # project: Optional[str] = None,
    # location: Optional[str] = None,
    # credentials: Optional[auth_credentials.Credentials] = None,
    training_encryption_spec_key_name: Optional[str] = None
    model_encryption_spec_key_name: Optional[str] = None
    training_fraction_split: float = 0.8
    validation_fraction_split: float = 0.1
    test_fraction_split: float = 0.1
    budget_milli_node_hours: int = 1000
    # model_display_name: Optional[str] = None
    disable_early_stopping: bool = False


class AutoMLImageManagedDatasetPipeline(DatasetTrainingDeployPipeline):
    def __init__(
        self,
        name: str,
        managed_dataset: managed_dataset_pipeline.ManagedDataset,
        training_info: AutoMLImageTrainingInfo,
        deploy_info: Optional[DeployInfo] = None,
        export_info: Optional[ExportInfo] = None,
    ):
        super().__init__(
            name=name,
            managed_dataset=managed_dataset,
            deploy_info=deploy_info,
            export_info=export_info,
        )

        self.training_info = training_info

    def create_training_op(
        self, project: str, pipeline_root: str, dataset: Dataset
    ) -> Callable:
        return gcc_aip.AutoMLImageTrainingJobRunOp(
            project=project,
            dataset=dataset,
            display_name=self.name,
            prediction_type=self.training_info.prediction_type,
            multi_label=self.training_info.multi_label,
            model_type=self.training_info.model_type,
            training_encryption_spec_key_name=self.training_info.training_encryption_spec_key_name,
            model_encryption_spec_key_name=self.training_info.model_encryption_spec_key_name,
            training_fraction_split=self.training_info.training_fraction_split,
            validation_fraction_split=self.training_info.validation_fraction_split,
            test_fraction_split=self.training_info.test_fraction_split,
            budget_milli_node_hours=self.training_info.budget_milli_node_hours,
            model_display_name=self.name,
            disable_early_stopping=self.training_info.disable_early_stopping,
        )
