from typing import Any, Callable
import abc

from google_cloud_pipeline_components import aiplatform as gcc_aip
import kfp
from kfp.v2.dsl import (
    Dataset,
)
import training.common.managed_dataset_pipeline as managed_dataset_pipeline


class DatasetTrainingDeployPipeline(managed_dataset_pipeline.ManagedDatasetPipeline):
    """
    Create a new Vertex AI managed dataset and trains an arbitrary AutoML or custom model
    """

    def __init__(
        self,
        name: str,
        managed_dataset: managed_dataset_pipeline.ManagedDataset,
        should_deploy: bool,
    ):
        super().__init__(name=name, managed_dataset=managed_dataset)
        self.should_deploy = should_deploy

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

            if self.should_deploy:
                deploy_op = gcc_aip.ModelDeployOp(
                    model=training_op.outputs["model"],
                )

        return pipeline
