from typing import Any, Callable, Optional, Sequence, Union
import abc

from pipelines.pipeline import Pipeline
from google_cloud_pipeline_components import aiplatform as gcc_aip
import kfp
from kfp.v2.dsl import (
    Dataset,
)
from training.common.managed_dataset_pipeline import ManagedDatasetPipeline


class DatasetTrainingDeployPipeline(ManagedDatasetPipeline):
    """
    Create a new Vertex AI managed dataset and trains an arbitrary AutoML or custom model
    """

    should_deploy: bool = False

    @abc.abstractmethod
    def create_training_op(self, project: str, dataset: Dataset) -> Callable:
        pass

    def create_pipeline(self, project: str, pipeline_root: str) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.pipeline_name, pipeline_root=pipeline_root)
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
