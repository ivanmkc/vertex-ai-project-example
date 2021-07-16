from typing import Any, Callable, Optional, Sequence, Union
import abc

from pipelines.pipeline import Pipeline
from google_cloud_pipeline_components import aiplatform as gcc_aip
import kfp
from kfp.v2.dsl import (
    Dataset,
)


class DatasetTrainingDeployPipeline(Pipeline):
    """
    Create a new Vertex AI managed dataset and trains an arbitrary AutoML or custom model
    """

    should_deploy: bool = False

    @property
    @abc.abstractmethod
    def dataset_display_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def dataset_gcs_source(self) -> Union[str, Sequence[str]]:
        pass

    @property
    @abc.abstractmethod
    def dataset_import_schema_uri(self) -> Optional[str]:
        pass

    @property
    @abc.abstractmethod
    def dataset_data_item_labels(self) -> Optional[dict]:
        pass

    @abc.abstractmethod
    def create_training_op(self, project: str, dataset: Dataset) -> Callable:
        pass

    def create_pipeline(self, project: str, pipeline_root: str) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.pipeline_name, pipeline_root=pipeline_root)
        def pipeline():
            importer = gcc_aip.ImageDatasetCreateOp(
                display_name=self.dataset_display_name,
                gcs_source=self.dataset_gcs_source,
                import_schema_uri=self.dataset_import_schema_uri,
                data_item_labels=self.dataset_data_item_labels,
                project=project,
            )

            training_op = self.create_training_op(
                project=project, pipeline_root=pipeline_root, dataset=importer.output
            )

            if self.should_deploy:
                deploy_op = gcc_aip.ModelDeployOp(
                    model=training_op.outputs["model"],
                )

        return pipeline
