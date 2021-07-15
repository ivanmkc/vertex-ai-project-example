from typing import Any, Callable
import abc
import training_pipeline
from kfp.dsl import importer_node

import training_pipeline
from google_cloud_pipeline_components import aiplatform as gcc_aip
import kfp
from kfp.v2.dsl import (
    Dataset,
)


class ImportAndTrainingPipeline(training_pipeline.TrainingPipeline):
    """
    Ingests an existing Vertex AI managed dataset and trains an arbitrary AutoML or custom model
    """

    @abc.abstractmethod
    def create_training_op(self, project: str, dataset: Dataset) -> Callable:
        pass

    def create_pipeline(self, project: str, pipeline_root: str) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.pipeline_name, pipeline_root=pipeline_root)
        def pipeline():
            importer = importer_node.importer(
                artifact_uri=self.annotation_dataset_uri,
                artifact_class=Dataset,
                reimport=False,
            )

            training_op = self.create_training_op(
                project=project, pipeline_root=pipeline_root, dataset=importer.output
            )

        return pipeline
