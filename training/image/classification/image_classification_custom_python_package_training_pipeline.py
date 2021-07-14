from image_classification_training_pipeline import (
    ImageClassificationTrainingPipeline,
)
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import Callable
from kfp.v2.dsl import Dataset


class ImageClassificationCustomPythonPackageTrainingPipeline(
    ImageClassificationTrainingPipeline
):
    id = "Image Classification Custom Python Package"
    annotation_dataset_uri: str = "aiplatform://v1/projects/1012616486416/locations/us-central1/datasets/7601275726536376320"

    def create_training_op(
        self, project: str, pipeline_root: str, dataset: Dataset
    ) -> Callable:
        return gcc_aip.CustomPythonPackageTrainingJobRunOp()
