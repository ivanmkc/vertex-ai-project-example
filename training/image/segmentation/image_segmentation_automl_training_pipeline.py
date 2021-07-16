from training.common.import_training_deploy_pipeline import (
    ImportTrainingDeployPipeline,
)
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import Callable
from kfp.v2.dsl import Dataset


class ImageSegmentationAutoMLManagedDatasetPipeline(ImportTrainingDeployPipeline):
    id = "Image Segmentation AutoML"
    managed_dataset_uri: str = "aiplatform://v1/projects/1012616486416/locations/us-central1/datasets/7601275726536376320"
    should_deploy = True

    def create_training_op(
        self, project: str, pipeline_root: str, dataset: Dataset
    ) -> Callable:
        return gcc_aip.AutoMLImageTrainingJobRunOp(
            project=project,
            display_name="train-iris-automl-mbsdk-automl",
            prediction_type="classification",
            model_type="CLOUD",
            base_model=None,
            dataset=dataset,
            model_display_name="iris-classification-model-mbsdk",
            training_fraction_split=0.6,
            validation_fraction_split=0.2,
            test_fraction_split=0.2,
            budget_milli_node_hours=8000,
        )
