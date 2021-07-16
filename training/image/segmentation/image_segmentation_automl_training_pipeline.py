from training.common.dataset_training_deploy_pipeline import (
    DatasetTrainingDeployPipeline,
)
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import Callable
from kfp.v2.dsl import Dataset
from google.cloud import aiplatform
from typing import Optional
import abc


class ImageSegmentationAutoMLManagedDatasetPipeline(DatasetTrainingDeployPipeline):
    id = "Image Segmentation AutoML"
    dataset_display_name = "Image Segmentation Dataset"
    dataset_gcs_source = "gs://ucaip-test-us-central1/dataset/isg_data.jsonl"
    dataset_import_schema_uri = (
        aiplatform.schema.dataset.ioformat.image.image_segmentation
    )
    dataset_data_item_labels = None

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
