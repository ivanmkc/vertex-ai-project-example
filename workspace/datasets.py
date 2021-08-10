from training.common.managed_dataset_pipeline import (
    ExistingManagedDataset,
    NewManagedDataset,
)
from google.cloud import aiplatform


class datasets:
    class classification:
        flowers = ExistingManagedDataset(
            dataset_uri="aiplatform://v1/projects/1012616486416/locations/us-central1/datasets/7601275726536376320"
        )

    class object_detection:
        salads = NewManagedDataset(
            display_name="Object Detection Dataset",
            gcs_source="gs://cloud-samples-data/vision/salads.csv",
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box,
            data_item_labels=None,
        )

        mineral_plants = NewManagedDataset(
            display_name="Object Detection Dataset",
            gcs_source=[
                "gs://mineral-cloud-data/ivanmkc/object_detection/train/image_annotations.jsonl",
                "gs://mineral-cloud-data/ivanmkc/object_detection/validation/image_annotations.jsonl",
            ],
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box,
            data_item_labels=None,
        )
