from training.common.managed_dataset_pipeline import (
    ExistingManagedDataset,
    NewImageDataset,
)
from google.cloud import aiplatform


class datasets:
    class classification:
        flowers = ExistingManagedDataset(
            dataset_uri="aiplatform://v1/projects/1012616486416/locations/us-central1/datasets/7601275726536376320"
        )

        car_damage = NewImageDataset(
            display_name="Car Damage Dataset 4",
            gcs_source="gs://car-damage-vertex/car-damage/annotations.jsonl",
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
            data_item_labels=None,
        )

    class object_detection:
        salads = NewImageDataset(
            display_name="Object Detection Dataset",
            gcs_source="gs://cloud-samples-data/vision/salads.csv",
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box,
            data_item_labels=None,
        )

        mineral_plants = NewImageDataset(
            display_name="Plants",
            gcs_source=[
                "gs://mineral-cloud-data/ivanmkc/object_detection/train/image_annotations.jsonl",
                "gs://mineral-cloud-data/ivanmkc/object_detection/validation/image_annotations.jsonl",
            ],
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box,
            data_item_labels=None,
        )

    class image_segmentation:
        mineral_leaves = NewImageDataset(
            display_name="Leaves",
            gcs_source=[
                "gs://mineral-cloud-data/ivanmkc/image_segmentation/train/image_annotations.jsonl",
                "gs://mineral-cloud-data/ivanmkc/image_segmentation/validation/image_annotations.jsonl",
            ],
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.image_segmentation,
            data_item_labels=None,
        )
