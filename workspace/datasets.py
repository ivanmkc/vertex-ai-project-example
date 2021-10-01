from training.common.managed_dataset_pipeline import (
    ExistingManagedDataset,
    NewImageDataset,
)
from google.cloud import aiplatform


class datasets:
    class classification:
        flowers = NewImageDataset(
            display_name="flowers",
            gcs_source="gs://cloud-samples-data/vision/automl_classification/flowers/all_data_v2.csv",
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
            data_item_labels=None,
        )

        car_damage = NewImageDataset(
            display_name="car-damage",
            gcs_source="gs://car-damage-vertex/car-damage/annotations.jsonl",
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
            data_item_labels=None,
        )

    class object_detection:
        salads = NewImageDataset(
            display_name="salad",
            gcs_source="gs://cloud-samples-data/vision/salads.csv",
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box,
            data_item_labels=None,
        )

        mineral_plants = NewImageDataset(
            display_name="plants",
            gcs_source=[
                "gs://mineral-cloud-data/ivanmkc/object_detection/train/image_annotations.jsonl",
                "gs://mineral-cloud-data/ivanmkc/object_detection/validation/image_annotations.jsonl",
            ],
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box,
            data_item_labels=None,
        )

    class image_segmentation:
        mineral_leaves = NewImageDataset(
            display_name="leaves",
            gcs_source=[
                "gs://mineral-cloud-data/ivanmkc/image_segmentation/train/image_annotations.jsonl",
                "gs://mineral-cloud-data/ivanmkc/image_segmentation/validation/image_annotations.jsonl",
            ],
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.image_segmentation,
            data_item_labels=None,
        )

        anthropod_taxonomy = NewImageDataset(
            display_name="Anthropods",
            gcs_source=[
                "gs://mlops-object-detection/ArTaxOr/ArTaxOr.csv",
            ],
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.image_segmentation,
            data_item_labels=None,
        )
