import abc

from pipelines_folder.pipeline import Pipeline
import dataclasses
from typing import Callable, Optional, Sequence, Union
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2.dsl import Dataset, importer


class ManagedDataset(abc.ABC):
    @abc.abstractmethod
    def as_kfp_op(self, project: str) -> Callable:
        pass


@dataclasses.dataclass
class ExistingManagedDataset(ManagedDataset):
    dataset_uri: str

    def as_kfp_op(self, project: str) -> Callable:
        return importer(
            artifact_uri=self.dataset_uri,
            artifact_class=Dataset,
            reimport=False,
        )


@dataclasses.dataclass
class NewImageDataset(ManagedDataset):
    display_name: str
    gcs_source: Union[str, Sequence[str]]
    import_schema_uri: Optional[str]
    data_item_labels: Optional[dict]

    def as_kfp_op(self, project: str) -> Callable:
        return gcc_aip.ImageDatasetCreateOp(
            display_name=self.display_name,
            gcs_source=self.gcs_source,
            import_schema_uri=self.import_schema_uri,
            data_item_labels=self.data_item_labels,
            project=project,
        )


@dataclasses.dataclass
class NewTabularDataset(ManagedDataset):
    display_name: str
    gcs_source: Union[str, Sequence[str]]
    bq_source: Optional[str]
    import_schema_uri: Optional[str]
    data_item_labels: Optional[dict]

    def as_kfp_op(self, project: str) -> Callable:
        return gcc_aip.TabularDatasetCreateOp(
            display_name=self.display_name,
            gcs_source=self.gcs_source,
            bq_source=self.bq_source,
            import_schema_uri=self.import_schema_uri,
            data_item_labels=self.data_item_labels,
            project=project,
        )


class ManagedDatasetPipeline(Pipeline):
    """
    Uses existing managed dataset and builds an arbitrary pipeline
    """

    def __init__(self, name: str, managed_dataset: Optional[ManagedDataset]):
        super().__init__(name=name)
        self.managed_dataset = managed_dataset
