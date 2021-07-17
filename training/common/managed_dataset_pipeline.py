import abc
from pipelines.pipeline import Pipeline
import dataclasses
from typing import Callable, Optional, Sequence, Union
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.dsl import importer_node
from kfp.v2.dsl import (
    Dataset,
)


class ManagedDataset(abc.ABC):
    @abc.abstractmethod
    def as_kfp_op(self, project: str) -> Callable:
        pass


@dataclasses.dataclass
class ExistingManagedDataset(ManagedDataset):
    dataset_uri: str

    def as_kfp_op(self, project: str) -> Callable:
        return importer_node.importer(
            artifact_uri=self.dataset_uri,
            artifact_class=Dataset,
            reimport=False,
        )


@dataclasses.dataclass
class NewManagedDataset(ManagedDataset):
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


class ManagedDatasetPipeline(Pipeline):
    """
    Uses existing managed dataset and builds an arbitrary pipeline
    """

    @property
    @abc.abstractmethod
    def managed_dataset(self) -> Optional[ManagedDataset]:
        pass
