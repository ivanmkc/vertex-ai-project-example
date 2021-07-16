import abc
from pipelines.pipeline import Pipeline


class ManagedDatasetPipeline(Pipeline):
    """
    Uses existing managed dataset and builds an arbitrary pipeline
    """

    @property
    @abc.abstractmethod
    def managed_dataset_uri(self) -> str:
        pass
