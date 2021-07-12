from typing import Any, Callable
import abc


class TrainingPipeline(abc.ABC):
    @property
    @abc.abstractmethod
    def id(self) -> str:  # Must be unique
        pass

    @property
    @abc.abstractmethod
    def annotation_dataset_id(self) -> str:
        pass

    @abc.abstractmethod
    def create_pipeline(
        self, project_id: str, pipeline_root: str
    ) -> Callable[..., Any]:
        pass

    @property
    def pipeline_name(self) -> str:
        # TODO: Must match the regular expression "^[a-z0-9][a-z0-9-]{0,127}$"
        return self.id.replace(" ", "-").lower()
