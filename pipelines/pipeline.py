from typing import Any, Callable
import abc


class Pipeline(abc.ABC):
    """
    An arbitrary pipeline
    """

    @property
    @abc.abstractmethod
    def id(self) -> str:  # Must be unique
        pass

    @abc.abstractmethod
    def create_pipeline(self, project: str, pipeline_root: str) -> Callable[..., Any]:
        pass

    # TODO: This should not be in this class
    @property
    def run_frequency(self) -> str:
        # Frequency in unix-cron format (https://cloud.google.com/scheduler/docs/configuring/cron-job-schedules)
        # Default is: Every Monday at 09:00
        return "0 9 * * 1"

    @property
    def pipeline_name(self) -> str:
        # TODO: Must match the regular expression "^[a-z0-9][a-z0-9-]{0,127}$"
        return self.id.replace(" ", "-").lower()
