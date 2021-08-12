# Copyright 2021 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List

import kfp
from kfp.components import OutputPath
from kfp.v2.dsl import Output, Dataset, component
from pipelines_folder.pipeline import Pipeline
from typing import Any, Callable, Optional
from google_cloud_pipeline_components import aiplatform as gcc_aip
from components.bigquery import bigquery


class BQMLTrainingPipeline(Pipeline):
    """
    Runs a BQ query, creates a model and generates evaluations
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    def create_pipeline(self, project: str, pipeline_root: str) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.name, pipeline_root=pipeline_root)
        def pipeline():
            select_query = "TODO"
            create_model_op = bigquery.create_model(
                create_mode=bigquery.BQMLModelCreateMode.CREATE_MODEL.value,
                model_name="test",
                create_model_options="bigquery.BQMLCreateModelOptions()",
                select_query=select_query,
            )

            roc_curve_op = bigquery.create_roc_curve(
                model_name=create_model_op.output,
                table_name="",
                thresholds_str="",
            )

        return pipeline


class BQQueryAutoMLPipeline(Pipeline):
    """
    Runs a BQ query, creates a model and generates evaluations
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    def create_pipeline(self, project: str, pipeline_root: str) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.name, pipeline_root=pipeline_root)
        def pipeline():
            query_op = bigquery.query(query="SELECT", output_uri="gcs://asdf")

            roc_curve_op = gcc_aip.AutoMLTabularTrainingJobRunOp(
                display_name=self.name,
                optimization_prediction_type="classification",
                dataset=dataset_create_op.output,
                target_column="info",
            )

        return pipeline
