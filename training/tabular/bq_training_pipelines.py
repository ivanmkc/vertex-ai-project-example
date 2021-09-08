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
import components.bigquery


class BQMLTrainingPipeline(Pipeline):
    """
    Runs a BQ query, creates a model and generates evaluations
    """

    def __init__(
        self,
        name: str,
        model_name: str,  # e.g. bqml_tutorial.sample_model. Dataset has to exist
        create_mode: components.bigquery.BQMLModelCreateMode,
        query_statement_training: str,
        query_statement_evaluation: str,
        query_statement_prediction: str,
        prediction_destination_table_id: str,
    ):
        super().__init__(name=name)

        self.create_mode = create_mode
        self.model_name = model_name
        self.query_statement_training = query_statement_training
        self.query_statement_evaluation = query_statement_evaluation
        self.query_statement_prediction = query_statement_prediction
        self.prediction_destination_table_id = prediction_destination_table_id

    def create_pipeline(
        self, project: str, pipeline_root: str, location: str
    ) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.name, pipeline_root=pipeline_root)
        def pipeline():
            create_model_op = components.bigquery.create_model(
                project=project,
                location=location,
                create_mode_str=self.create_mode.value,
                model_name=self.model_name,
                create_model_options_str=components.bigquery.BQMLCreateModelOptions().to_sql(),
                query_statement=self.query_statement_training,
            )

            create_evaluation_op = components.bigquery.create_evaluation(
                project=project,
                location=location,
                model_name=create_model_op.output,
                query_statement=self.query_statement_evaluation,
            )

            create_confusion_matrix_op = components.bigquery.create_confusion_matrix(
                project=project,
                location=location,
                model_name=create_model_op.output,
                query_statement=self.query_statement_evaluation,
            )

            create_roc_curve_op = components.bigquery.create_roc_curve(
                project=project,
                location=location,
                model_name=create_model_op.output,
                query_statement=self.query_statement_evaluation,
            )

            predict_op = components.bigquery.predict(
                project=project,
                location=location,
                model_name=create_model_op.output,
                query_statement=self.query_statement_prediction,
                destination_table_id=self.prediction_destination_table_id,
            )

        return pipeline


class BQQueryAutoMLPipeline(Pipeline):
    """
    Runs a BQ query, creates a model and generates evaluations
    """

    def __init__(self, name: str, query: str, bq_output_table_id: str):
        super().__init__(name=name)

        self.query = query
        self.bq_output_table_id = bq_output_table_id

    def create_pipeline(
        self, project: str, pipeline_root: str, location: str
    ) -> Callable[..., Any]:
        @kfp.dsl.pipeline(
            name=self.name, pipeline_root=pipeline_root, location=location
        )
        def pipeline():
            query_op = components.bigquery.query(
                query=self.query,
                bq_output_table_id=self.bq_output_table_id,
                project=project,
            )

            dataset_op = gcc_aip.TabularDatasetCreateOp(
                display_name=self.name,
                gcs_source=None,
                bq_source=query_op.output,
                project=project,
            )

            training_op = gcc_aip.AutoMLTabularTrainingJobRunOp(
                display_name=self.name,
                optimization_prediction_type="classification",
                dataset=dataset_op.output,
                target_column="info",
                project=project,
            )

        return pipeline
