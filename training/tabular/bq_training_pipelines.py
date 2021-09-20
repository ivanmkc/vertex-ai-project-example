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
from typing import Dict, List, Optional

import kfp
from pipelines_folder.pipeline import Pipeline
from typing import Any, Callable
from google_cloud_pipeline_components import aiplatform as gcc_aip
from components.bigquery.bqml import training, evaluation, prediction
from components.bigquery import data_processing


class BQMLTrainingPipeline(Pipeline):
    """
    Runs a BQ query, creates a model and generates evaluations
    """

    def __init__(
        self,
        name: str,
        query_training: str,  # Full query
        query_statement_evaluation: str,
        query_statement_prediction: str,
        prediction_destination_table_id: str = "",
    ):
        super().__init__(name=name)

        self.query_training = query_training
        self.query_statement_evaluation = query_statement_evaluation
        self.query_statement_prediction = query_statement_prediction
        self.prediction_destination_table_id = prediction_destination_table_id

    def create_pipeline(
        self, project: str, pipeline_root: str, location: str
    ) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.name, pipeline_root=pipeline_root)
        def pipeline():
            create_model_op = training.bqml_create_model_op(
                project=project,
                location=location,
                query=self.query_training,
            )

            create_evaluation_op = evaluation.bqml_create_evaluation_op(
                project=project,
                location=location,
                model=create_model_op.outputs["model"],
                query_statement=self.query_statement_evaluation,
            )

            create_confusion_matrix_op = evaluation.bqml_create_confusion_matrix(
                project=project,
                location=location,
                model=create_model_op.outputs["model"],
                query_statement=self.query_statement_evaluation,
            )

            create_roc_curve_op = evaluation.bqml_create_roc_curve(
                project=project,
                location=location,
                model=create_model_op.outputs["model"],
                query_statement=self.query_statement_evaluation,
            )

            predict_op = prediction.bqml_predict(
                project=project,
                location=location,
                model=create_model_op.outputs["model"],
                query_statement=self.query_statement_prediction,
                destination_table_id=self.prediction_destination_table_id,
            )

            # TODO: Model import
            # TODO: Model export
            # TODO: Model explain
            # TODO: Export to table

            # if self.destination_csv_uri:
            #     export_to_csv_op = other.export_to_csv(
            #         project=project,
            #         location=location,
            #         source_table_id=predict_op.outputs["destination_table_id"],
            #         source_table_location=location,
            #         destination_csv_uri=self.destination_csv_uri,
            #     )

        return pipeline


class BQQueryAutoMLPipeline(Pipeline):
    """
    Runs a BQ query, creates a model and generates evaluations
    """

    def __init__(
        self,
        name: str,
        query: str,
        optimization_prediction_type: str,
        target_column: str,
        source_table_location: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.query = query
        self.optimization_prediction_type = optimization_prediction_type
        self.target_column = target_column
        self.source_table_location = source_table_location

    def create_pipeline(
        self, project: str, pipeline_root: str, location: str
    ) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.name, pipeline_root=pipeline_root)
        def pipeline():
            query_op = data_processing.bq_query(
                project=project,
                location=location,
                query=self.query,
            )

            dataset_op = gcc_aip.TabularDatasetCreateOp(
                display_name=self.name,
                gcs_source=None,
                bq_source=query_op.outputs["destination_table_id"],
                project=project,
                location=location,
            )

            training_op = gcc_aip.AutoMLTabularTrainingJobRunOp(
                display_name=self.name,
                optimization_prediction_type=self.optimization_prediction_type,
                dataset=dataset_op.output,
                target_column=self.target_column,
                project=project,
                location=location,
            )

        return pipeline
