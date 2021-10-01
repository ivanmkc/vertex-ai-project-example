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
from kfp.v2.dsl import component
from pipelines_folder.pipeline import Pipeline
from typing import Any, Callable
from google_cloud_pipeline_components import aiplatform as gcc_aip
from components.bigquery.bqml import training, evaluation, import_export, prediction
from components.bigquery import data_processing
from training.common.custom_python_package_training_pipeline import (
    CustomPythonPackageTrainingInfo,
)


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
        self, project: str, pipeline_root: str, pipeline_run_name: str, location: str
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
        training_location: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.query = query
        self.training_location = training_location
        self.optimization_prediction_type = optimization_prediction_type
        self.target_column = target_column

    def create_pipeline(
        self, project: str, pipeline_root: str, pipeline_run_name: str, location: str
    ) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.name, pipeline_root=pipeline_root)
        def pipeline():
            query_op = data_processing.bq_query(
                project=project,
                location=location,
                query=self.query,
            )

            # Op to package args into an arg list
            @component(packages_to_install=[])
            def add_bq_prefix_op(bq_uri: str) -> str:
                return f"bq://{bq_uri}"

            add_bq_prefix_op = add_bq_prefix_op(
                bq_uri=query_op.outputs["destination_table_id"]
            )

            dataset_op = gcc_aip.TabularDatasetCreateOp(
                display_name=self.name,
                gcs_source=None,
                bq_source=add_bq_prefix_op.output,
                project=project,
                location=self.training_location,
            )

            training_op = gcc_aip.AutoMLTabularTrainingJobRunOp(
                display_name=self.name,
                optimization_prediction_type=self.optimization_prediction_type,
                dataset=dataset_op.output,
                target_column=self.target_column,
                project=project,
                location=self.training_location,
            )

        return pipeline


class BQQueryCustomPipeline(Pipeline):
    """
    Runs a BQ query, creates a model and generates evaluations
    """

    def __init__(
        self,
        name: str,
        query: str,
        training_image_uri: str,
        training_location: Optional[str],
        export_format: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.query = query
        self.training_image_uri = training_image_uri
        self.training_location = training_location
        self.export_format = export_format

    def create_pipeline(
        self, project: str, pipeline_root: str, pipeline_run_name: str, location: str
    ) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.name, pipeline_root=pipeline_root)
        def pipeline():
            query_op = data_processing.bq_query(
                project=project,
                location=location,
                query=self.query,
            )

            export_op = data_processing.export(
                project=project,
                location=location,
                source=query_op.outputs["destination_table_id"],
                destination_uri=pipeline_root,
                destination_format=self.export_format,
            )

            # Op to package args into an arg list
            @component(packages_to_install=[])
            def create_args_op(export_destination_uri: str) -> List[str]:
                return [
                    "--data_uri",
                    export_destination_uri,
                ]

            args_op = create_args_op(
                export_destination_uri=export_op.outputs["destination_uri"]
            )

            package_gcs_uri = "gs://package_uri"
            python_module_name = "training_task.py"

            training_op = gcc_aip.CustomPythonPackageTrainingJobRunOp(
                display_name=self.name,
                python_package_gcs_uri=package_gcs_uri,
                python_module_name=python_module_name,
                project=project,
                location=self.training_location or location,
                args=args_op.output,
                container_uri=self.training_image_uri,
            )

        return pipeline


class BQMLExportToVertexAI(Pipeline):
    """
    Runs a BQ query, creates a model and generates evaluations
    """

    def __init__(
        self,
        name: str,
        query_training: str,
    ):
        super().__init__(name=name)

        self.query_training = query_training

    def create_pipeline(
        self, project: str, pipeline_root: str, pipeline_run_name: str, location: str
    ) -> Callable[..., Any]:
        @kfp.dsl.pipeline(name=self.name, pipeline_root=pipeline_root)
        def pipeline():
            create_model_op = training.bqml_create_model_op(
                project=project,
                location=location,
                query=self.query_training,
            )

            export_op = import_export.bqml_export_model(
                project=project,
                location=location,
                model=create_model_op.outputs["model"],
                model_destination_path=f"{pipeline_root}/exported_model",
            )

            import_op = gcc_aip.ModelUploadOp(
                project=project,
                location=location,
                display_name=self.name,
                serving_container_image_uri="tensorflow/serving",
                artifact_uri=export_op.outputs["model_destination_path"],
            )

        return pipeline
