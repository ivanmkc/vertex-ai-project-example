from training.conditional_deployment_training_pipeline import (
    ConditionalDeploymentTrainingPipeline,
)


class ImageClassificationTrainingPipeline(ConditionalDeploymentTrainingPipeline):
    thresholds_dict = {"auPrc": 0.95}
