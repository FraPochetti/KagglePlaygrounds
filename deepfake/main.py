import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator

bucket = 's3://deepfake-fra/'

role = "arn:aws:iam::257446244580:role/sagemaker-icevision"

estimator = Estimator(
    role=role,
    image_uri="257446244580.dkr.ecr.eu-west-1.amazonaws.com/custom-images:pv",
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    hyperparameters={
        'epochs': 10,
        'lr': 0.1,
    },
    environment={"WANDB_API_KEY": "aeaed779ca569f4e5336ddddd1d12b234f338d98"},)

estimator.fit({
    'train': bucket,
})