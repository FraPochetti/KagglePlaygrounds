import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator

bucket = 's3://deepfake-fra/'

role = "myrole"

estimator = Estimator(
    role=role,
    image_uri="myimage",
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    hyperparameters={
        'epochs': 10,
        'lr': 0.1,
    },
    environment={"WANDB_API_KEY": "mykey"},)

estimator.fit({
    'train': bucket,
})