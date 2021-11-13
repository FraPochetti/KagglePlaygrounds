import argparse
from sagemaker.estimator import Estimator

bucket = 's3://deepfake-fra/'
role = "IAM role with full SM and S3 permissions"

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    estimator = Estimator(
        role=role,
        image_uri="ECR image uri" if not args.local else "pv",
        instance_count=1,
        instance_type="ml.p3.2xlarge" if not args.local else "local_gpu",
        hyperparameters={
            'epochs': 6,
            'lr': 0.001,
            'backbone': 'x3d_xs',
            'batch-size': 16,
            'milestone': 3,
        },
        environment={"WANDB_API_KEY": "my api key"},)

    estimator.fit({
        'train': bucket if not args.local else 'file:///home/ubuntu/data/faces_s3',
    })