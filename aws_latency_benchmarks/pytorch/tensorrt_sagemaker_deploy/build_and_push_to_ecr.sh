TAG="trt_pytorch"
IMAGE_URI="custom-images"
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
REGION="eu-west-1"

docker build -t $TAG .
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker tag $TAG:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_URI:$TAG
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_URI:$TAG