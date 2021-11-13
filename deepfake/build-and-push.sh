#!/usr/bin/env bash

chmod +x code/train.py

aws ecr describe-repositories --repository-names "custom-images" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "custom-images" > /dev/null
fi

region="eu-west-1"
account=$(aws sts get-caller-identity --query Account --output text)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/custom-images:pv"

docker rmi -f $(docker images -a -q)
docker build -t pv .    
docker tag pv:latest ${fullname}

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${region}.amazonaws.com
aws ecr get-login-password | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com
aws ecr batch-delete-image \
     --repository-name custom-images \
     --image-ids imageTag=pv

docker push ${fullname}