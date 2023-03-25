import json, boto3, botocore

# This file is to store a lambda handler which gets triggered by an Amazon API Gateway POST method.
# A lambda handler is a serverless python function executed each time the trigger is pulled.
# In this specific case, the function does the following:
# 1) spins up a p2.xlarge EC2 instance with a custom deep learning AMI
# 2) clones our transfer learning repo
# 3) launches the server side websocket
# 4) kills the EC2 after 10 minutes
# 5) returns to the browser the pulic DNS of the spinned up EC2 instance

REGION = "eu-west-1"  # region to launch instance.
AMI = "ami-0ac4506ffe115b721"  # our custom AMI
available_instances = [
    "p2.xlarge",
]

EC2 = boto3.client("ec2", region_name=REGION)


def format_response(message, status_code):
    return {
        "statusCode": str(status_code),
        "body": json.dumps(message),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }


def spin_up_ec2(ec2_object, instance_type, script):

    try:
        instance = ec2_object.run_instances(
            ImageId=AMI,
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            KeyName="GabKP",
            SecurityGroups=[
                "transfer-learning",
            ],
            InstanceInitiatedShutdownBehavior='terminate', # make shutdown in script terminate ec2
            UserData=script # file to run on instance init.
        )

    except botocore.exceptions.ClientError:
        instance = None

    return instance


def lambda_handler(event, context):

    output = '{{"id":"{}", "type":"{}", "dns":"{}"}}'
    # Important to have no space    
    init_script = f"""#!/bin/bash
cd /home/ubuntu
cd ml-prototypes
git pull
shutdown -h +7
# Necessary to being able to load all the necessary environment
sudo -i -u ubuntu bash <<-EOF
source ~/.bashrc
source activate tensorflow_p36
export PYTHONPATH=/home/ubuntu/ml-prototypes
python -m prototypes.styletransfer.app --address=0.0.0.0 --port=8000
EOF
shutdown -h now
    """
    

    for INSTANCE_TYPE in available_instances:
        instance = spin_up_ec2(EC2, INSTANCE_TYPE, init_script)
        if instance:
            instance_id = instance["Instances"][0]["InstanceId"]
            break

    if not instance:
        return format_response(
            output.format('limit exceeded', 'limit exceeded', 'limit exceeded'), 200
        )

    inst = boto3.resource("ec2", region_name=REGION).Instance(instance_id)
    inst.wait_until_running()
    inst.load()
    public_dns = inst.public_dns_name

    return format_response(
        output.format(instance_id, INSTANCE_TYPE, public_dns), 200
    )
