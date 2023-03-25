import boto3, json
from botocore.exceptions import ClientError

# VARIABLES
AWS_REGION = "eu-west-1"
AMI = "ami-0ac4506ffe115b721"
available_instances = ["p2.xlarge"]
SENDER = "fra.pochetti@gmail.com"
CHARSET = "UTF-8"

EC2 = boto3.client("ec2", region_name=AWS_REGION)
s3 = boto3.client('s3')
client = boto3.client('ses', region_name=AWS_REGION)

def spin_up_ec2(ec2_object, instance_type, script):

    
    instance = ec2_object.run_instances(
        ImageId=AMI,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        KeyName="FraDeepLearn",
        SecurityGroups=[
            "transfer-learning",
        ],
        IamInstanceProfile={"Name": "TransferLearningEC2Role"}, # this is needed to give EC2 the right permissions
        InstanceInitiatedShutdownBehavior='terminate', # make shutdown in script terminate ec2
        UserData=script # file to run on instance init.
        )

    return instance

def send_email_to_user(RECIPIENT, SUBJECT, BODY_TEXT, BODY_HTML):
    try:
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])
    
    return

def lambda_handler(event, context):
    
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = record['s3']['object']['key']
    
    response = s3.head_object(Bucket=bucket, Key=key)
    email = response['Metadata']['email']
    RECIPIENT = email
    style = response['Metadata']['style']

    print(bucket)
    print(key)
    print(email)
    print(style)

    ###################################
    # init_script IS THE CORE PART WHICH NEEDS TO BE FILLED! 
    ###################################
    init_script = f"""#!/bin/bash
cd /home/ubuntu
cd ml-prototypes
git pull
shutdown -h +30
# Necessary to being able to load all the necessary environment
sudo -i -u ubuntu bash <<-EOF
source ~/.bashrc
source activate tensorflow_p36
export PYTHONPATH=/home/ubuntu/ml-prototypes
python /home/ubuntu/ml-prototypes/prototypes/styletransfer/video_pipeline.py {key}
EOF
shutdown -h now
    """
    
    for INSTANCE_TYPE in available_instances:
        instance = spin_up_ec2(EC2, INSTANCE_TYPE, init_script)
        if instance:
            SUBJECT = "VisualNeurons.com - your GIF has been ingested!"

            BODY_TEXT = ("VisualNeurons.com - your GIF has been ingested! \r\n"
                         "The purpose of this email is to confirm that we successfully ingested your file"
                         "and that we are currently processing it."
                        )

            BODY_HTML = """
            <html>

            <head></head>

            <body>
                <h4>You just uploaded a GIF to S3. Congratulations!</h4>
                <p>The purpose of this email is to confirm that we successfully ingested your file and that we are currently processing it.
                    <br>
                    When our GPUs finish crunching your request, you will receive another email with the link to your Style-Transferred GIF.
                    <br>
                    This will happen in a hour or so. Thanks for your patience!
                </p>
            </body>

            </html>
                        """
            send_email_to_user(RECIPIENT, SUBJECT, BODY_TEXT, BODY_HTML)
            return

    if not instance:
        SUBJECT = "VisualNeurons.com - Can you please re-try later?"
        BODY_TEXT = ("VisualNeurons.com - Can you please re-try later? \r\n"
                     "Ouch... Sorry, it seems we ran out of artistic GPUs! Can you try again later?"
                    )
        
        BODY_HTML = """
        <html>
        <head></head>
        <body>
            <h4>:( We cannot process your request...</h4>
            <p>Ouch... Sorry, it seems we ran out of artistic GPUs! Can you try again later?
                <br>
                We have a limited number of EC2 instances we can run in parallel.
                <br>
                As now we are at full capacity, we cannot process your request. Sorry about this!
            </p>
        </body>
        </html>
                    """
        send_email_to_user(RECIPIENT, SUBJECT, BODY_TEXT, BODY_HTML)
        return