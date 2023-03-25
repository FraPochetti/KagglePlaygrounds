import json, boto3, random, time
from boto3.dynamodb.conditions import Key

REGION = "eu-west-1"
INSTANCE_ID = 'i-006f42d20adf236c8'
ec2_client = boto3.client("ec2", region_name=REGION)
ec2_resource = boto3.resource("ec2", region_name=REGION)
dynamo_resource = boto3.resource('dynamodb', region_name=REGION)
ssm_client = boto3.client('ssm', region_name=REGION)

def execute_commands_on_linux_instances(client, commands, instance_ids):
    """Runs commands on remote linux instances
    :param client: a boto/boto3 ssm client
    :param commands: a list of strings, each one a command to execute on the instances
    :param instance_ids: a list of instance_id strings, of the instances on which to execute the command
    :return: the response from the send_command function (check the boto3 docs for ssm client.send_command() )
    """

    resp = client.send_command(
        DocumentName="AWS-RunShellScript", # One of AWS' preconfigured documents
        Parameters={'commands': commands},
        InstanceIds=instance_ids,
    )
    return resp

def format_response(message, status_code):
    return {
        "statusCode": str(status_code),
        "body": json.dumps(message),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }

def lambda_handler(event, context):
    
    body = json.loads(event['body'])
    prompt = body['prompt']
    samples = int(body['num_samples'])
    words = int(body['length'])
    temperature = float(body['temperature'])
    nucleus = float(body['top_p'])
    topn = int(body['top_k'])
    
    status = ec2_client.describe_instance_status(InstanceIds=[INSTANCE_ID])
    if len(status['InstanceStatuses']) == 0: ec2_client.start_instances(InstanceIds=[INSTANCE_ID])
    
    instance = ec2_resource.Instance(INSTANCE_ID)
    instance.wait_until_running()
    waiter = ec2_client.get_waiter('instance_status_ok')
    waiter.wait(InstanceIds=[INSTANCE_ID])
    
    dynamoid = random.randint(1, 1000000)
    
    commands = ["cd /home/ubuntu",
                "shutdown -h +30",
                "sudo -i -u ubuntu bash <<-EOF",
                "source ~/.bashrc",
                "source activate pytorch_p36",
                f"python huggingface.py --prompt=\"{prompt}\" --dynamoid={dynamoid} --num_samples={samples} --length={words} --temperature={temperature} --top_p={nucleus} --top_k={topn}"]
    
    print(commands)
    execute_commands_on_linux_instances(ssm_client, commands, [INSTANCE_ID])
    
    table = dynamo_resource.Table('huggingface')
    
    timeout = time.time() + 60*2   # 2 minutes from now
    while True:
        resp = table.query(KeyConditionExpression=Key('id').eq(dynamoid))
        if len(resp['Items'])>0 or time.time() > timeout:
            break
        time.sleep(3)
    
    print("The query returned the following items:")
    print(resp['Items'])
    
    return format_response(resp['Items'][0]['text'], 200)
