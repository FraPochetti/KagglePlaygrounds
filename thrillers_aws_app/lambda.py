import base64, os, boto3, ast, json 

endpoint = 'is-thriller-movie-ep--2019-02-21-18-30-19'

def format_response(message, status_code):
    return {
        'statusCode': str(status_code),
        'body': json.dumps(message),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
            }
        }

def lambda_handler(event, context):

    try:
        body = json.loads(event['body'])
        image = base64.b64decode(body['data'].replace('data:image/png;base64,', '')) 
    
        runtime = boto3.Session().client(service_name='sagemaker-runtime', region_name='eu-west-1')
        response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/x-image', Body=image)
        probs = response['Body'].read().decode()

        probs = ast.literal_eval(probs)
        pred = probs.index(max(probs))
    
        if pred == 0:
            resp = 'This is not a thriller/crime movie! Not sure you will like it...'
        else:
            resp = 'This is a thriller/crime movie! Enjoy!'
    
        return format_response(resp, 200)
    except:
        return format_response('Ouch! Something went wrong on the server side! Sorry about that', 200)