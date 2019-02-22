from chalice import Chalice
from chalice import BadRequestError
import base64, os, boto3, ast
#import numpy as np

app = Chalice(app_name='movies_posters')
app.debug=True
endpoint = 'is-thriller-movie-ep--2019-02-21-18-30-19'

@app.route('/', methods=['POST'])
def index():
    body = app.current_request.json_body

    if 'data' not in body:
        raise BadRequestError('Missing image data')
    #if 'ENDPOINT_NAME' not in os.environ:
    #    raise BadRequestError('Missing endpoint')

    image = base64.b64decode(body['data']) # byte array
    #endpoint = os.environ['ENDPOINT_NAME'] 

    print(endpoint) 

    runtime = boto3.Session().client(service_name='sagemaker-runtime', region_name='eu-west-1')
    response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/x-image', Body=image)
    probs = response['Body'].read().decode() # byte array

    probs = ast.literal_eval(probs) # array of floats
    pred = probs.index(max(probs))
    #probs = np.array(probs) # numpy array of floats

    #pred = probs.argmax() 
    #label = ['Other', 'Thriller/Crime'][pred]

    return {'response': str(probs)}