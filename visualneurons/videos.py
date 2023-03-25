import cv2
import os
import numpy as np
import subprocess
from typing import List
from PIL import Image
import shutil
from botocore.exceptions import ClientError

# AWS credentials are secured by an appropriate IAM Role we attach to the EC2 instance at spin-up
import boto3 

def send_email_to_user(email: str,
                       s3_path: str,
                       style: str):

    style2name = {
        "The_Scream.jpg": "Select your style",
        "The_Scream.jpg": "Munch, The Scream",
        "Kand1.jpeg": "Kandinsky, Composition VII",
        "Kand2.png": "Kandinsky, Composition VIII",
        "Monet.jpg": "Monet, Wheatstacks (End of Summer)",
        "Picasso.png": "Picasso, Guernica",
        "VanGogh.png": "Van Gogh, Starry Night"
    }            

    SENDER = "fra.pochetti@gmail.com"
    RECIPIENT = email
    AWS_REGION = "eu-west-1"
    SUBJECT = "VisualNeurons.com - your Style-Transferred GIF is ready!"
    
    BODY_TEXT = ("VisualNeurons.com - your Style-Transferred GIF is ready! \r\n"
                 "Our AWS artists finished crunching your GIF, which is now available for download."
                )

    # The HTML body of the email.
    BODY_HTML = f"""
    <html>

    <head></head>

    <body>
        <h4>Our AWS artists finished crunching your GIF!</h4>
        <p>
            We asked them to pick a brush and render the file as <b>{style2name[style]}</b>.
            <br>
            The result is really cool!
            <br>
            You can download the GIF <a href="{s3_path}">here</a>.
            <br>
        </p>
    </body>

    </html>
                """            

    # The character encoding for the email.
    CHARSET = "UTF-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses', region_name=AWS_REGION)

    # Try to send the email.
    try:
        #Provide the contents of the email.
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

def get_style_and_email(bucket: str,
                        object_name: str):
    s3 = boto3.client('s3')
    response = s3.head_object(Bucket=bucket, Key=object_name)
    email = response['Metadata']['email']
    style = response['Metadata']['style']

    s3.download_file("visualneurons.com", style, style)

    return email, style

def get_gif_from_s3(bucket: str,
                    object_name: str):
    s3 = boto3.client('s3')
    s3.download_file(bucket, object_name, object_name)

def upload_gif_to_s3(bucket: str,
                     object_name: str):
    s3 = boto3.client('s3')
    s3.upload_file(object_name, bucket, object_name, ExtraArgs={'ACL': 'public-read'})
    s3_path = f"https://s3-eu-west-1.amazonaws.com/{bucket}/{object_name}" 
    return s3_path

def fix_img(img: np.array) -> np.array:
    if len(img.shape) > 2 and img.shape[2] == 4:
      img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def extract_frames_from_gif(gif_path: str) -> List[np.array]:
    # this assumes the gif has been already downloaded from S3 to EC2

    directory = "./frames"
    if os.path.exists(directory): shutil.rmtree(directory)
    os.makedirs(directory)

    subprocess.run(["ffmpeg", "-i",  f"{gif_path}", f"{directory}/frame%05d.png"])

    frames = []
    for img_name in sorted(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        img = Image.open(img_path)
        img = np.array(img)
        img = fix_img(img)
        frames.append(img)

    return frames

def save_frames_to_disk(images: List[np.array],
                        directory: str):
    if os.path.exists(directory): shutil.rmtree(directory)
    os.makedirs(directory)

    for i, image in enumerate(images):
        Image.fromarray(image).save(os.path.join(directory, f"frame{i:05}.jpeg"))

def make_gif(images: List[np.array], 
             directory: str = "./transferred", 
             gif_name: str = "gif.gif",
             fps: int = 24):
    save_frames_to_disk(images, directory)
    subprocess.run(["ffmpeg", "-y", "-f", "image2", "-framerate", f"{fps}", "-i", f"{directory}/frame%05d.jpeg", f"{gif_name}"])

def make_video(images: List[np.array], 
               save_to: str, 
               video_name: str = "video.avi"):
  
  height, width, layers = images[0].shape

  video = cv2.VideoWriter(os.path.join(save_to, video_name), 0, 30, (width,height))

  for image in images:
      video.write(image)

  cv2.destroyAllWindows()
  video.release()
  
  convert_avi2mp4(os.path.join(save_to, video_name))
  
  return

def autocrop(image, threshold=10):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]
    
    return image
  
def add_topbottom_stripes(img):
  return cv2.copyMakeBorder(img, 40, 40, 0, 0, cv2.BORDER_CONSTANT, value=10)

def convert_avi2mp4(video_path):
  subprocess.run(["ffmpeg", "-y", "-i", f"{video_path}", f"{video_path.replace('avi', 'mp4')}"])
