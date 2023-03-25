from prototypes.styletransfer.videos import extract_frames_from_gif, make_gif, get_gif_from_s3
from prototypes.styletransfer.videos import upload_gif_to_s3, get_style_and_email, send_email_to_user, fix_img
from prototypes.styletransfer.images import load_image
from prototypes.styletransfer.model import StyleTransfer, gram_matrix, LossWeights, make_blog_style_transfer
import sys

def run_video_pipeline(object_name):
    email, style = get_style_and_email("visualneurons.com-videos", object_name)

    get_gif_from_s3("visualneurons.com-videos", object_name)

    gif = extract_frames_from_gif(object_name)
    # gif = gif[:15]
    style_img = load_image(style)
    style_img = fix_img(style_img)

    model = make_blog_style_transfer()  
    transferred = model.run_style_transfer_video(frames=gif, style_img=style_img, num_iterations=30)

    make_gif(transferred, gif_name=object_name.replace(".gif", "_styled.gif"), fps=20)
    s3_path = upload_gif_to_s3("visualneurons.com-gifs", object_name.replace(".gif", "_styled.gif"))

    send_email_to_user(email, s3_path, style)

if __name__ == "__main__":

    object_name = sys.argv[1]
    run_video_pipeline(object_name)
