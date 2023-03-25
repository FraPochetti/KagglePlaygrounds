import numpy as np
import tensorflow as tf
from prototypes.styletransfer.model import StyleTransfer, gram_matrix, LossWeights, make_blog_style_transfer
import tensorflow.contrib.eager as tfe
from prototypes.styletransfer.videos import extract_frames_from_gif, make_gif, get_gif_from_s3, upload_gif_to_s3, get_style_and_email, send_email_to_user
from prototypes.styletransfer.images import load_image
import os


rng = np.random.RandomState(42)
TEST_GIF_PATH = "prototypes/styletransfer/tests/no_god_no.gif"

def test_video_pipeline():
    email, style = get_style_and_email("visualneurons.com-videos", "frapochettigmailcom_IronManTrim.mp4")
    assert os.path.isfile("./The_Scream.jpg") == True
    assert email == "fra.pochetti@gmail.com"
    assert style == "The_Scream.jpg"

    get_gif_from_s3("style-transfer-webapptest", "no_god_no.gif")
    assert os.path.isfile("./no_god_no.gif") == True

    gif = extract_frames_from_gif("no_god_no.gif")
    gif = gif[:15]
    style_img = load_image(style)
    assert isinstance(style_img, np.ndarray)

    model = make_blog_style_transfer()  
    transferred = model.run_style_transfer_video(frames=gif, style_img=style_img)
    assert len(transferred) == len(gif)
    assert transferred[4].shape == gif[4].shape

    make_gif(transferred)
    s3_path = upload_gif_to_s3("visualneurons.com-gifs", "gif.gif")
    assert "https://s3-eu-west-1.amazonaws.com/visualneurons.com-gifs/gif.gif" == s3_path

    send_email_to_user(email, s3_path, style)

def test_get_email_and_style():
    email, style = get_style_and_email("visualneurons.com-videos", "frapochettigmailcom_IronManTrim.mp4")

    assert os.path.isfile("./The_Scream.jpg") == True
    assert email == "fra.pochetti@gmail.com"

def _sample_img(size):
    return rng.randint(0, 255, (size, size, 3)).astype("float32")


def test_model():
    content_img = _sample_img(512)
    style_img = _sample_img(512)

    model = make_blog_style_transfer()

    content_rep, style_rep = model.feature_representations(content_img, style_img)
    assert isinstance(content_rep, list)
    assert isinstance(style_rep, list)


def test_model_loss():
    # NOTE: you're not supposed to test private methods but those are
    # complicated private methods that justify their usage. We could
    # also refactor to make more things public and testable.
    content_img = _sample_img(512)
    style_img = _sample_img(512)
    init_img = _sample_img(512)

    model = make_blog_style_transfer()

    content_rep, style_rep = model.feature_representations(content_img, style_img)

    init_img = model._process_img(init_img)
    init_img = tfe.Variable(init_img, dtype=tf.float32)

    gram_style_features = [gram_matrix(style_feature) for style_feature in style_rep]
    loss_weights = LossWeights()

    losses = model._loss(loss_weights, init_img, gram_style_features, content_rep)

    assert isinstance(losses, tuple)
    assert isinstance(losses[0], tf.Tensor)

def test_run_styletransfer_video():
    gif = extract_frames_from_gif(TEST_GIF_PATH)
    gif = gif[:5]
    style_img = _sample_img(512)

    model = make_blog_style_transfer()
    transferred = model.run_style_transfer_video(frames=gif, style_img=style_img)

    assert isinstance(transferred, list)
    assert isinstance(transferred[0], np.ndarray)
    assert transferred[0].shape[2] == 3
    assert len(transferred) == len(gif)
    assert transferred[4].shape == gif[4].shape

def test_save_gif():
    gif = extract_frames_from_gif(TEST_GIF_PATH)
    gif = gif[:20]
    style_img = _sample_img(512)

    model = make_blog_style_transfer()
    transferred = model.run_style_transfer_video(frames=gif, style_img=style_img)

    make_gif(transferred)

    assert os.path.isfile("./gif.gif") == True
    assert os.stat("./gif.gif").st_size != 0

def test_from_s3_to_s3():
    get_gif_from_s3("style-transfer-webapptest", "no_god_no.gif")
    assert os.path.isfile("./no_god_no.gif") == True

    gif = extract_frames_from_gif("no_god_no.gif")
    gif = gif[:15]
    style_img = _sample_img(512)

    model = make_blog_style_transfer()
    transferred = model.run_style_transfer_video(frames=gif, style_img=style_img)

    make_gif(transferred)
    s3_path = upload_gif_to_s3("visualneurons.com-gifs", "gif.gif")

    assert "https://s3-eu-west-1.amazonaws.com/visualneurons.com-gifs/gif.gif" == s3_path

def test_run_styletransfer():
    content_img = _sample_img(512)
    style_img = _sample_img(512)

    model = make_blog_style_transfer()

    for st in model.run_style_transfer(content_img, style_img, num_iterations=10):
        assert isinstance(st.image, np.ndarray)

def test_content2weight():
    content_img = _sample_img(512)
    style_img = _sample_img(512)
    init_img = _sample_img(512)        
    model = make_blog_style_transfer()
    init_img = model._process_img(init_img)
    init_image = tfe.Variable(init_img, dtype=tf.float32)      
    loss_weights = LossWeights()       
    c2s = model._estimate_content2weight(content_img, style_img, loss_weights, init_image)     
    assert(isinstance(c2s, float))

if __name__ == "__main__":
    test_video_pipeline()