import numpy as np
from prototypes.styletransfer.images import load_image
from prototypes.styletransfer.videos import extract_frames_from_gif

TEST_IMG_PATH = "prototypes/styletransfer/tests/img_lights.jpg"
TEST_GIF_PATH = "prototypes/styletransfer/tests/no_god_no.gif"


def test_load_image():

    data = load_image(TEST_IMG_PATH)

    assert isinstance(data, np.ndarray)
    assert data.shape == (341, 512, 3)

def test_load_gif(): 
    gif = extract_frames_from_gif(TEST_GIF_PATH)
    assert isinstance(gif, list)
    assert isinstance(gif[0], np.ndarray)
    assert gif[302].shape[2] == 3