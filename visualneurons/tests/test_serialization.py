import numpy as np
from prototypes.styletransfer.serialization import base64_to_image, image_to_base64


rng = np.random.RandomState(42)


def _sample_img(size):
    return rng.randint(0, 255, (size, size, 3)).astype("float32")


def test_round_trip():
    img = _sample_img(512)

    serialized = image_to_base64(img)
    deserialized = base64_to_image(serialized)

    np.testing.assert_almost_equal(img, deserialized)
