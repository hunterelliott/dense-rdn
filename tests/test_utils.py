"""
Tests of ops module functions
"""

import numpy as np
import utils


def test_load_norm_image():
    x_shape = (64, 64, 3)

    im_path = '../assets/Ilya_Prigogine_1977c_compressed.jpg'

    # First test the normalization
    im = utils.load_and_normalize_image(x_shape, im_path, clip_below=None)

    assert im.shape[0:2] == x_shape[0:2]

    assert np.allclose(np.mean(im, axis=(0, 1)), 0, atol=1e-6)
    assert np.allclose(np.std(im, axis=(0, 1)), 1, atol=1e-6)

    # ..then the clipping
    clip_val = -.234
    im = utils.load_and_normalize_image(x_shape, im_path, clip_below=clip_val)

    assert np.all(im >= clip_val)

