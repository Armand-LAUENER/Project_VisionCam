"""
tests/test_clothing.py

Descripteur couleur haut / bas du corps (core/clothing.py) sur des images
synthétiques : même tenue, tenue inversée, autre tenue.
"""

import numpy as np

from core.clothing import bands, descriptor, similarity

BOX = (100, 50, 200, 450)


def person(top_bgr, bottom_bgr, noise=0):
    image = np.full((500, 300, 3), 128, np.uint8)
    image[50:250, 100:200] = top_bgr
    image[250:450, 100:200] = bottom_bgr
    if noise:
        rng = np.random.default_rng(noise)
        image = np.clip(image.astype(int) + rng.integers(-10, 10, image.shape), 0, 255).astype(np.uint8)
    return image


RED, BLUE, GREEN = (0, 0, 220), (220, 0, 0), (0, 200, 0)


def test_same_clothes_are_similar_despite_noise():
    a = descriptor(person(RED, BLUE, noise=1), BOX)
    b = descriptor(person(RED, BLUE, noise=2), BOX)

    assert similarity(a, b) > 0.9


def test_top_and_bottom_are_told_apart():
    a = descriptor(person(RED, BLUE), BOX)
    swapped = descriptor(person(BLUE, RED), BOX)
    other = descriptor(person(GREEN, GREEN), BOX)

    assert similarity(a, swapped) < 0.3 and similarity(a, other) < 0.3


def test_keypoints_move_the_bands():
    keypoints = np.zeros((17, 3))
    keypoints[[5, 6]] = (0, 100, 1.0)      # épaules
    keypoints[[11, 12]] = (0, 300, 1.0)    # hanches

    (upper, lower) = bands(BOX, keypoints)

    assert upper == (120, 300) and lower[0] == 315


def test_a_box_too_small_has_no_descriptor():
    assert descriptor(person(RED, BLUE), (100, 50, 104, 54)) is None
