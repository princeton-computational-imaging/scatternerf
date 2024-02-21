import os
import cv2
import numpy as np


def transform(path_to_left_image: str):
    """
        This function transforms and image coming from the left camera into the color scheme of the right camera,
        assuming a linear model.
        :param path_to_left_image: the path to the left image
        :param model: the model to use to perform the conversion
        :return: the converted image (in opencv format!)
        """

    left_image = cv2.imread(
        path_to_left_image
    ) / 255.

    right_image_reshaped = left_image.reshape((left_image.shape[0] * left_image.shape[1], left_image.shape[2]))

    weights = np.load('weights.npy')
    bias = np.load('bias.npy')
    right_image_corrected = (np.linalg.inv(weights) @ (right_image_reshaped - bias).T).T
    right_image_corrected = right_image_corrected.reshape(left_image.shape)

    return np.clip(right_image_corrected * 255., 0., 255.)


if __name__ == '__main__':

    path_to_left_image = os.path.join(
        '2022-06-07_13-53-45',
        'cam_stereo',
        'left',
        'image_lut',
        '00730_1654602865885976075.png'
    )

    corrected_image_l_to_r = transform(path_to_left_image)
    cv2.imwrite('test_r_to_l.png', corrected_image_l_to_r)
