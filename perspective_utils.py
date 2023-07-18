import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from binarization_utils import binarize


def birdeye(img, verbose=False):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame
    :param verbose: if True, show the transformation result
    :return: warped image, and both forward and backward transformation matrices
    """
    h, w = img.shape[:2]

    src = np.float32([[600,300],    # br
                      [150,720],    # bl
                      [1300,720],   # tl
                      [750,300]])  # tr
    dst = np.float32([[100,0],       # br
                      [100,720],       # bl
                      [1100,720],       # tl
                      [1100,0]])      # tr

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if verbose:
        f, axarray = plt.subplots(1, 2)
        f.set_facecolor('white')
        axarray[0].set_title('Before perspective transform')
        axarray[0].imshow(img, cmap='gray')
        for point in src:
            axarray[0].plot(*point, '.')
        axarray[1].set_title('After perspective transform')
        axarray[1].imshow(warped, cmap='gray')
        for point in dst:
            axarray[1].plot(*point, '.')
        for axis in axarray:
            axis.set_axis_off()
        plt.show()

    return warped, M, Minv


if __name__ == '__main__':


    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)

        

        img_binary = binarize(img, verbose=False)

        img_birdeye, M, Minv = birdeye(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), verbose=True)


