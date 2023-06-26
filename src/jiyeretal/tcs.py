import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments_hu
from skimage.color import rgb2gray


class TCS:
    def __init__(self, img: Image.Image) -> None:
        grayimg = rgb2gray(img) * 255
        grayimg = grayimg.astype(np.uint8)
        glcm = graycomatrix(grayimg, [1], [0])
        self.contrast = graycoprops(glcm, "contrast")[0, 0]
        self.homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
        self.correlation = graycoprops(glcm, "correlation")[0, 0]
        self.energy = graycoprops(glcm, "energy")[0, 0]
        self.hu = moments_hu(grayimg)
        self.hsv_histogram = np.asarray(img.convert("HSV").histogram()).reshape(-1)

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [
                self.contrast,
                self.homogeneity,
                self.correlation,
                self.energy,
                *self.hu,
                *self.hsv_histogram,
            ]
        )
