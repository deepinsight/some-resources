import numpy as np
import cv2
from skimage import transform as trans

arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041] ], dtype=np.float32 )

def estimate_norm(lmk, image_size):
    assert lmk.shape==(5,2)
    tform = trans.SimilarityTransform()
    _src = float(image_size)/112 * arcface_src
    tform.estimate(lmk, _src)
    M = tform.params[0:2,:]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
    return warped, M
