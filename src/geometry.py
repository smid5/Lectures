import numpy as np
import cv2

def warp(img, tx, dsize=None):
    """ Just for now, until we write our own:
    warp img according to tx, a matrix representing a geometric transformation.
    Pre: tx is 3x3, 2x3, or 2x2"""

    H, W = img.shape[:2]
    txH, txW = tx.shape
    
    M = np.zeros((2, 3))
    M[:txH,:txW] = tx
    return cv2.warpAffine(img, M, dsize)


def estimate_translation(correspondences):
    """ Returns a translation vector (tx, ty) that is the average
    of the correspondences, given in the format as returned by
    features.get_matches """
