import numpy as np
import cv2

def warp(img, tx, dsize=None):
    """ Warp img using tx, a matrix representing a geometric transformation.
    Pre: tx is 3x3 (or some upper-left slice of a 3x3 matrix). img is grayscale.
    Returns: an output image of shape dsize with the warped img"""
    H, W = img.shape[:2]

    # turn a 2x2 or 2x3 tx into a full 3x3 matrix
    txH, txW = tx.shape
    M = np.eye(3)
    M[:txH,:txW] = tx

    # set the output size to the input size if not specified
    if dsize is None:
        DH, DW = (H, W)
    else:
        DH, DW = dsize[::-1]
    out = np.zeros((DH, DW))

    # pseudocode written in class:
    Tinv = np.linalg.inv(T)
    for yp in range(out.shape[0]):
        for xp in range(out.shape[1]):
            p_prime = Tinv @ np.array([xp, yp, 1])
            p_prime /= p_prime[2]
            x, y = p_prime[:2]
            out[yp,xp] = img[round(y), round(x)]
    # not tested, may have bugs
    # known bug: missing bounds checks on img access
    return out

def warp_cv(img, tx, dsize=None):
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




# def warp(img, tx, dsize=None):
#     """ Warp img using tx, a matrix representing a geometric transformation.
#     Pre: tx is 3x3 (or some upper-left slice of a 3x3 matrix). img is grayscale.
#     Returns: an output image of shape dsize with the warped img"""
#     H, W = img.shape[:2]

#     # turn a 2x2 or 2x3 tx into a full 3x3 matrix
#     txH, txW = tx.shape
#     M = np.eye(3)
#     M[:txH,:txW] = tx

#     # set the output size to the input size if not specified
#     if dsize is None:
#         DH, DW = (H, W)
#     else:
#         DH, DW = dsize[::-1]
#     out = np.zeros((DH, DW))

#     Minv = np.linalg.inv(M)
#     for y in range(DH):
#         for x in range(DW):
#             xh, yh, wh = Minv @ [x, y, 1]
#             xsrc = round(xh/wh)
#             ysrc = round(yh/wh)
#             if (0 <= xsrc < W) and (0 <= ysrc < H):
#                 out[y, x] = img[ysrc, xsrc]
#     return out