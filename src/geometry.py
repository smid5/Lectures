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

    # your code here
    Tinv = np.linalg.inv(M)
    for xd in range(DW):
        for yd in range(DH):
            dest = np.array([xd, yd, 1])
            xs, ys, w = (Tinv @ dest)
            xs, ys = round(xs / w), round(ys / w) # nearest neighbor
            if xs not in range(W) or ys not in range(H):
                out[yd, xd] = 0
            else:
                out[yd, xd] = img[ys, xs]
            # xs, ys = xs / w, ys / w
            # x1, y1 = math.floor(xs), math.floor(ys)
            # x2, y2 = x1 + 1, y1 + 1
            # if x1 not in range(W) or x2 not in range(W) or y1 not in range(H) or y2 not in range(H):
            #     out[yd, xd] = 0
            # else:
            #     dx1 = xs - x1
            #     dy1 = ys - y1
            #     dx2 = x2 - xs
            #     dy2 = y2 - ys
            #     out[yd, xd] = img[y1, x1] * (dx2 * dy2) + img[y2, x1] * dy1 * dx2 + img[y1, x2] * dy2 * dx1 + img[y2, x2] * dy1 * dx1

    # # numpy speed-up
    # Tinv = np.linalg.inv(M)
    # dest_matrix = np.zeros((3, (DH*DW)), dtype=int)
    # for i in range(DW):
    #     for j in range(DH):
    #         dest_matrix[:,i*DH+j] = np.array([i, j, 1])
    # source_matrix = Tinv @ dest_matrix
    # source_matrix /= source_matrix[2]
    # xys = source_matrix[:2]
    # xy1 = np.floor(xys).astype(int)
    # xy2 = xy1 + 1
    # out[::-1] 
    
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