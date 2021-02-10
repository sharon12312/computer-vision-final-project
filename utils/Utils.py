import numpy as np
import cv2

from skimage.segmentation import clear_border


def get_contours(cnts):
    """
    Get the contours object based on the OpenCV version. If the length of the contours tuple returned
    by cv2.findContours is 2, then we are using either OpenCV v2.4, v4-beta, or v4-official, and if the
    length of the contours tuple is 3, then we are using either OpenCV v3, v4-pre, or v4-alpha.
    :param cnts: list
    :return: list
    """

    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    else:
        raise Exception('Unknown error: contours tuple must have length of 2 or 3.')

    return cnts


def order_points(points):
    """
    Initialize a list of coordinates including the points in order: top-left, top-right, bottom-right, and bottom-left.
    :param points: ndarray, that contains a list of 4 points
    :return: ndarray, that contains the list of points by order: top-left, top-right, bottom-right, and bottom-left.
    """

    # initial variables
    rectangle_points = np.zeros((4, 2), dtype='float32')

    # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    pts_sum = points.sum(axis=1)
    rectangle_points[0] = points[np.argmin(pts_sum)]
    rectangle_points[2] = points[np.argmax(pts_sum)]

    # the top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
    pts_diff = np.diff(points, axis=1)
    rectangle_points[1] = points[np.argmin(pts_diff)]
    rectangle_points[3] = points[np.argmax(pts_diff)]

    return rectangle_points


def get_max_width(top_left, top_right, bottom_left, bottom_right):
    """
    Compute the width of an image using a euclidean distance, which will be the maximum distance between
    the bottom-right and bottom-left x-coordinates or the top-right and top-left x-coordinates
    :param top_left: ndarray
    :param top_right: ndarray
    :param bottom_left: ndarray
    :param bottom_right: ndarray
    :return: the maximum width
    """
    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    return max(int(width_a), int(width_b))


def get_max_height(top_left, top_right, bottom_left, bottom_right):
    """
    Compute the height of an image using a euclidean distance, which will be the maximum distance between
    the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    :param top_left: ndarray
    :param top_right: ndarray
    :param bottom_left: ndarray
    :param bottom_right: ndarray
    :return: the maximum height
    """
    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    return max(int(height_a), int(height_b))


def perspective_transform(image, points):
    """
    Generate a straight plain shape image according to the contour points that are defined in a consistent
    ordering representation.
    :param image: ndarray, contains the image values
    :param points: ndarray, contains the contour points by order
    :return: ndarray, a warped image that presents the straight plain image according to the contours points
    """

    # obtain a consistent order of the points and unpack them individually
    rectangle_points = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = rectangle_points

    # compute the max width and height of the destination image
    max_width = get_max_width(top_left, top_right, bottom_left, bottom_right)
    max_height = get_max_height(top_left, top_right, bottom_left, bottom_right)

    # construct the set of destination points to obtain a top-down view of the image
    destination_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype='float32')

    # compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rectangle_points, destination_points)

    # warp the destination image based on the perspective matrix
    warped_image = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped_image


def find_sudoku_board_contours(image, debug=False):
    """
    Detect contours in the input image, and object localization, in order to find the sudoku board puzzle
    :param image: ndarray
    :param debug: Boolean, for visualization
    :return: the warped RGB and gray transformed images, and the board contours coordinates
    """

    # convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur the image using a Gaussian kernel to remove noises
    blurred_img = cv2.GaussianBlur(gray_img, (7, 7), 3)

    # apply adaptive thresholding, that transforms a grayscale image to a binary image
    thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # invert the thresholded map to get a white sudoku board on a black background
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow('Sudoku Board Threshold', thresh)
        cv2.waitKey(0)

    # find the sudoku board's contours in the thresholded image, using the RETR_EXTERNAL method
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = get_contours(contours)

    # sort the sudoku board's contours by size in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    board_contours = None

    for c in contours:
        # determine the perimeter of the contour
        perimeter = cv2.arcLength(c, closed=True)
        # approximate the contour
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, closed=True)

        # if our approximated contour has four points, then we can assume that
        # we have found the outline of the sudoku board
        if len(approx) == 4:
            board_contours = approx
            break

    if board_contours is None:
        raise Exception('Could not found the Sudoku board puzzle in the image.')

    if debug:
        # draw the contour upon the sudoku board in the image
        img_contours = image.copy()
        cv2.drawContours(img_contours, [board_contours], -1, (0, 0, 255), thickness=2)
        cv2.imshow('Sudoku Board Contours', img_contours)
        cv2.waitKey(0)

    # apply a sudoku board extraction for a straight plain shape
    # using a perspective transform to both the original image and grayscale image
    sudoku_board_img_perspective = perspective_transform(image, board_contours.reshape(4, 2))
    sudoku_board_gray_img_perspective = perspective_transform(gray_img, board_contours.reshape(4, 2))

    if debug:
        # show the output of the warped transformed image
        cv2.imshow('Sudoku Board Perspective Transform', sudoku_board_img_perspective)
        cv2.waitKey(0)

    # compute the board coordinates by order (for applying the AR)
    board_coordinates = order_points(board_contours.reshape(4, 2))

    return sudoku_board_img_perspective, sudoku_board_gray_img_perspective, board_coordinates


def get_digit(cell, debug=False):
    """
    Generate a greyscale digit image according to the given image cell. The image cell can contain a digit,
    then the function will create a specific mask for obtaining the greyscale digit image, or the image cell
    can be empty, in this case, the function return None.
    :param cell: ndarray, presents the cell digit (if exists)
    :param debug: Boolean, for visualization
    :return: ndarray, digit image
    """

    # apply thresholding to the cell
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # clear objects connected to the label image border
    thresh = clear_border(thresh)

    if debug:
        cv2.imshow('Cell Thresholding', thresh)
        cv2.waitKey(0)

    # find contours in the thresholded cell
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = get_contours(contours)

    # if no contours were found than this is an empty cell
    if len(contours) == 0:
        return None

    # find the maximum contour in the cell
    max_contour = max(contours, key=cv2.contourArea)

    # initial a mask for the contour
    mask = np.zeros(thresh.shape, dtype='uint8')
    cv2.drawContours(mask, [max_contour], -1, 255, -1)

    # if the masked pixels (relative to the total area of the image) is filled less than 5% of the total mask,
    # then we consider it as noise
    h, w = thresh.shape
    mask_filled = cv2.countNonZero(mask) / float(w * h)
    if mask_filled < 0.05:
        return None

    # apply the mask on the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    if debug:
        cv2.imshow('Sudoku Digit', digit)
        cv2.waitKey(0)

    return digit

