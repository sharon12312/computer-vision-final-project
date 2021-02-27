import numpy as np
import cv2

from skimage.segmentation import clear_border


class ImageProcessing:
    """
    Utils class provides helper functions for image processing including perspective transform using 4 points,
    object localization, and contours detections.

    Code example:
        image = cv2.imread('../images/sudoku1.jpg')
        img_processing = ImageProcessing()
        img_processing.find_sudoku_board_contours(image=image, debug=True)
    """
    def __init__(self):
        pass

    @staticmethod
    def adapt_points_in_img_order(points):
        """
        Initialize a list of coordinates including the points in order:
            top-left, top-right, bottom-right, and bottom-left.

        :param points: ndarray, that contains a list of 4 points
        :return: ndarray, that contains the list of points by order: top-left, top-right, bottom-right, and bottom-left.
        """

        # initial variables
        rectangle_points = np.zeros((4, 2), dtype='float32')

        # the top-left point will have the smallest sum,
        # whereas the bottom-right point will have the largest sum
        pts_sum = points.sum(axis=1)
        rectangle_points[0] = points[np.argmin(pts_sum)]
        rectangle_points[2] = points[np.argmax(pts_sum)]

        # the top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        pts_diff = np.diff(points, axis=1)
        rectangle_points[1] = points[np.argmin(pts_diff)]
        rectangle_points[3] = points[np.argmax(pts_diff)]

        return rectangle_points

    @staticmethod
    def get_max_width(top_left, top_right, bottom_left, bottom_right):
        """
        Compute the width of an image using a euclidean distance, which will be the maximum distance between
        the bottom-right and bottom-left x-coordinates or the top-right and top-left x-coordinates.

        :param top_left: ndarray
        :param top_right: ndarray
        :param bottom_left: ndarray
        :param bottom_right: ndarray
        :return: the maximum width
        """

        width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
        width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        return max(int(width_a), int(width_b))

    @staticmethod
    def get_max_height(top_left, top_right, bottom_left, bottom_right):
        """
        Compute the height of an image using a euclidean distance, which will be the maximum distance between
        the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates.

        :param top_left: ndarray
        :param top_right: ndarray
        :param bottom_left: ndarray
        :param bottom_right: ndarray
        :return: the maximum height
        """

        height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
        height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
        return max(int(height_a), int(height_b))

    @staticmethod
    def generate_fixed_image(image):
        """
        Enhance the digit image since the digit can be localized in the top/bottom or thin/thick within the image.
        This function improves the classification of the image using the CNN model.

        Utils:
        * cv2.boundingRect() function for locating the digit within the image.
        * cv2.copyMakeBorder() function for boundary handling when generate a new clearer digit image.
        * cv2.getStructuringElement() function for computing the structuring element (kernel)
            when using the erosion function.

        :param image: ndarray, contains the digit image
        :return: ndarray, the fixed digit image
        """

        # computes the up-right bounding rectangle
        x, y, w, h = cv2.boundingRect(image)

        # extract the digit (roi = region of interest)
        roi = image[y:y + h, x:x + w]

        # sets the padding of the new image border from the digit object within the image
        h, w = image.shape
        if h > w:
            top, left = round(h * 0.2), round((1.4 * h - w) / 2)
        else:
            top, left = round(w * 0.2), round((1.4 * w - h) / 2)

        # simplify image boundary handling, convert the image into the middle
        digit = cv2.resize(cv2.copyMakeBorder(roi, top, top, left, left, cv2.BORDER_CONSTANT), (500, 750))

        # thinning the digit within the image, using 5x5 kernel and erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        image = cv2.erode(digit, kernel, iterations=1)

        return image

    def perspective_transform(self, image, points):
        """
        Generate a straight plain shape image according to the contour points that are defined in a consistent
        ordering representation.

        Utils:
        * cv2.getPerspectiveTransform() function computes a perspective transform matrix from four pairs of the
            corresponding points.
        * cv2.warpPerspective() function transforms the source image using the specified matrix and return the
            warped image.

        :param image: ndarray, contains the image values
        :param points: ndarray, contains the contour points by order
        :return: ndarray, a warped image that presents the straight plain image according to the contours points
        """

        # obtain a consistent order of the points and unpack them individually
        rectangle_points = self.adapt_points_in_img_order(points)
        (top_left, top_right, bottom_right, bottom_left) = rectangle_points

        # compute the max width and height of the destination image
        max_width = self.get_max_width(top_left, top_right, bottom_left, bottom_right)
        max_height = self.get_max_height(top_left, top_right, bottom_left, bottom_right)

        # construct the set of destination points to obtain the straight plain image
        destination_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype='float32')

        # compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src=rectangle_points, dst=destination_points)

        # warp the destination image based on the perspective matrix
        warped_image = cv2.warpPerspective(src=image, M=M, dsize=(max_width, max_height))

        return warped_image

    def find_sudoku_board_contours(self, image, debug=False):
        """
        Detect contours in the input image, and object localization, in order to find the sudoku board.

        Utils:
        * cv2.GaussianBlur() function smooths the image (by the x-axis) using a Gaussian kernel for removing noises.
        * cv2.adaptiveThreshold() function applies an adaptive threshold (transforms a grayscale image to a binary image)
            - maxValue: Non-zero value assigned to the pixels for which the condition is satisfied.
            - adaptiveMethod: cv2.ADAPTIVE_THRESH_GAUSSIAN_C is a threshold value which means a Gaussian-weighted sum
                of the neighbourhood values (constant value).
            - thresholdType: cv2.THRESH_BINARY is the basic thresholding technique is Binary Thresholding. For every
                pixel, the same threshold value is applied.
            - blockSize: 11, the size of a pixel neighborhood that is used to compute a threshold value for the pixel.
            - C constant subtracted from the mean or weighted mean.
        * cv2.findContours() function for finding contours in a binary image.
            - mode: cv2.RETR_EXTERNAL retrieves only the extreme outer contours.
            - method: cv2.CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour (saving memory).
        * cv2.arcLength() function computes a curve length or a closed contour perimeter.
        * cv2.approxPolyDP() function approximates a curve or a polygon with another curve/polygon with less vertices
            so that the distance between them is less or equal to the specified precision.

        :param image: ndarray, the input image
        :param debug: boolean, for visualization purposes
        :return: tuple, the warped RGB and gray transformed images, and the board contours coordinates
        """

        # convert the image to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # blur the image using a Gaussian kernel to remove noises
        blurred_img = cv2.GaussianBlur(src=gray_img, ksize=(7, 7), sigmaX=3)

        if debug:
            cv2.imshow('Blurred Image', blurred_img)
            cv2.waitKey(0)

        # apply adaptive thresholding, that transforms a grayscale image to a binary image
        thresh = cv2.adaptiveThreshold(src=blurred_img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)

        if debug:
            cv2.imshow('Sudoku Board Thresholded (before inverse operation)', thresh)
            cv2.waitKey(0)

        # invert the thresholded map to get a white sudoku board on a black background
        thresh = cv2.bitwise_not(thresh)

        if debug:
            cv2.imshow('Sudoku Board Threshold (after inverse operation)', thresh)
            cv2.waitKey(0)

        # find the sudoku board's contours in the thresholded image, using the RETR_EXTERNAL method
        # if the length of the contours tuple is 2, then we are using either OpenCV v2.4, v4-beta, or v4-official
        # if the length of the contours tuple is 3, then we are using either OpenCV v3, v4-pre, or v4-alpha
        contours, _ = cv2.findContours(image=thresh.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        # sort the sudoku board's contours by size in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # initialize a contour that corresponds to the sudoku board outline
        board_contours = None

        for c in contours:
            # determine the perimeter of the contour
            perimeter = cv2.arcLength(curve=c, closed=True)
            # approximate the contour
            approx = cv2.approxPolyDP(curve=c, epsilon=0.02 * perimeter, closed=True)

            # if our approximated contour has four points, then we can assume that
            # we have found the outline of the sudoku board
            if len(approx) == 4:
                board_contours = approx
                break

        if board_contours is None:
            raise Exception('Could not found the Sudoku board in the image.')

        if debug:
            # draw the contour upon the sudoku board in the image
            img_contours = image.copy()
            cv2.drawContours(img_contours, [board_contours], -1, (0, 0, 255), thickness=2)
            cv2.imshow('Sudoku Board Contours', img_contours)
            cv2.waitKey(0)

        # apply a sudoku board extraction for a straight plain shape
        # using a perspective transform to both the original image and grayscale image
        sudoku_board_img_perspective = self.perspective_transform(image, board_contours.reshape(4, 2))
        sudoku_board_gray_img_perspective = self.perspective_transform(gray_img, board_contours.reshape(4, 2))

        if debug:
            # show the output of the warped transformed image
            cv2.imshow('Sudoku Board Perspective Transform', sudoku_board_img_perspective)
            cv2.waitKey(0)

        # compute the board coordinates by order (for applying the AR)
        board_coordinates = self.adapt_points_in_img_order(board_contours.reshape(4, 2))

        return sudoku_board_img_perspective, sudoku_board_gray_img_perspective, board_coordinates

    def get_digit(self, cell, debug=False):
        """
        Generate a greyscale digit image according to the given image cell. The image cell can contain a digit,
        then the function will create a specific mask for obtaining the greyscale digit image, or the image cell
        can be empty, in this case, the function return None.

        Utils:
        * cv2.threshold() function for applying a fixed-level threshold (get a binary image out of a grayscale image).
            - type: cv2.THRESH_BINARY_INV inverses the binary image (from black to white and vice versa).
                    cv2.THRESH_OTSU finds the optimal threshold value.
        * skimage.segmentation.clear_border() function for clearing objects connected to the label image border.
        * cv2.findContours() function for finding contours in a binary image.
            - mode: cv2.RETR_EXTERNAL retrieves only the extreme outer contours.
            - method: cv2.CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour (saving memory).
        * cv2.drawContours() function draws contour outlines within the image.

        :param cell: ndarray, presents the cell digit (if exists)
        :param debug: boolean, for visualization purposes
        :return: ndarray, digit image
        """

        # apply thresholding to the cell
        thresh = cv2.threshold(src=cell, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # clear objects connected to the label image border
        thresh = clear_border(thresh)

        if debug:
            cv2.imshow('Cell Thresholding', thresh)
            cv2.waitKey(0)

        # find contours in the thresholded cell
        # if the length of the contours tuple is 2, then we are using either OpenCV v2.4, v4-beta, or v4-official
        # if the length of the contours tuple is 3, then we are using either OpenCV v3, v4-pre, or v4-alpha
        contours, _ = cv2.findContours(image=thresh.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        # if no contours were found than this is an empty cell
        if len(contours) == 0:
            return None

        # find the maximum contour in the cell
        max_contour = max(contours, key=cv2.contourArea)

        # initial a mask for the contour
        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(image=mask, contours=[max_contour], contourIdx=-1, color=255, thickness=-1)

        # if the masked pixels (relative to the total area of the image) is filled less than 2.6% of the total mask,
        # then we consider it as noise
        h, w = thresh.shape
        mask_filled = cv2.countNonZero(mask) / float(w * h)
        if mask_filled < 0.026:
            return None

        # apply the mask on the thresholded cell, this way it makes certain pixels in the image to be black
        # according to our mask (when merging them together)
        digit = cv2.bitwise_and(thresh, thresh, mask=mask)

        if debug:
            cv2.imshow('Sudoku Digit (Before Fixing)', digit)
            cv2.waitKey(0)

        # the final step is to generate a new fixed digit image from the basic mask, for the classification improvement
        digit = self.generate_fixed_image(digit)

        if debug:
            cv2.imshow('Sudoku Digit (After Fixing)', digit)
            cv2.waitKey(0)

        return digit