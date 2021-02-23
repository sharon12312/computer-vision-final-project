import numpy as np
import cv2


class AugmentedReality:
    """
    The class provides an augmented reality (AR) functionality in which can demonstrate enhanced virtual objects onto a
    real-world environment by computer-generated perceptual information.
    The construction of the virtual objects within the environment can be applied here using a combination of two
    different images and Homography computations (based on four original coordinates of the original/destination image).

    Code example:
        AugmentedReality().overlay_images_using_ar(dst_img, src_img, dst_coords)
    """
    @staticmethod
    def overlay_images_using_ar(dst_img, src_img, dst_coords, debug=False):
        """
        Performs augmented reality between a source image and a destination image using homography matrix computations
        to construct an initial perspective warp image.
        The warped image will contain the final result from both input images.
        This function requires the destination image's coordinates in which will consider the overlay process in place.

        Utils:
        * cv2.findHomography() function finds a perspective transformation between the two planes: the source's
            and the destination's coordinates.
        * cv2.warpPerspective() function transforms the source image using the specified matrix (homography
            transformation matrix H) and return the warped image according to the destination image shape.
        * cv2.fillConvexPoly() function fills a convex polygon, according to the provided vertices of the
            convex polygon. In our case it will create the mask which contains the exact polygon within the
            destination image (according to the destination's coordinates).
            - lineType: cv2.LINE_AA gives anti-aliased line which looks great for curves.
        * cv2.multiply() function computes the per-element scaled product of two arrays (images). In our case,
            when using the multiplication of the image and the mask, it will obtain control of the range of the
            output image.
        * cv2.add() function computes the per-element sum of two arrays or an array and a scalar. In our case,
            combining the two images according to the destination image.

        :param dst_img: ndarray,  the destination image in which the overlaying operation will be performed
        :param src_img: ndarray, the source image which will take place over the destination image according to
            the homography computation coordinates results
        :param dst_coords: ndarray, the destination coordinates for computing the homography matrix
        :param debug: boolean, for visualization purposes
        :return: ndarray (uint8), the destination image that contains the overlayed source image based on the
            input destination coordinates
        """

        # set the source image and the destination image shapes
        dst_h, dst_w = dst_img.shape[:2]
        src_h, src_w = src_img.shape[:2]

        # initial the matrix coordinates according to the source image shape
        src_coords = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])

        # compute the homography matrix
        (H, _) = cv2.findHomography(srcPoints=src_coords, dstPoints=dst_coords)

        # warp the source image based on the destination image
        homography_warped = cv2.warpPerspective(src=src_img, M=H, dsize=(dst_w, dst_h))

        if debug:
            # visualization for the homography result
            cv2.imshow('Homography Warped Result', homography_warped)
            cv2.waitKey(0)

        # overlay the result onto the original image
        mask = np.zeros((dst_h, dst_w), dtype='uint8')
        cv2.fillConvexPoly(img=mask, points=dst_coords.astype('int32'), color=(255, 255, 255), lineType=cv2.LINE_AA)

        if debug:
            # visualization for the mask result
            cv2.imshow('Mask Result', mask)
            cv2.waitKey(0)

        # scale the mask image
        scaled_mask = mask.copy() / 255.0

        # creates a 3D matrix of the mask in order to copy the warped source image within the colored destination image
        scaled_mask = np.dstack([scaled_mask] * 3)

        # multiply both source warped image and the mask
        src_warped_multiplied = cv2.multiply(homography_warped.astype('float'), scaled_mask)

        if debug:
            # visualization for the warped multiplied result
            cv2.imshow('Homography Warped Multiplied Result', src_warped_multiplied)
            cv2.waitKey(0)

        # multiply the destination image with the mask,
        # providing more weight to the pixels which are not from the mask image
        dst_multiplied = cv2.multiply(dst_img.astype(float), 1.0 - scaled_mask)

        if debug:
            # visualization for the warped multiplied result
            cv2.imshow('Destination Multiplied Result', dst_multiplied)
            cv2.waitKey(0)

        # combining the two images according to the destination image
        overlay_result = cv2.add(src_warped_multiplied, dst_multiplied)

        if debug:
            # visualization for the scaled overlay result
            cv2.imshow('Overlay Result', overlay_result)
            cv2.waitKey(0)

        # convert to uint8 type for constructing the image to its original color values
        overlay_result = overlay_result.astype('uint8')

        return overlay_result