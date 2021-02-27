import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity
import sudoku_solver_cv


class EvaluateDigits:
    """
    The class provides an evaluation between two arrays with the same dimension shape for sudoku digits
    comparison in both initial and solved sudoku digits.

    Code example:
        EvaluateDigits().evaluate(eval_sudoku_digits, eval_solved_sudoku_digits,
                                  origin_sudoku_board_path, origin_solved_sudoku_board_path)
    """

    @staticmethod
    def evaluate(eval_digits, eval_solved_digits, origin_digits_path, origin_solved_digits_path):
        """
        Evaluates between two matrices that contain the sudoku board digits for both evaluated and original figures.
        Perform it for the initial sudoku board digits and the solved sudoku board digits.
        Prints the results to the terminal.

        :param eval_digits: ndarray, the evaluated digits matrix
        :param eval_solved_digits: list, the evaluated solution list
        :param origin_digits_path: ndarray, the original digits matrix
        :param origin_solved_digits_path: list, the original solution list
        :return: None
        """

        # read sudoku matrix
        origin_digits = np.load(origin_digits_path).tolist()
        origin_solved_digits = np.load(origin_solved_digits_path).tolist()

        # perform evaluation on the sudoku board digits and the sudoku board solution
        if (eval_digits == origin_digits).all():
            print(f'{__class__.__name__} INFO: Evaluating sudoku board digits: '
                  f'[SUCCESS] the digits were successfully stored and compared.')
        else:
            print(f'{__class__.__name__} INFO: Evaluating sudoku board digits: '
                  f'[FAILED] the evaluated digits were not equal to the original digits.')

        if eval_solved_digits == origin_solved_digits:
            print(f'{__class__.__name__} INFO: Evaluating solved sudoku board digits: '
                  f'[SUCCESS] the solution was successfully found and matched.')
        else:
            print(f'{__class__.__name__} INFO: Evaluating solved sudoku board digits: '
                  f'[FAILED] the solution was not equal to the original solution.')


class EvaluateImages:
    """
    The class provides a difference computation between two images with the same dimension shape
    for evaluation operation.

    Code example:
        original_image = cv2.imread('path-to-the-original-image')
        evaluated_image = sudoku_solver_cv.run(image_path='path-to-the-evaluated-image')
        eval_imgs = EvaluateImages(original_image=original_image, evaluated_image=evaluated_image)
        eval_imgs.evaluate(method='mse', debug=False)
    """

    def __init__(self, original_image, evaluated_image):
        # initial variables
        self._original_image = original_image
        self._evaluated_image = evaluated_image

    @staticmethod
    def _mse(origin_img, eval_img):
        """
        Perform a "Mean Squared Error" computation between two images which is the sum of the squared difference.
        If the error result is lower, it means that the two images are more similar to each other.

        NOTE: the two images must have the same dimension.

        :param origin_img: ndarray, the original image
        :param eval_img: ndarray, the evaluated image
        :return: tuple (float, ndarray),
            - The mean squared error result
            - The full MSE image
        """

        # computes the mse
        err = np.sum((origin_img.astype('float') - eval_img.astype('float')) ** 2)
        err /= float(origin_img.shape[0] * origin_img.shape[1])

        # computes a full MSE image
        diff = (origin_img.astype('float') - eval_img.astype('float')) ** 2

        return err, diff

    @staticmethod
    def _ssim(origin_img, eval_img):
        """
        Perform a "Structural Similarity Index" computation between two images.
        If the score result is 1, it means that the two images are more similar to each other.

        NOTE: the two images must have the same dimension.

        :param origin_img: ndarray, the original image
        :param eval_img: ndarray, the evaluated image
        :return: tuple (float, ndarray),
            - The mean structural similarity index result
            - The full SSIM image
        """

        return structural_similarity(origin_img, eval_img, full=True)

    @staticmethod
    def _display_diff(diff, origin_img, eval_img):
        """
        Displays the difference regions within the two images. Draw in each image the rectangles areas in which
        their is a difference between them.

        Utils:
        * cv2.threshold() function for applying a fixed-level threshold (get a binary image out of a grayscale image).
            - type: cv2.THRESH_BINARY_INV inverses the binary image (from black to white and vice versa).
                    cv2.THRESH_OTSU finds the optimal threshold value.
        * cv2.findContours() function for finding contours in a binary image.
            - mode: cv2.RETR_EXTERNAL retrieves only the extreme outer contours.
            - method: cv2.CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour (saving memory).
        * cv2.boundingRect() function for locating the digit within the image.

        :param diff: ndarray, The full diff image result based on the evaluation computation
        :param origin_img: ndarray, the original image
        :param eval_img: ndarray, the evaluated image
        :return: None
        """

        # convert to unsigned 8-bit integers image
        diff = (diff * 255).astype('uint8')

        # display the diff image
        # cv2.imshow('Diff Image', diff)
        # cv2.waitKey()

        # apply thresholding to the difference image result
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # display the thresholded image
        # cv2.imshow('Thresholded Image', thresh)
        # cv2.waitKey()

        # find the contours for obtaining the differentiated regions of the two images
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)

            # draw the bounding box on both images to represent the differences between the two images
            cv2.rectangle(origin_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(eval_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # display the diff images results
        cv2.imshow('Original Image Differences', origin_img)
        cv2.imshow('Evaluated Image Differences', eval_img)
        cv2.waitKey(0)

    def evaluate(self, method='mse', debug=False, save_path='./evaluation/results/evaluation-results.jpg'):
        """
        Computes the difference between two images for evaluation and print the result to the terminal.
        Saves the results to a local path.

        [OPTIONAL] Relevant only to the 'ssim' method:
            visualization difference between the two images according to their contours values.

        :param save_path: string, save the evaluation results in the given path
        :param method: string, includes 2 option types:
            - 'mse': Mean Squared Error (MSE) [default]
            - 'ssim': Mean Structural Similarity Index (SSIM)
        :param debug: boolean, for visualization purposes
        :return: None
        """

        # convert the input images to grayscale
        origin_img = cv2.cvtColor(self._original_image, cv2.COLOR_BGR2GRAY)
        eval_img = cv2.cvtColor(self._evaluated_image, cv2.COLOR_BGR2GRAY)

        err, diff = None, None
        if method == 'ssim':
            # compute the mean structural similarity index (ssim) between the two images
            err, diff = self._ssim(origin_img, eval_img)
        elif method == 'mse':
            # compute the mean squared error (mse) between the two images
            err, diff = self._mse(origin_img, eval_img)

        # display results
        print(f'{__class__.__name__} INFO: {method.upper()} result: {err:.2f}')

        # relevant only for ssim method
        if debug:
            # visualization of the diff result
            self._display_diff(diff, origin_img.copy(), eval_img.copy())

        # save results
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        ax1.imshow(cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB), cmap='gray', interpolation='nearest'), \
            ax1.set_title('Origin Image'), ax1.axis('off')
        ax2.imshow(cv2.cvtColor(eval_img, cv2.COLOR_BGR2RGB), cmap='gray', interpolation='nearest'), \
            ax2.set_title('Evaluated Image'), ax2.axis('off')
        f.suptitle(f'Evaluation Results\n{method.upper()}: {err:.2f}')
        f.savefig(save_path)

        # display to the terminal
        print(f'{__class__.__name__} INFO: evaluation images were saved in path: {save_path}.')


if __name__ == '__main__':
    # initial variables
    sudoku_solver_cv = sudoku_solver_cv.SudokuSolverCV()
    count = 1

    # initial images path
    images = [('./evaluation/images/sudoku5.jpg', './evaluation/images/sudoku5origin.jpg'),
              ('./evaluation/images/sudoku6.jpg', './evaluation/images/sudoku6origin.jpg'),
              ('./evaluation/images/sudoku7.jpg', './evaluation/images/sudoku7origin.jpg')]

    # initial digits path
    np_files = ('./evaluation/numpy_files/sudoku_digits.npy', './evaluation/numpy_files/sudoku_digits_result.npy')

    # perform evaluation for each image
    for eval_image_path, origin_image_path in images:

        # display image properties
        print(f'Main INFO: Evaluating images: {eval_image_path}-{origin_image_path}...')

        # read images
        original_image = cv2.imread(origin_image_path)
        evaluated_image = sudoku_solver_cv.run(image_path=eval_image_path, debug=False)

        # read sudoku digits matrices
        eval_sudoku_digits = sudoku_solver_cv.sudoku_board
        eval_solved_sudoku_digits = sudoku_solver_cv.solved_sudoku_board
        (origin_sudoku_board_path, origin_solved_sudoku_board_path) = np_files

        # perform evaluation on the sudoku board digits and the sudoku board solution
        EvaluateDigits().evaluate(eval_sudoku_digits, eval_solved_sudoku_digits,
                                  origin_sudoku_board_path, origin_solved_sudoku_board_path)

        # perform evaluation between the two images
        eval_imgs = EvaluateImages(original_image=original_image, evaluated_image=evaluated_image)
        eval_imgs.evaluate(method='mse', save_path=f'./evaluation/results/evaluation-results{count}.jpg')
        count += 1

        print(f'Main INFO: Evaluating images [DONE]\n')
