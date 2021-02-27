import numpy as np
import cv2

import os
import argparse

from utils.image_processing import ImageProcessing
from neural_networks.cnn import DigitsNet
from sudoku.backtracking import Sudoku
from utils.augmented_reality import AugmentedReality


class SudokuSolverCV:
    """
    A sudoku solver computer-vision (CV) class for performing an augmented solution of the sudoku
    board within the input image.

    Code example:
        image_path = './images/sudoku1.jpg'
        input_image = cv2.imread(image_path)
        cv2.imshow('Input Sudoku', input_image)
        cv2.waitKey(0)

        sudoku_solver_cv = SudokuSolverCV()
        final_result_img = sudoku_solver_cv.run(image_path=image_path, debug=False)
        cv2.imshow('Final Sudoku Result', final_result_img)
        cv2.waitKey(0)
    """
    def __init__(self):
        # initial variables
        self._img_processing = ImageProcessing()
        self._digits_net = DigitsNet(model_path='models/digits_classifier.h5')
        self.sudoku_board = None
        self.solved_sudoku_board = None

        # initial settings
        self._MAX_ROW = 9
        self._MAX_COL = 9

    @staticmethod
    def _solve_sudoku_matrix(board, debug=False):
        """
        Solve the sudoku board matrix using the Sudoku class. The class performs a sudoku solver algorithm and
        returns the solution matrix if exists.

        :param board: ndarray, the sudoku matrix 2d array
        :param debug: boolean, for visualization purposes
        :return: tuple (ndarray, boolean),
            - the solved sudoku matrix if exists, otherwise return None
            - if True, the sudoku was solved successfully, otherwise False
        """

        # initial the sudoku class based on the OCR operation
        sudoku = Sudoku(board.tolist())

        if debug:
            # displays the sudoku board
            print('# Sudoku Board #')
            sudoku.print()

        # solve the sudoku
        sudoku_solution = sudoku.solve_sudoku()
        solved = sudoku_solution is not None

        if debug:
            if solved:
                # displays the solved sudoku board
                print('# Sudoku Board Solution #')
                sudoku.print()

        return sudoku_solution, solved

    @staticmethod
    def _drawn_image_grid_solution(board, cell_coordinates, sudoku_solution, warped_sudoku_board):
        """
        Draws the solution matrix within the warped perspective sudoku image.
        The function follows each cell and generates a digit text (where the cell is empty) within the cell image.

        Utils:
        * cv2.putText() function renders the specified text string (digit) in the cell image.
            - img: the cell image.
            - text: the digit string.
            - org: the bottom-left position (x,y) of the text string in the cell image.
            - fontFace: using cv2.FONT_HERSHEY_SIMPLEX as the font type, which is the normal size sans-serif font.
            - fontScale: the font size in which can be computed generically for each cell shape.
            - color: the text's color.
            - thickness: the text's thickness.

        :param board: ndarray, the sudoku board 2d matrix
        :param cell_coordinates: list, the coordinates' list
        :param sudoku_solution: list, the solved sudoku board matrix array
        :param warped_sudoku_board: ndarray, the perspective warped sudoku board image
        :return: ndarray, the warped sudoku board solution image
        """

        # initial the boolean sudoku board existence
        board_digit_exists = np.array(board, dtype=bool).tolist()

        # display the solution on the sudoku warped transformed image
        for (cell_row, board_row, digit_exists_row) in zip(cell_coordinates, sudoku_solution, board_digit_exists):
            for (box, digit, digit_exists) in zip(cell_row, board_row, digit_exists_row):
                # put text only within the empty cell
                if not digit_exists:
                    # extract the cell's coordinates
                    start_j, start_i, end_j, end_i = box

                    # compute the coordinates for locating the drawn digit within the image
                    text_j = int((end_j - start_j) * 0.32)
                    text_i = int((end_i - start_i) * -0.18)

                    # set the bottom-left position of the text string in the cell
                    text_j += start_j  # add ~30% to the start width position
                    text_i += end_i  # add ~20% to the start height position

                    # compute the font scale for the image
                    cell_height, cell_width, scale = end_i - start_i, end_j - start_j, 1
                    font_scale = min(cell_width, cell_height) / (50 / scale)

                    # draw the result digit on the sudoku warped transformed image
                    cv2.putText(img=warped_sudoku_board, text=str(digit), org=(text_j, text_i),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(205, 0, 0),
                                thickness=int(round(font_scale*2)))

        return warped_sudoku_board

    @staticmethod
    def _overlay_solution_image_on_the_original_image(original_image, warped_sudoku_board, sudoku_board_coords):
        """
        Merge both original image (destination image) and warped image (source image) using an augmented
        reality (AR) functionality.

        :param original_image: ndarray, the destination image
        :param warped_sudoku_board: ndarray, the source image
        :param sudoku_board_coords: ndarray, the destination coordinates for homography matrix computations
        :return:
        """

        # overlay the solution image within the original image using augmented reality functionality
        return AugmentedReality().overlay_images_using_ar(original_image, warped_sudoku_board, sudoku_board_coords)

    def _scan_sudoku(self, image, debug=False):
        """
        Detects the sudoku board's contours in the input image, and object localization, in order to find the
        sudoku board using the ImageProcessing class.

        :param image: ndarray, the input image
        :param debug: boolean, for visualization purposes
        :return: tuple (warped_color_img, warped_gray_img, board_coords), return the warped RGB and gray transformed
            images, and the board contours coordinates
        """

        # find the sudoku contours and the warped images (in color and grayscale)
        return self._img_processing.find_sudoku_board_contours(image, debug)

    def _character_recognition_and_cell_coordinates(self, warped_sudoku_board, debug=False):
        """
        Performs an Optical Character Recognition (OCR) on the warped sudoku board.
        The function requires to generate a 2D sudoku board matrix (9x9) following the steps:
            - Separating each cell in the board's grid by computing both height and width steps with a simple board
                deviation by 9 (since the sudoku board contains 9x9 cells).
            - For each cell, a shape calculation is been performed using the height's and width's steps.
            - Extracting the cell image from the original sudoku board image using the cell's positions.
            - Warping the sudoku digit image according to the extracted cell image using the ImageProcessing
                class (get_digit() function).
            - Performing a prediction classification using the DigitsNet model (CNN) for the identified digits.

        :param warped_sudoku_board: ndarray, the perspective warped sudoku board image
        :param debug: boolean, for visualization purposes
        :return: tuple (ndarray, list),
            - the extracted generated sudoku board from the sudoku image
            - the cells' coordinates
        """

        # initial the sudoku board matrix
        board = np.zeros((self._MAX_ROW, self._MAX_COL), dtype=np.int)

        # computes the steps pixels according to the grid image board
        height_step = warped_sudoku_board.shape[0] // self._MAX_ROW
        width_step = warped_sudoku_board.shape[1] // self._MAX_COL
        cell_coordinates = []

        for i in range(self._MAX_ROW):
            row = []
            for j in range(self._MAX_COL):
                # compute the starting and ending coordinates of the current cell
                start_i = i * height_step
                end_i = (i + 1) * height_step
                start_j = j * width_step
                end_j = (j + 1) * width_step

                # add the coordinates to the cell coordinates list
                row.append((start_j, start_i, end_j, end_i))

                # get the cell from the warped perspective image
                cell = warped_sudoku_board[start_i:end_i, start_j:end_j]

                # extract the digit from the cell
                digit = self._img_processing.get_digit(cell, debug)

                # verify that their is a digit withing the cell
                if digit is not None:
                    digit_pred = self._digits_net.predict(digit)
                    board[i, j] = digit_pred

            # store the coordinate's cell
            cell_coordinates.append(row)

        return board, cell_coordinates

    def run(self, image_path, debug=False):
        """
        Perform an augmented solution of the sudoku board within the input image.
        The function serves perceptual computer vision operations for image processing and augmented
        reality (AR) functionalities.

        The main flow following the steps:
            1. Image Processing:
                a. Detect contours and object localization, in which can find the sudoku board contours.
                b. Sudoku board extraction for a straight plain shape.
                c. Optical Character Recognition (OCR) and digit classification using a Convolutional Neural
                    Network (CNN).
            2. Sudoku Solver:
                a. Sudoku matrix creation according to the OCR.
                b. Sudoku solver algorithm that performs the sudoku matrix solution.
            3. A Straight Plain Image Result Creation (Image Processing):
                a. Localize the solution’s digits onto the straight plain image.
            4. Augmented Reality:
                a. Merge/overlay the solution result upon the original image using both destination and source images.

        :param image_path: string, the path of the input image
        :param debug: boolean, for visualization purposes
        :return: ndarray, the final image result that contains the original image and the augmented reality solution
            within the original image
        """

        # load the input image
        original_image = cv2.imread(image_path)

        # scan the image and locate the sudoku board contours
        warped_sudoku_board, warped_sudoku_board_gray, sudoku_board_coords = self._scan_sudoku(original_image, debug)

        # performs an Optical Character Recognition (OCR) based on the warped sudoku board
        board, cell_coordinates = self._character_recognition_and_cell_coordinates(warped_sudoku_board_gray, debug)

        # solve the sudoku and display the solution if exists
        sudoku_solution, solved = self._solve_sudoku_matrix(board, debug=debug)

        # store sudoku board figures for evaluation
        self.sudoku_board = board
        self.solved_sudoku_board = sudoku_solution

        # if the sudoku was not solved print an info message and return
        if not solved:
            print(f'{__class__.__name__} INFO: a sudoku solution has not been found.')
            return None
        else:
            print(f'{__class__.__name__} INFO: a sudoku solution has been found.')

        # return the drawn warped sudoku board solution
        warped_sudoku_board = \
            self._drawn_image_grid_solution(board, cell_coordinates, sudoku_solution, warped_sudoku_board)

        if debug:
            # show the output image
            cv2.imshow('Sudoku Result', warped_sudoku_board)
            cv2.waitKey(0)

        # merge both original image and solved image using AR
        final_result_warped_image = \
            self._overlay_solution_image_on_the_original_image(original_image, warped_sudoku_board, sudoku_board_coords)

        if debug:
            # show the output image
            cv2.imshow('Final Sudoku Result', final_result_warped_image)
            cv2.waitKey(0)

        return final_result_warped_image


def main():
    def str2bool(v):
        """
        Parse a string type input to a boolean type.
        :param v: the input value from the terminal arguments
        :return: boolean
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # define the given arguments variables: image-path, save-mode, visualize-mode and debug-mode
    args_parser = argparse.ArgumentParser(description='Performs an augmented solution within a given '
                                                      'sudoku board image.')
    args_parser.add_argument('-i', '--image', required=True,
                             help='the path to the input sudoku image')
    args_parser.add_argument('-s', '--save', type=str2bool, default=True,
                             help='save the augmented solved sudoku output image')
    args_parser.add_argument('-v', '--visualize', type=str2bool, default=True,
                             help='display the input image and the augmented solved sudoku output image')
    args_parser.add_argument('-d', '--debug', type=str2bool, default=False,
                             help='visualization of the flow steps images')
    args = vars(args_parser.parse_args())

    # initial args
    image_path = args['image']
    display = args['visualize']
    save = args['save']
    debug = args['debug']

    # validate the image files extensions, if it is not an image then exit the program
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        print(f'Main ERROR: the input image does not contain the suffix images: [.png, .jpg, .jpeg, .tiff, .bmp, .gif]')
        return

    if display:
        # display the input image
        input_image = cv2.imread(image_path)
        cv2.imshow('Input Sudoku', input_image)
        cv2.waitKey(0)

    # run the sudoku-solver-cv algorithm
    sudoku_solver_cv = SudokuSolverCV()
    final_result_img = sudoku_solver_cv.run(image_path=image_path, debug=debug)

    # save the output image
    if save:
        output = f'result-{os.path.basename(image_path)}'
        cv2.imwrite(output, final_result_img)
        print(f'Main INFO: the output image has been successfully saved in path: {output}.')

    if display:
        # display the augmented solved sudoku image result
        cv2.imshow('Final Sudoku Result', final_result_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
