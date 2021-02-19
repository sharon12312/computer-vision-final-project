import numpy as np
import cv2

from utils.image_processing import ImageProcessing
from neural_networks.cnn import DigitsNet
from sudoku.backtracking import Sudoku


class SudokuSolverCV:

    def __init__(self):
        # initial variables
        self._img_processing = ImageProcessing()
        self._digits_net = DigitsNet(model_path='models/digits_classifier.h5')

        # initial settings
        self._MAX_ROW = 9
        self._MAX_COL = 9

    def scan_sudoku(self):
        pass

    def character_recognition_and_cell_coordinates(self):
        pass

    def solve_sudoku_matrix(self):
        pass

    def add_solution_to_image_grid(self):
        pass

    def overlay_solution_image_on_the_original_image(self):
        pass

    def run(self, image_path, debug=False):
        # load the input image
        image = cv2.imread(image_path)

        # find the sudoku contours and the warped images (in color and grayscale)
        warped_sudoku_board, warped_sudoku_board_gray, sudoku_board_coords = \
            self._img_processing.find_sudoku_board_contours(image, debug)

        # initial the sudoku board matrix
        board = np.zeros((self._MAX_ROW, self._MAX_COL), dtype=np.int)

        # computes the steps pixels according to the grid image board
        height_step = warped_sudoku_board_gray.shape[0] // self._MAX_ROW
        width_step = warped_sudoku_board_gray.shape[1] // self._MAX_COL
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
                cell = warped_sudoku_board_gray[start_i:end_i, start_j:end_j]

                # extract the digit from the cell
                digit = self._img_processing.get_digit(cell, debug)

                # verify that their is a digit withing the cell
                if digit is not None:
                    digit_pred = self._digits_net.predict(digit)
                    board[i, j] = digit_pred

            # store the coordinate's cell
            cell_coordinates.append(row)

        print(Sudoku(board.tolist()).print())


sudoku_solver_cv = SudokuSolverCV()
sudoku_solver_cv.run(image_path='./images/sudoku1.jpg', debug=False)
