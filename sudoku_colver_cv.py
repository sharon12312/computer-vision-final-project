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

        # displays the sudoku board after using the OCR operation
        sudoku = Sudoku(board.tolist())
        print('# Sudoku Board #')
        sudoku.print()

        # solve the sudoku and display the solution if exists
        sudoku_solution = sudoku.solve_sudoku()
        if sudoku_solution is not None:
            print('# Sudoku Board Solution #')
            sudoku.print()
        else:
            print('Sudoku solution not found.')

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
                    text_j += start_j  # add ~30% to the start width position
                    text_i += end_i  # add ~20% to the start height position

                    # compute the font scale for the image
                    cell_height, cell_width, scale = end_i - start_i, end_j - start_j, 1
                    font_scale = min(cell_width, cell_height) / (50 / scale)

                    # draw the result digit on the sudoku warped transformed image
                    cv2.putText(img=warped_sudoku_board, text=str(digit), org=(text_j, text_i),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(205, 0, 0), thickness=2)

        # show the output image
        cv2.imshow('Sudoku Result', warped_sudoku_board)
        cv2.waitKey(0)


sudoku_solver_cv = SudokuSolverCV()
sudoku_solver_cv.run(image_path='./images/sudoku5.jpg', debug=False)
