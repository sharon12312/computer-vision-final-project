import numpy as np


class Sudoku:
    """
    The class uses a backtracking method.
    The solution can be done by assigning a number, but before it, we need to check whether it is safe to assign
    and if the same number is not present in the current row, current column, and the current 3X3 sub-grid.
    After checking for safety, assign the number, and recursively check whether this assignment leads to a solution or
    not. If the assignment does not lead to a solution, then try the next number for the current empty cell.
    And if none of the numbers (1 to 9) leads to a solution, return false.

    Code example:
        sudoku_board = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
                        [5, 2, 0, 0, 0, 0, 0, 0, 0],
                        [0, 8, 7, 0, 0, 0, 0, 3, 1],
                        [0, 0, 3, 0, 1, 0, 0, 8, 0],
                        [9, 0, 0, 8, 6, 3, 0, 0, 5],
                        [0, 5, 0, 0, 9, 0, 6, 0, 0],
                        [1, 3, 0, 0, 0, 0, 2, 5, 0],
                        [0, 0, 0, 0, 0, 0, 0, 7, 4],
                        [0, 0, 5, 2, 0, 6, 3, 0, 0]]
        sudoku = Sudoku(sudoku_board)
        sudoku.solve()
        sudoku.print()
    """
    def __init__(self, board):
        # setting the sudoku board variable
        self._board = board

        # save the initial board boolean settings
        self._initial_board = self._set_initial_board_bool()

        # initial variables
        self._MAX_ROW = 9
        self._MAX_COL = 9
        self._MAX_BOX_ROW = 3
        self._MAX_BOX_COL = 3

    def _set_initial_board_bool(self):
        """
        Convert the initial sudoku board to a boolean type.
        :return: ndarray
        """
        initial_board = np.array(self._board.copy(), dtype=bool)
        return initial_board

    def _get_empty_location(self, location):
        """
        Finds the entry in the board that is still not used.
        The function searches in the sudoku board to find an entry that is still unassigned.
        If found, the reference parameters row and col (column) will be set the unassigned location, and return True.
        If no unassigned entries remain, false is returned.

        :param location: an initial position in the sudoku board
        :return: boolean, if a location has been found then return True, otherwise return False
        """
        for row in range(self._MAX_ROW):
            for col in range(self._MAX_COL):
                if self._board[row][col] == 0:
                    location[0] = row
                    location[1] = col
                    return True
        return False

    def _validate_row(self, row, num):
        """
        Indicates whether any assigned entry in the specified row matches the given number.

        :param row: integer, position in the row
        :param num: integer, the verified number in the given row
        :return: boolean, if the number has been found return True, otherwise False
        """
        for i in range(self._MAX_ROW):
            if self._board[row][i] == num:
                return True
        return False

    def _validate_col(self, col, num):
        """
        Indicates whether any assigned entry in the specified column matches the given number.

        :param col: integer, position in the column
        :param num: integer, the verified number in the given row
        :return: boolean, if the number has been found return True, otherwise False
        """
        for i in range(self._MAX_COL):
            if self._board[i][col] == num:
                return True
        return False

    def _validate_box(self, row, col, num):
        """
        Indicates whether any assigned entry within the specified 3x3 box matches the given number.

        :param row: integer, position in the row
        :param col: integer, position in the column
        :param num: integer, the verified number in the given row
        :return: boolean, if the number has been found return True, otherwise False
        """
        for i in range(self._MAX_BOX_ROW):
            for j in range(self._MAX_BOX_COL):
                if self._board[i + row][j + col] == num:
                    return True
        return False

    def _is_safe(self, row, col, num):
        """
        Checks whether it will be legal to assign a specific number to the given row and column.
        Returns a boolean which indicates whether it will be legal to assign num to the given row and column location.

        :param row: integer, position in the row
        :param col: integer, position in the column
        :param num: integer, the verified number in the given row
        :return: boolean, if the number can be located in the given row and column then return True, otherwise False
        """
        return not self._validate_row(row, num) and \
               not self._validate_col(col, num) and \
               not self._validate_box(row - row % 3, col - col % 3, num)

    def _solve(self):
        """
        Gets a partially filled-in sudoku board and strives to assign values to all unassigned locations in such
        a way to meet the requirements for the sudoku solution (non-duplication across rows, columns, and boxes).

        :return: boolean, if the solution is valid return True, otherwise False
        """

        # initial location in the board
        location = [0, 0]

        # if there is no empty location available, return True
        if not self._get_empty_location(location):
            return True

        # assigning the location's values to the row and the column
        row, col = location

        # range over the digits from 1 to 9
        for num in range(1, 10):
            # validate the cell
            if self._is_safe(row, col, num):
                # preform tentative assignment
                self._board[row][col] = num

                # consider as success, if a solution exists
                if self._solve():
                    return True

                # consider as failure, and follow the next digit
                self._board[row][col] = 0

        # apply the backtracking method
        return False

    def solve_sudoku(self):
        """
        Solves the sudoku board.
        :return: list, return the solution array if exists, otherwise return None
        """
        if self._solve():
            return self.get_board()
        else:
            return None

    def print(self):
        """
        Display the sudoku board.
        :return: None
        """
        print(f'{__class__.__name__} INFO: sudoku board visualization.')
        for i in range(len(self._board)):
            if i % 3 == 0 and i != 0:
                print('-----------------------')

            for j in range(len(self._board[0])):
                if j % 3 == 0 and j != 0:
                    print(' | ', end='')

                if j == 8:
                    print(self._board[i][j])
                else:
                    print(self._board[i][j], end=' ')
        print('\n')

    def get_board(self):
        """
        :return: 2d-array, return the sudoku board
        """
        return self._board

    def get_initial_board(self):
        """
        :return: ndarray, return the initial sudoku board as boolean
        """
        return self._initial_board