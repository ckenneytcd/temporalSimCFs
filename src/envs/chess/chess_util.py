import chess
import numpy as np
import pandas as pd

FEATURE_NAMES = chess.SQUARE_NAMES + ['player']
COL_NAMES = FEATURE_NAMES + ['best_move']

CODE_TO_PIECE = {
    1: 'P',
    2: 'N',
    3: 'B',
    4: 'R',
    5: 'Q',
    6: 'K',
    7: 'p',
    8: 'n',
    9: 'b',
    10: 'r',
    11: 'q',
    12: 'k'
}

ROWS = [1, 2, 3, 4, 5, 6, 7, 8]
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
SQUARE_NAMES = [f + str(r) for r in ROWS for f in FILES]
ACTION_NAMES = [s1 + s2 for s1 in SQUARE_NAMES for s2 in SQUARE_NAMES if s1 != s2] + [''] # adding impossible situation
PROMOTIONAL_MOVES = [f + str(r) + f1 + str(r - 1) for f in FILES for f1 in FILES for r in [2] if abs(ord(f)-ord(f)) <= 1] + \
                    [f + str(r) + f1 + str(r + 1) for f in FILES for f1 in FILES for r in [7] if abs(ord(f)-ord(f)) <= 1]
PROMOTIONAL_PIECES = ['r', 'b', 'n', 'q', 'R', 'B', 'N', 'Q']
ACTION_NAMES += [m + p for m in PROMOTIONAL_MOVES for p in PROMOTIONAL_PIECES]


def from_board_to_fen(board):
    ''' Transforms the board from a 65 element array (64 squares + player) into a fen descriptor '''
    board = [int(elem) for elem in board]
    num_rows = 8
    fen = ''
    r = 0
    while r < num_rows:
        c = 0
        while c < num_rows:
            curr_index = num_rows * (7 - r) + c
            if int(board[curr_index]) == 0:
                blank_counter = 0

                while int(board[curr_index]) == 0:
                    blank_counter += 1
                    c += 1
                    if c >= num_rows:
                        break

                    curr_index = num_rows * (7 - r) + c

                fen += str(blank_counter)
            else:
                fen += CODE_TO_PIECE[int(board[curr_index])]
                c += 1

        r += 1
        if r < num_rows:  # not the last row
            fen += '/'

    player = 'w' if board[64] == 0 else 'b'
    fen += ' ' + player + ' '
    fen += '- - 0 1'  # hardcoding the last parameters as they are just about number of moves and castling options

    return fen


def from_fen_to_board(fen):
    ''' Transforms the fen of the board into a 65-element array (64 squares + 1 player descriptor)'''
    # create board using fen code
    board = chess.Board(fen)

    # transform into 64 array with pieces + 1 element for whose move it is
    board_entry = np.zeros((65,), dtype=int)
    for piece in chess.PIECE_TYPES:
        for color in chess.COLORS:
            piece_sqset = board.pieces(piece, color)

            # get locations on board of specific pieces
            piece_locs = list(piece_sqset)
            board_entry[piece_locs] = int(piece + (1-color) * len(chess.PIECE_TYPES))  # color is binary, white pieces have ids 1-6, black pieces 7-12

    player = fen.split(' ')[1]
    board_entry[-1] = 0 if player == 'w' else 1
    return board_entry


def generate_neighborhood(board, n_samples, step, count):
    ''' Generates neighbourhood around a board by removing/adding pieces (for Growing Spheres) '''
    board = board.squeeze()
    board = np.array([int(elem) for elem in board])
    replicated_board = np.repeat(
        board.reshape(1, -1),
        n_samples,
        axis=0,
    )

    # cannot remove or replace kings in the board or change whose move it is
    immutable_mask = (replicated_board == 12) | (replicated_board == 6)
    immutable_mask[:, -1] = True  # cant change player
    immutable_mask_inverse = 1 - immutable_mask

    # random sampling of categorical variables
    binary_mask = np.random.choice(
        np.arange(0, 2), p=[min(1, step * count), max(0, 1 - (step * count))],
        size=n_samples * board.shape[-1]
    ).reshape(n_samples, -1)

    change_mask = np.random.choice(
        np.arange(0, 7), p=[0.99, 0.002, 0.002, 0.002, 0.002, 0.002, 0],
        size=n_samples * board.shape[-1]
    ).reshape(n_samples, -1)

    change_binary_mask = change_mask == 0

    # perform changes
    neighborhood = (change_mask + board * change_binary_mask) * binary_mask

    # rewrite immutable features from fact
    neighborhood = neighborhood * immutable_mask_inverse + board * immutable_mask

    df = pd.DataFrame(neighborhood, columns=FEATURE_NAMES)

    return df

def is_valid_fen(fen):
    ''' Checks if the string is a valid fen descriptor '''
    white_kings = fen.count('K')
    black_kings = fen.count('k')

    # check number of kings
    if white_kings != 1 or black_kings != 1:
        return False

    rows = fen.split("/")
    first_row = rows[0]
    last_row = rows[7]

    # check no pawns on first an last row
    if first_row.count('p') > 0 or first_row.count('P') > 0 or last_row.count('p') > 0 or last_row.count('P') > 0:
        return False

    return True

def is_playable_fen(fen):
    ''' Checks if string represents a fen of a playable position -- player can make a move'''
    if not is_valid_fen(fen):
        return False

    board = chess.Board(fen)
    if board.is_game_over():
        return False

    # flip current player to check if they are in check
    board.turn = not board.turn
    check = board.is_check()

    # if opponent is in check, the position is not playable
    return not check