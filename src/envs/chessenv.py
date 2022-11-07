import gym
import chess


class ChessEnv(gym.Env):

    def __init__(self, expert):
        self.board = chess.Board()
        self.state = self.from_fen_to_array(self.board.fen())

        self.state_dim = 64 + 1

        self.expert = expert

    def step(self, action):
        move = chess.Move.from_uci(action)
        self.board.push(move)

        new_state = self.from_fen_to_array(self.board.fen())

        done = self.board.is_game_over()

        rew = self.calculate_reward(self.state, new_state)

        self.state = new_state

        return self.state, rew, done, {}

    def render(self):
        print(self.board)

    def render_state(self, state):
        render_board = chess.Board(state)
        print(render_board)

    def reset(self):
        self.board = chess.Board()
        self.state = self.from_fen_to_array(self.board.fen())


        return self.state

    def close(self):
        pass

    def realistic(self, x):
        fen = self.from_array_to_fen(x)
        board = chess.Board(fen)

        return board.is_valid()

    def actionable(self, x, fact):
        return True

    def generate_state_from_json(self, json_dict):
        fen = json_dict['fen']

        state = self.from_fen_to_array(fen)

        return state

    def get_actions(self, state):
        board = chess.Board(state)
        return board.legal_moves

    def set_state(self, state):
        self.board = chess.Board(state)

    def check_done(self, state):
        fen = self.from_array_to_fen(state)
        board = chess.Board(fen)

        return board.is_game_over()

    def equal_states(self, s1, s2):
        return sum(s1.squeeze() != s2.squeeze())

    def calculate_reward(self, state, new_state):
        return self.expert.analyse(new_state).score - self.expert.analyse.eval(state).score

    def from_array_to_fen(self, array):
        board = chess.Board()

        for i, square in enumerate(chess.SQUARE_NAMES):
            piece_type = array[i] % 6
            piece_color = array[i] > 0
            piece = chess.Piece(piece_type, piece_color)
            board.set_piece_at(square, piece)

        turn = array[-1]
        board.turn = turn

        return board.fen()

    def from_fen_to_array(self, fen):
        board = chess.Board(fen)
        array = []

        for square in chess.SQUARE_NAMES:
            piece = board.piece_at(square)
            array.append(piece.piece_type + piece.color * 6)

        array.append(board.turn)

        return array