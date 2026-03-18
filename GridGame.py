import numpy as np

class GridGame():

    def __init__(self, dim=3):
        board = np.zeros((dim, dim))
        board[0][0] = 2
        self.board = board        

    def get_init_state(self):
        return self.board
    
    # claude implemented
    @staticmethod
    def get_actions(state):
        actions = []
        for action in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
            rotations = {'LEFT': 0, 'RIGHT': 2, 'UP': 3, 'DOWN': 1}
            board = np.rot90(state.copy(), rotations[action])
            slid = np.array([GridGame.slide_row(row)[0] for row in board])
            if not np.array_equal(slid, board):
                actions.append(action)
        return actions
    
    # claude implemented
    @staticmethod
    def slide_row(row):
        r = row[row != 0]
        score, merged, i = 0, [], 0
        while i < len(r):
            if i + 1 < len(r) and r[i] == r[i+1]:
                merged.append(r[i] * 2)
                score += r[i] * 2
                i += 2
            else:
                merged.append(r[i])
                i += 1
        padded = merged + [0] * (len(row) - len(merged))
        return np.array(padded), score
    
    # claude implemented
    @staticmethod
    def perform_action(state, action):
        rotations = {'LEFT': 0, 'RIGHT': 2, 'UP': 3, 'DOWN': 1}
        board = np.rot90(state.copy(), rotations[action])

        score = 0
        for i in range(board.shape[0]):
            board[i], s = GridGame.slide_row(board[i])
            score += s

        board = np.rot90(board, -rotations[action] % 4)

        empty = list(zip(*np.where(board == 0)))
        if empty:
            pos = empty[np.random.randint(len(empty))]
            board[pos] = 2 if np.random.random() < 0.9 else 4

        isDone = not (np.any(board == 0) or
                    np.any(board[:, :-1] == board[:, 1:]) or
                    np.any(board[:-1] == board[1:]))

        return board, score, isDone
    
    @staticmethod
    def visualize_state(state):
        print(state)

    

# game = GridGame()
# state = game.get_init_state()
# print(GridGame.get_actions(state))
# GridGame.visualize_state(state)