import numpy as np

class GridGame():

    def __init__(self, dim=3):
        self.dim = dim
        self.position = (0, 0)
        

    def get_init_state(self):
        return (self.position, self.dim)
    
    @staticmethod
    def get_actions(state):
        position, dim = state
        x, y = position
        actions = []
        if x > 0: actions.append('LEFT')
        if x < dim-1: actions.append('RIGHT')
        if y > 0: actions.append('UP')
        if y < dim-1: actions.append('DOWN')
        
        return actions
    
    # assumes legal action
    @staticmethod
    def perform_action(state, action):
        position, dim = state
        x, y = position
        match action:
            case 'RIGHT': x += 1
            case 'LEFT': x -= 1
            case 'UP': y -= 1
            case 'DOWN': y += 1

        R, isDone = (1, True) if x == dim - 1 and y == dim - 1 else (0, False)
        updated_state = ((x, y), dim)

        return updated_state, R, isDone
    
    @staticmethod
    def visualize_state(state):
        position, dim = state
        x, y = position
        board = np.zeros((dim, dim))
        board[x][y] = 1
        board[-1][-1] = -1
        print(board)

    

# game = GridGame()
# state = game.get_init_state()
# print(GridGame.get_actions(state))
# GridGame.visualize_state(state)