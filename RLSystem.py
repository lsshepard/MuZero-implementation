from GridGame import GridGame
from MCTS import MCTS

class RLSystem():

    def __init__(self, game: GridGame, search_alg):
        self.game = game
        self.GameClass = type(game)
        self.search_alg = search_alg

    def run(self):
        state = self.game.get_init_state()
        self.GameClass.visualize_state(state)
        isDone = False
        i = 0
        while not isDone and i < 10:
            i += 1
            action = self.search_alg(self.GameClass, state)
            state, R, isDone = self.GameClass.perform_action(state, action)
            print(action)
            self.GameClass.visualize_state(state)


game = GridGame()
rl_system = RLSystem(game, MCTS)

rl_system.run()