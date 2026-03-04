from GridGame import GridGame
from MCTS import MCTS, MCTSNode

class RLSystem():

    def __init__(self, game: GridGame, search_alg, search_iterations):
        self.game = game
        self.GameClass = type(game)
        self.search_alg = search_alg
        self.search_iterations = search_iterations

    def run(self):
        state = self.game.get_init_state()
        self.GameClass.visualize_state(state)
        isDone = False
        i = 0
        root = MCTSNode(state, self.GameClass.get_actions(state))
        while not isDone and i < 100:
            i += 1
            root = self.search_alg(self.GameClass, root, self.search_iterations)
            action = root.parent_action
            state, R, isDone = self.GameClass.perform_action(state, action)
            print(action)
            self.GameClass.visualize_state(state)


game = GridGame(dim=4)
rl_system = RLSystem(game, MCTS, 1500)

rl_system.run()