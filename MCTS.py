import numpy as np

class MCTSNode():

    c = 0.2

    def __init__(self, state, actions, parent=None, parent_action=None, R=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.R = R
        self.visits = 0
        self.rewards = 0
        self.unexplored_actions = actions
        self.children = []

    
    def U(self, N):
        Q = self.rewards / self.visits
        UCB1 = Q + self.c * np.sqrt(np.log(N) / self.visits)
        return UCB1
    
    def expand(self, GameClass):
        if len(self.unexplored_actions) == 0:
            return self, self.R, True
        random_index = np.random.randint(0, len(self.unexplored_actions))
        action = self.unexplored_actions.pop(random_index)
        child_state, R, isDone = GameClass.perform_action(self.state, action)
        child_node = MCTSNode(child_state, GameClass.get_actions(child_state), parent=self, parent_action=action, R=R)
        self.children.append(child_node)

        return child_node, R, isDone



def MCTS(GameClass, state):
    root = MCTSNode(state, GameClass.get_actions(state))
    for i in range(500):
        node = select_node(root)
        leaf_node, R, isDone = node.expand(GameClass)
        R = simulate(GameClass, leaf_node.state, R, isDone)
        backpropegate(leaf_node, R)
    
    best_choice = max(root.children, key=lambda c: c.visits)
    return best_choice.parent_action


    
def select_node(root: MCTSNode):
    node = root
    while not node.unexplored_actions and node.children:
        node = max(node.children, key=lambda c: c.U(node.visits))
    return node


def simulate(GameClass, init_state, init_R, init_isDone):

    state, R, isDone = init_state, init_R, init_isDone
    while not isDone:
        actions = GameClass.get_actions(state)
        action = np.random.choice(actions)
        state, R, isDone = GameClass.perform_action(state, action)

    return R

def backpropegate(leaf_node, R):
    node = leaf_node
    while node:
        node.visits += 1
        node.rewards += R
        node = node.parent

        