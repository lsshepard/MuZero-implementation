from NN import NN
import numpy as np

class MCTSNode():

    c = 0.2

    def __init__(self, state, actions, parent=None, parent_action=None, R=None, is_root=False, GameClass=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.R = R
        self.visits = 1
        self.rewards = 0
        self.unexplored_actions = actions
        self.children = []

        if is_root and GameClass:
            for action in actions:
                child_state, R, isDone = GameClass.perform_action(self.state, action)
                child_node = MCTSNode(child_state, GameClass.get_actions(child_state), parent=self, parent_action=action, R=R)
                self.children.append(child_node)
            self.unexplored_actions = []

    
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

def u_MCTS(GameClass, state, iterations, NN_P, max_depth):
    
    available_actions = GameClass.get_actions(state)
    root = MCTSNode(state, available_actions, is_root=True, GameClass=GameClass)

    for i in range(iterations):
        node, d = select_node(root)
        
        leaf_node, R, isDone = node.expand(GameClass)
        
        accum_reward = do_rollout(GameClass, leaf_node.state, R, isDone, NN_P, max_depth-d)
        
        backpropegate(leaf_node, accum_reward)
    

    pi_uf = np.array(list(map(lambda c: c.visits, root.children)))
    sampled_node_i = np.random.choice(len(pi_uf), p=pi_uf/sum(pi_uf))
    sampled_node = root.children[sampled_node_i]
    action = sampled_node.parent_action

    pi = np.array(list(pi_uf[available_actions.index(action_name)] if action_name in available_actions else 0 for action_name in GameClass.get_all_actions()))
    pi_normalized = pi/sum(pi)

    data = [state, root.rewards, pi_normalized, sampled_node.R]

    return action, data

    
def select_node(root: MCTSNode):
    node = root
    d = 0
    while not node.unexplored_actions and node.children:
        d += 1
        node = max(node.children, key=lambda c: c.U(node.visits))
    return node, d

def do_rollout(GameClass, init_state, init_R, init_isDone, NN_P, depth):

    state, accum_reward, isDone = init_state, [init_R], init_isDone
    for d in range(depth):
        if isDone: break
        prediction = NN.forward(state.flatten(), *NN_P)
        v = prediction[0]
        pi = np.array(prediction[1:])
        total = pi.sum()
        p = pi / total if total > 0 else np.ones(len(pi)) / len(pi)
        p[-1] = 1.0 - p[:-1].sum()
        action = np.random.choice(len(pi), p=p)
        state, R, isDone = GameClass.perform_action(state, action)
        accum_reward.append(R)
    
    prediction = NN.forward(state.flatten(), *NN_P)
    v = prediction[0]
    pi = prediction[1:]
    accum_reward.append(v)

    return accum_reward

def backpropegate(leaf_node, accum_reward):
    node = leaf_node
    R = sum(accum_reward)
    while node:
        node.visits += 1
        node.rewards += R
        node = node.parent

        