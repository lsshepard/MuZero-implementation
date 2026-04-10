from GridGame import GridGame
from u_MCTS import u_MCTS, MCTSNode
from NN import NN
import jax
import jax.numpy as jnp
import numpy as np
import random
import matplotlib.pyplot as plt

class RLSystem():

    def __init__(self, GameClass, search_alg, game_config, search_iterations):
        self.GameClass = GameClass
        self.game_config = game_config
        self.search_alg = search_alg
        self.search_iterations = search_iterations
        self.key = jax.random.PRNGKey(42)
        self.grad_fn = jax.jit(jax.value_and_grad(mb_loss), static_argnums=(1,))
        self.loss_hist = []

    def episode_loop(self, N_e, I_t, max_depth, mbs, lr):
        EH = []
        input_size = self.GameClass.get_dims_in(self.game_config)
        output_size = 5
        NN_P = NN.init([input_size, 24, 16, output_size], ['RELU', 'RELU', 'MH_RELU_SOFTMAX'])
        for e in range(N_e):
            print("EPISODE", e+1)
            epidata = self.simluate_episode(NN_P, max_depth)
            EH.append(epidata)
            if (e+1) % I_t == 0:
                print('DOING BPTT')
                NN_P = self.BPTT(NN_P, EH, mbs, lr)

        plt.plot(self.loss_hist)
        plt.grid()
        plt.show()

    def simluate_episode(self, NN_P, max_depth):
        epidata = []
        state = self.GameClass.get_init_state()
        # self.GameClass.visualize_state(state)
        isDone = False
        i = 0
        while not isDone:
            # print('STEP', i)
            i += 1
            action, action_data = self.search_alg(self.GameClass, state, self.search_iterations, NN_P, max_depth)
            epidata.append(action_data)
            state, R, isDone = self.GameClass.perform_action(state, action)
            # print(action, R)
            # self.GameClass.visualize_state(state)
        return epidata
    
    def BPTT(self, NN_P, EH, mbs, lr):
        NN_P_params = NN_P[:2]
        NN_P_activations = NN_P[2]

        roll_ahead = 10

        states = []
        vs = []
        pis = []
        for m in range(mbs):
            episode = random.choice(EH)
            starting_i = np.random.randint(0, len(episode)-10)
            # samples.append((episode, starting_i))
            sample_states = jnp.stack([episode[starting_i + w][0] for w in range(roll_ahead)])
            sample_vs = jnp.stack([episode[starting_i + w][1] for w in range(roll_ahead)])
            sample_pis = jnp.stack([episode[starting_i + w][2] for w in range(roll_ahead)])
            states.append(sample_states)
            vs.append(sample_vs)
            pis.append(sample_pis)

        loss, grads = self.grad_fn(NN_P_params, NN_P_activations, jnp.array(states), jnp.array(vs), jnp.array(pis))
        print(loss)

        self.loss_hist.append(loss)
        NN_P_params = jax.tree.map(lambda p, g: p - g * lr, NN_P_params, grads)
        return [*NN_P_params, NN_P_activations]


def mb_loss(NN_P_params, NN_P_activations, states, vs, pis):

    batched_loss = jax.vmap(loss, in_axes=(None, None, 0, 0, 0))
    total = batched_loss(NN_P_params, NN_P_activations, states, vs, pis)
    return total.mean()


def loss(NN_P_params, NN_P_activations, sample_states, sample_vs, sample_pis):
        
    def step_fn(carry, step_data):
        loss = carry
        state, v, pi = step_data
        prediction = NN.forward(state.flatten(), NN_P_params[0], NN_P_params[1], NN_P_activations)
        v_hat = prediction[0]
        pi_hat = prediction[1:]
        v_loss = jnp.power(v-v_hat, 2)
        pi_loss = -jnp.sum(pi * jax.nn.log_softmax(pi_hat))
        loss += v_loss + pi_loss

        return loss, None

    total_loss, _ = jax.lax.scan(step_fn, 0.0, (sample_states, sample_vs, sample_pis))
    total_loss /= len(sample_states)

    return total_loss
            
                



game_config = 4
rl_system = RLSystem(GridGame, u_MCTS, game_config, 20)

rl_system.episode_loop(3, 1, 10, 30, 1e-14)