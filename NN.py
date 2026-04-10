import numpy as np
import jax
import jax.numpy as jnp

class NN:

    @staticmethod
    def init(layer_sizes, activations):
        activation_funcs = tuple(map(NN.resolve_activation_function, activations))
        Ws = []
        bs = []
        for i in range(1, len(layer_sizes)):
            Ws.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]))
            bs.append(np.random.randn(layer_sizes[i]))

        return (Ws, bs, activation_funcs)

    @staticmethod
    def forward(x, Ws, bs, activations):
        y = x
        for i in range(len(Ws)):
            y = activations[i](y @ Ws[i] + bs[i])
        return y
            

    @staticmethod
    def resolve_activation_function(function_name):
        match function_name.upper():
            case 'RELU': return jax.nn.relu
            case 'SIGMOID': return jax.nn.sigmoid
            case 'SOFTMAX': return jax.nn.softmax
            case 'LINEAR': return lambda x: x
            case 'MH_RELU_SOFTMAX': return lambda x: jnp.concatenate([jax.nn.relu(x[..., :1]), jax.nn.softmax(x[..., 1:])], axis=-1)