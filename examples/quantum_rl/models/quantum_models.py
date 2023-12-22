import cirq
import models.quantum_genotypes as genotypes
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
from models.quantum_operations import *


def generate_circuit(qubits, genotype, newtheta=None, newlamda=None, state=None):
    """Prepare a NSGANet circuit on `qubits` with `genotype` layers."""
    op_vpqc, pos_vpqc = zip(*genotype.vpqc)
    op_dpqc, pos_dpqc = zip(*genotype.dpqc)
    op_entangle, pos_entangle = zip(*genotype.entangle)
    op_measure, pos_measure = zip(*genotype.measure)

    dict = {}
    for name, pos in zip(op_vpqc, pos_vpqc):
        dict[pos] = name
    for name, pos in zip(op_dpqc, pos_dpqc):
        dict[pos] = name
    for name, pos in zip(op_entangle, pos_entangle):
        dict[pos] = name
    for name, pos in zip(op_measure, pos_measure):
        dict[pos] = name
    length = len(dict)

    circuit = cirq.Circuit()
    params = []
    inputs = []
    p_count = 0
    i_count = 0
    for i in range(length):
        if dict[i] == "variationalPQC":
            cir, pa = OPS[dict[i]](qubits, p_count, newtheta)
            circuit += cir
            params += pa
            p_count += 1
        elif dict[i] == "dataencodingPQC":
            cir, inp = OPS[dict[i]](qubits, i, i_count, newlamda, state)
            circuit += cir
            inputs += inp
            i_count += 1
        elif dict[i] == "entanglement":
            cir = OPS[dict[i]](qubits)
            circuit += cir
        elif dict[i] == "measurement":
            pass
        else:
            raise NameError("Unknown quantum genotype operation")

    # Last varitional layer
    cir, pa = OPS["variationalPQC"](qubits, len(pos_vpqc), newtheta)
    circuit += cir
    params += pa

    return circuit, params, inputs


def get_model_circuit_params(qubits, genotype, model):
    """Get parameters from trained model"""
    theta, lamda = model.get_layer("nsganet_PQC").get_weights()
    theta = theta[0]

    _, theta_symbols, input_symbols = generate_circuit(qubits, genotype)
    symbols = [str(symb) for symb in theta_symbols + input_symbols]
    indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
    indices = list(indices.numpy())
    newindex = [indices.index(a) for a in sorted(indices)]

    newtheta = []
    newlamda = []
    for i in range(len(theta)):
        newtheta.append(theta[newindex[i]])
    for i in range(len(theta), len(theta) + len(lamda)):
        newlamda.append(lamda[newindex[i] - len(theta)])
    return newtheta, newlamda


class NSGANetPQC(tf.keras.layers.Layer):
    """Define NSGANet PQC based on keras layer"""

    def __init__(
        self, qubits, genotype, observables, activation="linear", name="nsganet_PQC"
    ):
        super(NSGANetPQC, self).__init__(name=name)
        self.n_qubits = len(qubits)
        _, pos_dpqc = zip(*genotype.dpqc)
        self.n_layers = len(pos_dpqc)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, genotype)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True,
            name="thetas",
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])


class Alternating(tf.keras.layers.Layer):
    """Apply action-specific weights."""

    def __init__(self, output_dim, env):
        super(Alternating, self).__init__()
        if env == "CartPole-v1":
            self.w = tf.Variable(
                initial_value=tf.constant([[(-1.0) ** i for i in range(output_dim)]]),
                dtype="float32",
                trainable=True,
                name="obs-weights",
            )
        elif env == "MountainCar-v0":
            self.w = tf.Variable(
                initial_value=tf.constant(
                    [
                        [(-1.0) ** i for i in range(output_dim)],
                        [(-1.0) ** i for i in range(output_dim)],
                        [(-1.0) ** i for i in range(output_dim)],
                    ]
                ),
                dtype="float32",
                trainable=True,
                name="obs-weights",
            )

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


def generate_model_policy(qubits, genotype, n_actions, beta, observables, env):
    """Generate a Keras model for a NSGANet PQC policy."""
    input_tensor = tf.keras.Input(
        shape=(len(qubits),), dtype=tf.dtypes.float32, name="input"
    )
    nsganet_pqc = NSGANetPQC(qubits, genotype, observables)([input_tensor])
    process = tf.keras.Sequential(
        [
            Alternating(n_actions, env),
            tf.keras.layers.Lambda(lambda x: x * beta),
            tf.keras.layers.Softmax(),
        ],
        name="observables-policy",
    )
    policy = process(nsganet_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

    return model
