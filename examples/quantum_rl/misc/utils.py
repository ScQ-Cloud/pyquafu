import os
import re
import shutil
from collections import defaultdict

import gym
import numpy as np
import tensorflow as tf
from models.quantum_models import generate_circuit, get_model_circuit_params

from quafu import QuantumCircuit as quafuQC
from quafu import Task, User


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir : {}".format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


def get_res_exp(res):
    """
    Access to probabilities of all possibilities, observable specifies to Z_0*Z_1*Z_2*Z_3.
    """
    prob = res.probabilities
    sumexp = 0
    for k, v in prob.items():
        count = 0
        for i in range(len(k)):
            if k[i] == "1":
                count += 1
        if count % 2 == 0:
            sumexp += v
        else:
            sumexp -= v
    return sumexp


def get_quafu_exp(circuit, n_qubits, backend_quafu, shots):
    """
    Execute circuits on quafu cloud platform and return the expectation.
    """
    # convert Cirq circuts to qasm
    openqasm = circuit.to_qasm(header="")
    openqasm = re.sub("//.*\n", "", openqasm)
    openqasm = "".join([s for s in openqasm.splitlines(True) if s.strip()])

    # fill in with your token, register on website http://quafu.baqis.ac.cn/
    user = User()
    user.save_apitoken(" ")

    # initialize to Quafu circuits
    q = quafuQC(n_qubits)
    q.from_openqasm(openqasm)

    # create the task
    task = Task()

    task.config(backend_quafu, shots, compile=True, priority=3)
    task_id = task.send(q, wait=True).taskid
    print("task_id:", task_id)

    # retrieve the result of completed tasks and compute expectations
    task_status = task.retrieve(task_id).task_status
    if task_status == "Completed":
        task = Task()
        res = task.retrieve(task_id)
        OB = get_res_exp(res)
    return task_id, tf.convert_to_tensor([[OB]])


def get_compiled_gates_depth(circuit, n_qubits, backend_quafu, shots):
    """
    Get the gates and layered circuits of compiled circuits.
    """
    openqasm = circuit.to_qasm(header="")
    openqasm = re.sub("//.*\n", "", openqasm)
    openqasm = "".join([s for s in openqasm.splitlines(True) if s.strip()])

    user = User()
    user.save_apitoken(" ")

    q = quafuQC(n_qubits)
    q.from_openqasm(openqasm)

    task = Task()

    task.config(backend_quafu, shots, compile=True)
    task_id = task.send(q, wait=True).taskid
    print("task_id:", task_id)

    task_status = task.retrieve(task_id).task_status
    if task_status == "Completed":
        task = Task()
        res = task.retrieve(task_id)
        gates = res.transpiled_circuit.gates
        layered_circuit = res.transpiled_circuit.layered_circuit()
    return task_id, gates, layered_circuit


class Alternating_(tf.keras.layers.Layer):
    """
    Load observable weights of pre-trained models.
    """

    def __init__(self, obsw):
        super(Alternating_, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant(obsw),
            dtype="float32",
            trainable=True,
            name="obsw",
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


def get_obs_policy(obsw, beta):
    """
    Output the final policy.
    """
    process = tf.keras.Sequential(
        [
            Alternating_(obsw),
            tf.keras.layers.Lambda(lambda x: x * beta),
            tf.keras.layers.Softmax(),
        ],
        name="obs_policy",
    )
    return process


def get_height(position):
    """
    Get the height of position in MountainCar-v0.
    """
    return np.sin(3 * position) * 0.45 + 0.55


def gather_episodes(
    state_bounds,
    n_actions,
    model,
    n_episodes,
    env_name,
    beta,
    backend,
    backend_quafu="ScQ-P10",
    shots=1000,
    n_qubits=4,
    qubits=None,
    genotype=None,
):
    """
    Interact with environment, you can choose the backend between `cirq` simulator and `quafu` cloud platform.
    """
    trajectories = [defaultdict(list) for _ in range(n_episodes)]
    envs = [gym.make(env_name) for _ in range(n_episodes)]

    done = [False for _ in range(n_episodes)]
    states = [e.reset() for e in envs]

    tasklist = []

    while not all(done):
        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = [
            s / state_bounds for i, s in enumerate(states) if not done[i]
        ]
        # height = [get_height(s[0]) for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            trajectories[i]["states"].append(state)

        # Compute policy for all unfinished envs in parallel
        states = tf.convert_to_tensor(normalized_states)

        if backend == "cirq":
            action_probs = model([states])
        elif backend == "quafu":
            newtheta, newlamda = get_model_circuit_params(qubits, genotype, model)
            circuit, _, _ = generate_circuit(
                qubits, genotype, newtheta, newlamda, states.numpy()[0]
            )
            taskid, expectation = get_quafu_exp(circuit, n_qubits, backend_quafu, shots)
            tasklist.append(taskid)
            # print('gather_episodes_exp:', expectation)

            obsw = model.get_layer("observables-policy").get_weights()[0]
            obspolicy = get_obs_policy(obsw, beta)
            action_probs = obspolicy(expectation)
        else:
            print("This backend is not supported now.")

        # Store action and transition all environments to the next state
        states = [None for i in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            trajectories[i]["action_probs"].append(policy)
            action = np.random.choice(n_actions, p=policy)
            states[i], reward, done[i], _ = envs[i].step(action)
            trajectories[i]["actions"].append(action)
            if env_name == "CartPole-v1":
                trajectories[i]["rewards"].append(reward)
            elif env_name == "MountainCar-v0":
                trajectories[i]["rewards"].append(reward + get_height(states[i][0]))
            else:
                print("This environment is not supported now.")

    return tasklist, trajectories


def compute_returns(rewards_history, gamma):
    """
    Compute discounted returns with discount factor `gamma`.
    """
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize them for faster and more stable learning
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    returns = returns.tolist()

    return returns
