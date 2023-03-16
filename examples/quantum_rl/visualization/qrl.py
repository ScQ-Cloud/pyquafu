import argparse
import re
import sys
from functools import reduce

import cirq
import gym
# model imports
import models.quantum_genotypes as genotypes
import numpy as np
import tensorflow as tf
from models.quantum_models import generate_circuit
from models.quantum_models import generate_model_policy as Network
from models.quantum_models import get_model_circuit_params
from PIL import Image
from quafu import QuantumCircuit as quafuQC
from quafu import Task, User

parser = argparse.ArgumentParser('Reinforcement learining with quantum computing cloud Quafu')
parser.add_argument('--env_name', type=str, default='CartPole-v1', help='environment name')
parser.add_argument('--state_bounds', type=np.array, default=np.array([2.4, 2.5, 0.21, 2.5]), help='state bounds')
parser.add_argument('--n_qubits', type=int, default=4, help='the number of qubits')
parser.add_argument('--n_actions', type=int, default=2, help='the number of actions')
parser.add_argument('--arch', type=str, default='NSGANet_id10', help='which architecture to use')
parser.add_argument('--shots', type=int, default=1000, help='the number of sampling')
parser.add_argument('--backend', type=str, default='ScQ-P20', help='which backend to use')
parser.add_argument('--model_path', type=str, default='./weights/weights_id10_quafu.h5', help='path of pretrained model')
args = parser.parse_args(args=[])


def get_res_exp(res):
    # access to probabilities of all possibilities 
    prob = res.probabilities
    sumexp = 0
    for k, v in prob.items():
        count = 0
        for i in range(len(k)):
            if k[i] == '1':
                count += 1
        if count % 2 == 0:
            sumexp += v
        else:
            sumexp -= v
    return sumexp


def get_quafu_exp(circuit):
    # convert Cirq circuts to qasm
    openqasm = circuit.to_qasm(header='')
    openqasm = re.sub('//.*\n', '', openqasm)
    openqasm = "".join([s for s in openqasm.splitlines(True) if s.strip()])
    
    # fill in with your token, register on website http://quafu.baqis.ac.cn/
    user = User()
    user.save_apitoken(" ")
    
    # initialize to Quafu circuits
    q = quafuQC(args.n_qubits)
    q.from_openqasm(openqasm)
    
    # create the task
    task = Task()
    task.load_account()
    
    # choose sampling number and specific quantum devices
    shots = args.shots   
    task.config(backend=args.backend, shots=shots, compile=True)
    task_id = task.send(q, wait=True).taskid
    print('task_id:', task_id)
    
    # retrieve the result of completed tasks and compute expectations
    task_status = task.retrieve(task_id).task_status
    if task_status == 'Completed':
        task = Task()
        task.load_account()
        res = task.retrieve(task_id)
        OB = get_res_exp(res)
    return task_id, tf.convert_to_tensor([[OB]])


class Alternating_(tf.keras.layers.Layer):
    def __init__(self, obsw):
        super(Alternating_, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant(obsw), dtype="float32", trainable=True, name="obsw")

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


def get_obs_policy(obsw):
    process = tf.keras.Sequential([ Alternating_(obsw),
                                    tf.keras.layers.Lambda(lambda x: x * 1.0),
                                    tf.keras.layers.Softmax()
                                ], name="obs_policy")
    return process


if __name__ == "__main__":
    qubits = cirq.GridQubit.rect(1, args.n_qubits)
    genotype = eval("genotypes.%s" % args.arch)
    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3
    model = Network(qubits, genotype, args.n_actions, observables)
    model.load_weights(args.model_path)

    # update gym to the version having render_mode, which is 0.26.1 in this file
    # initialize the environment
    env = gym.make(args.env_name, render_mode="rgb_array")
    state, _ = env.reset()
    frames = []

    # set the number of episodes
    for epi in range(20):
        im = Image.fromarray(env.render())
        frames.append(im)  
    
        # get PQC model parameters and expectations
        stateb = state/args.state_bounds
        newtheta, newlamda = get_model_circuit_params(qubits, genotype, model)
        circuit, _, _ = generate_circuit(qubits, genotype, newtheta, newlamda, stateb)
        _, expectation = get_quafu_exp(circuit)
    
        # get policy model parameters
        obsw = model.get_layer('observables-policy').get_weights()[0]
        obspolicy = get_obs_policy(obsw)
        policy = obspolicy(expectation)
        print('policy:', policy)
    
        # choose actions and make a step
        action = np.random.choice(args.n_actions, p=policy.numpy()[0])
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            print(epi+1)
            break
    env.close()

    # save to your path
    frames[1].save(' ', save_all=True, append_images=frames[2:], optimize=False, duration=20, loop=0)
    # for example, ./visualization/id10_quafu_gif/gym_CartPole_20.gif
