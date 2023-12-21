# update gym to the version having render_mode, which is 0.26.1 in this file
import argparse
import sys

sys.path.insert(0, " ")
from functools import reduce

import cirq
import gym
import models.quantum_genotypes as genotypes
import numpy as np
import tensorflow as tf
from misc.utils import get_obs_policy, get_quafu_exp
from models.quantum_models import generate_circuit
from models.quantum_models import generate_model_policy as Network
from models.quantum_models import get_model_circuit_params
from PIL import Image

parser = argparse.ArgumentParser(
    "Plot gif of pre-trained quantum models on quafu cloud platform"
)
parser.add_argument(
    "--env_name", type=str, default="CartPole-v1", help="environment name"
)
parser.add_argument(
    "--state_bounds",
    type=np.array,
    default=np.array([2.4, 2.5, 0.21, 2.5]),
    help="state bounds",
)
parser.add_argument("--n_qubits", type=int, default=4, help="the number of qubits")
parser.add_argument("--n_actions", type=int, default=2, help="the number of actions")
parser.add_argument(
    "--arch", type=str, default="NSGANet_id10", help="which architecture to use"
)
parser.add_argument("--shots", type=int, default=1000, help="the number of sampling")
parser.add_argument(
    "--backend_quafu", type=str, default="ScQ-P10", help="which quafu backend to use"
)
parser.add_argument("--beta", type=float, default=1.0, help="output parameter")
parser.add_argument(
    "--model_path",
    type=str,
    default="./weights/train_p10/weights_id10_quafu_132.h5",
    help="path of pretrained model",
)
args = parser.parse_args(args=[])

if __name__ == "__main__":
    qubits = cirq.GridQubit.rect(1, args.n_qubits)
    genotype = eval("genotypes.%s" % args.arch)
    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)]  # Z_0*Z_1*Z_2*Z_3
    model = Network(
        qubits, genotype, args.n_actions, args.beta, observables, args.env_name
    )
    model.load_weights(args.model_path)

    for epoch in range(20):
        env = gym.make(args.env_name, render_mode="rgb_array")
        state, _ = env.reset()
        frames = []
        for epi in range(100):
            im = Image.fromarray(env.render())
            frames.append(im)

            # get PQC model parameters and expectations
            stateb = state / args.state_bounds
            newtheta, newlamda = get_model_circuit_params(qubits, genotype, model)
            circuit, _, _ = generate_circuit(
                qubits, genotype, newtheta, newlamda, stateb
            )
            _, expectation = get_quafu_exp(
                circuit, args.n_qubits, args.backend_quafu, args.shots
            )

            # get policy model parameters
            obsw = model.get_layer("observables-policy").get_weights()[0]
            obspolicy = get_obs_policy(obsw, args.beta)
            policy = obspolicy(expectation)
            print("policy:", policy)

            # choose actions and make a step
            action = np.random.choice(args.n_actions, p=policy.numpy()[0])
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print(epi + 1)
                break
        env.close()

        # save gif to your path
        frames[1].save(
            "./visualization/gif/test_{}.gif".format(epoch),
            save_all=True,
            append_images=frames[2:],
            optimize=False,
            duration=40,
            loop=0,
        )
