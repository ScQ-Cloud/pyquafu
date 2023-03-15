# Note: If you want to train the model from scratch, please use the command 'pip install gym==0.18.0' first.
import argparse
import logging
import os
import sys
import time
from functools import reduce

import cirq
import gym
# model imports
import models.quantum_genotypes as genotypes
import numpy as np
import tensorflow as tf
from misc import utils
from models.quantum_models import generate_circuit
from models.quantum_models import generate_model_policy as Network
from models.quantum_models import get_model_circuit_params
from search.quantum_train_search import compute_returns, gather_episodes
from sympy import im
from visualization.qrl import get_obs_policy, get_quafu_exp

parser = argparse.ArgumentParser('Quantum RL Training')
parser.add_argument('--save', type=str, default='qEXP-quafu', help='experiment name')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--n_episodes', type=int, default=10, help='the number of episodes')
parser.add_argument('--infer_episodes', type=int, default=5, help='the number of infer episodes')
parser.add_argument('--gamma', type=float, default=1.0, help='discount parameter')
parser.add_argument('--env_name', type=str, default="CartPole-v1", help='environment name')
parser.add_argument('--state_bounds', type=np.array, default=np.array([2.4, 2.5, 0.21, 2.5]), help='state bounds')
parser.add_argument('--n_qubits', type=int, default=4, help='the number of qubits')
parser.add_argument('--n_actions', type=int, default=2, help='the number of actions')
parser.add_argument('--arch', type=str, default='NSGANet_id10', help='which architecture to use')
parser.add_argument('--epochs', type=int, default=5, help='num of training epochs')

args = parser.parse_args(args=[])
args.save = 'train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

qubits = cirq.GridQubit.rect(1, args.n_qubits)
genotype = eval("genotypes.%s" % args.arch)
ops = [cirq.Z(q) for q in qubits]
observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3

def main():
    logging.info("args = %s", args)

    model = Network(qubits, genotype, args.n_actions, observables)

    n_epochs = args.epochs

    optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

    # Assign the model parameters to each optimizer
    w_in, w_var, w_out = 1, 0, 2

    best_reward = 0
    for epoch in range(n_epochs):
        logging.info('epoch %d', epoch)

        reward_his = train(model, optimizer_in, optimizer_var, optimizer_out, w_in, w_var, w_out)
        # valid_reward = infer(model)

        # if np.mean(valid_reward) >= best_reward:
        #     model.save_weights(os.path.join(args.save, 'weights_id97_quafu.h5'))
        #     best_reward = np.mean(valid_reward)


# Training
def train(model, optimizer_in, optimizer_var, optimizer_out, w_in, w_var, w_out):

    @tf.function
    def reinforce_update(states, actions, returns, model):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            logits = model(states)
            # newtheta, newlamda = get_model_circuit_params(qubits, genotype, model)
            # circuit, _, _ = generate_circuit(qubits, genotype, newtheta, newlamda, states[0].numpy()[0])
            # expectation = get_quafu_exp(circuit)
            # print('update_exp:', expectation)

            # obsw = model.get_layer('observables-policy').get_weights()[0]
            # obspolicy = get_obs_policy(obsw)
            # logits = obspolicy(expectation)
            # print('update_policy:', logits)

            p_actions = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_actions)
            loss = tf.math.reduce_sum(-log_probs * returns) / args.batch_size
        grads = tape.gradient(loss, model.trainable_variables)
        for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
            optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

    episode_reward_history = []
    best_reward = 0
    for batch in range(args.n_episodes // args.batch_size):
        # Gather episodes
        tasklist, episodes = gather_episodes(args.state_bounds, args.n_actions, model, args.batch_size, args.env_name, qubits, genotype)
        logging.info(tasklist)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([compute_returns(ep_rwds, args.gamma) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        reinforce_update(states, id_action_pairs, returns, model)

        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))
        

        if episode_reward_history[-1] >= best_reward:
            model.save_weights(os.path.join(args.save, 'weights_id10_quafu.h5'))
            best_reward = episode_reward_history[-1]
        # elif episode_reward_history[-1] >= 30:
        #     model.save_weights(os.path.join(args.save, 'weights_id10_quafu.h5'))


        # avg_rewards = np.mean(episode_reward_history[-10:])

        logging.info('train finished episode: %f', (batch + 1) * args.batch_size)
        logging.info('train average rewards: %f', episode_reward_history[-1])
    
        if episode_reward_history[-1] >= 200.0:
            break
    return episode_reward_history


# def infer(model):
#     episode_reward_history = []
#     for batch in range(args.infer_episodes // args.batch_size):
#         # Gather episodes
#         episodes = gather_episodes(args.state_bounds, args.n_actions, model, args.batch_size, args.env_name, qubits, genotype)

#         # Group states, actions and returns in numpy arrays
#         rewards = [ep['rewards'] for ep in episodes]

#         # Store collected rewards
#         for ep_rwds in rewards:
#             episode_reward_history.append(np.sum(ep_rwds))

#         avg_rewards = np.mean(episode_reward_history[-10:])

#         logging.info('valid finished episode: %f', (batch + 1) * args.batch_size)
#         logging.info('valid average rewards: %f', avg_rewards)
    
#         if avg_rewards >= 500.0:
#             break
#     return episode_reward_history


if __name__ == '__main__':
    main()
