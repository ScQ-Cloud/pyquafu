import logging
import os
import sys
import time
from collections import defaultdict
from functools import reduce

import cirq
import gym
import numpy as np
import tensorflow as tf
from misc import utils
from models.quantum_models import generate_circuit
from models.quantum_models import generate_model_policy as Network
from models.quantum_models import get_model_circuit_params
from search import quantum_encoding
from visualization.qrl import get_obs_policy, get_quafu_exp


def gather_episodes(state_bounds, n_actions, model, n_episodes, env_name, qubits=None, genotype=None):
    """Interact with environment in batched fashion."""

    trajectories = [defaultdict(list) for _ in range(n_episodes)]
    envs = [gym.make(env_name) for _ in range(n_episodes)]

    done = [False for _ in range(n_episodes)]
    states = [e.reset() for e in envs]

    tasklist = []

    while not all(done):
        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            trajectories[i]['states'].append(state)

        # Compute policy for all unfinished envs in parallel
        states = tf.convert_to_tensor(normalized_states)

        # You can choose action probabilities trained by Cirq simulator or Quafu cloud
        # action_probs = model([states])
        newtheta, newlamda = get_model_circuit_params(qubits, genotype, model)
        circuit, _, _ = generate_circuit(qubits, genotype, newtheta, newlamda, states.numpy()[0])
        taskid, expectation = get_quafu_exp(circuit)
        tasklist.append(taskid)
        # print('gather_episodes_exp:', expectation)

        obsw = model.get_layer('observables-policy').get_weights()[0]
        obspolicy = get_obs_policy(obsw)
        action_probs = obspolicy(expectation)
        # print('gather_episodes_policy:', action_probs)

        # Store action and transition all environments to the next state
        states = [None for i in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            action = np.random.choice(n_actions, p=policy)
            states[i], reward, done[i], _ = envs[i].step(action)
            trajectories[i]['actions'].append(action)
            trajectories[i]['rewards'].append(reward)

    return tasklist, trajectories


def compute_returns(rewards_history, gamma):
    """Compute discounted returns with discount factor `gamma`."""
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


def main(bit_string, qubits, n_actions, observables, n_episodes = 1000, batch_size = 10, gamma = 1,
         state_bounds = np.array([2.4, 2.5, 0.21, 2.5]), env_name = "CartPole-v1", save='quantum', expr_root='search'):
    """Main training process"""
    save_pth = os.path.join(expr_root, '{}'.format(save))
    utils.create_exp_dir(save_pth)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_pth, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    nb, genotype = quantum_encoding.convert2arch(bit_string)
    model = Network(qubits, genotype, n_actions, observables)
    
    logging.info("Genome = %s", nb)
    logging.info("Architecture = %s", genotype)

    optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

    # Assign the model parameters to each optimizer
    w_in, w_var, w_out = 1, 0, 2

    @tf.function
    def reinforce_update(states, actions, returns, model):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            logits = model(states)
            p_actions = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_actions)
            loss = tf.math.reduce_sum(-log_probs * returns) / batch_size
        grads = tape.gradient(loss, model.trainable_variables)
        for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
            optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

    # Start training the agent
    episode_reward_history = []
    for batch in range(n_episodes // batch_size):
        # Gather episodes
        episodes = gather_episodes(state_bounds, n_actions, model, batch_size, env_name)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([compute_returns(ep_rwds, gamma) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        reinforce_update(states, id_action_pairs, returns, model)

        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))

        avg_rewards = np.mean(episode_reward_history[-10:])

        logging.info('Finished episode: %f', (batch + 1) * batch_size)
        logging.info('Average rewards: %f', avg_rewards)
    
        if avg_rewards >= 500.0:
            break
    return episode_reward_history
