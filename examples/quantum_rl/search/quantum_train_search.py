import logging
import os
import sys

import numpy as np
import tensorflow as tf
from misc.utils import compute_returns, create_exp_dir, gather_episodes
from models.quantum_models import generate_model_policy as Network
from search import quantum_encoding


def main(
    bit_string,
    qubits,
    n_actions,
    observables,
    n_episodes=1000,
    batch_size=10,
    gamma=1,
    beta=1.0,
    state_bounds=np.array([2.4, 2.5, 0.21, 2.5]),
    env_name="CartPole-v1",
    save="quantum",
    expr_root="search",
    lr_in=0.1,
    lr_var=0.01,
    lr_out=0.1,
    backend="cirq",
):
    """
    Main training process in multi-objective search.
    """
    save_pth = os.path.join(expr_root, "{}".format(save))
    create_exp_dir(save_pth)
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(save_pth, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    nb, genotype = quantum_encoding.convert2arch(bit_string)
    model = Network(qubits, genotype, n_actions, beta, observables, env_name)

    logging.info("Genome = %s", nb)
    logging.info("Architecture = %s", genotype)

    optimizer_in = tf.keras.optimizers.Adam(learning_rate=lr_in, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=lr_var, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=lr_out, amsgrad=True)

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
        for optimizer, w in zip(
            [optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]
        ):
            optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

    # Start training the agent
    episode_reward_history = []
    for batch in range(n_episodes // batch_size):
        # Gather episodes
        _, episodes = gather_episodes(
            state_bounds, n_actions, model, batch_size, env_name, beta, backend
        )

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep["states"] for ep in episodes])
        actions = np.concatenate([ep["actions"] for ep in episodes])
        rewards = [ep["rewards"] for ep in episodes]
        returns = np.concatenate(
            [compute_returns(ep_rwds, gamma) for ep_rwds in rewards]
        )
        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        reinforce_update(states, id_action_pairs, returns, model)

        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))

        avg_rewards = np.mean(episode_reward_history[-10:])

        logging.info("Finished episode: %f", (batch + 1) * batch_size)
        logging.info("Average rewards: %f", avg_rewards)

        if avg_rewards >= 500.0 and env_name == "CartPole-v1":
            break
        elif avg_rewards >= -110 and env_name == "MountainCar-v0":
            break
    return episode_reward_history
