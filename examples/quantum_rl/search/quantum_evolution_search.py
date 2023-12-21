# Note: If you want to search the architecture from scratch, please use the command 'pip install pymoo==0.3.0' first.
import argparse
import logging
import os
import sys

sys.path.insert(0, " ")
import time
from functools import reduce

import cirq
import numpy as np
from misc.utils import create_exp_dir
from pymoo.optimize import minimize
from pymop.problem import Problem
from search import nsganet as engine
from search import quantum_encoding, quantum_train_search

parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for quantum NAS")
parser.add_argument("--save", type=str, default="quantumGA", help="experiment name")
parser.add_argument(
    "--n_var", type=int, default=30, help="the maximum length of architecture"
)
parser.add_argument(
    "--pop_size", type=int, default=10, help="population size of networks"
)
parser.add_argument("--n_gens", type=int, default=10, help="number of generation")
parser.add_argument(
    "--n_offspring",
    type=int,
    default=10,
    help="number of offspring created per generation",
)
parser.add_argument(
    "--n_episodes",
    type=int,
    default=300,
    help="number of episodes to train during architecture search",
)

args = parser.parse_args(args=[])
args.save = "search-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

pop_hist = []  # keep track of every evaluated architecture


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    """Define the multi-objetive problem of quantum architecture search.
    The first aim is to maximize the perfoemance of the task and the second need is control the number of entanglements.
    """

    # first define the NAS problem (inherit from pymop)
    def __init__(
        self,
        qubits,
        n_actions,
        observables,
        n_var=30,
        n_obj=2,
        n_constr=0,
        lb=None,
        ub=None,
        n_episodes=300,
        save_dir=None,
    ):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = lb
        self.xu = ub
        self.n_obj = n_obj
        self.n_episodes = n_episodes
        self._save_dir = save_dir
        self.qubits = qubits
        self.n_actions = n_actions
        self.observables = observables
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self.nb_list = []

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print("\n")
            logging.info("Network id = {}".format(arch_id))

            bit_string = x[i, :]
            nb, _ = quantum_encoding.convert2arch(bit_string)
            if list(nb) not in self.nb_list:
                self.nb_list.append(list(nb))
                performance = quantum_train_search.main(
                    bit_string,
                    qubits=self.qubits,
                    n_actions=self.n_actions,
                    observables=self.observables,
                    n_episodes=self.n_episodes,
                    save="arch_{}".format(arch_id),
                    expr_root=self._save_dir,
                )

                # all objectives assume to be MINIMIZED !!!!!
                objs[i, 0] = -np.mean(performance)
                objs[i, 1] = np.sum(nb == 3)
            else:
                objs[i, 0] = objs[i - 1, 0]
                objs[i, 1] = objs[i - 1, 1]
            self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    """Store statistic information of every generation."""
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # report generation info to files
    logging.info("generation = {}".format(gen))
    logging.info(
        "population collected rewards: best = {}, mean = {}, "
        "median = {}, worst = {}, best_pos = {}".format(
            -np.min(pop_obj[:, 0]),
            -np.mean(pop_obj[:, 0]),
            -np.median(pop_obj[:, 0]),
            -np.max(pop_obj[:, 0]),
            np.where(pop_obj[:, 0] == np.min(pop_obj[:, 0])),
        )
    )
    logging.info(
        "population entangle number: best = {}, mean = {}, "
        "median = {}, worst = {}, best_pos = {}".format(
            np.min(pop_obj[:, 1]),
            np.mean(pop_obj[:, 1]),
            np.median(pop_obj[:, 1]),
            np.max(pop_obj[:, 1]),
            np.where(pop_obj[:, 1] == np.min(pop_obj[:, 1])),
        )
    )


def main(qubits, n_actions, observables):
    """
    Main search process in multi-obj algorithms.
    """
    logging.info("args = %s", args)

    # setup NAS search problem
    lb = np.zeros(args.n_var)
    ub = np.ones(args.n_var) * 3

    problem = NAS(
        qubits,
        n_actions,
        observables,
        lb=lb,
        ub=ub,
        n_var=args.n_var,
        n_episodes=args.n_episodes,
        save_dir=args.save,
    )

    # configure the nsga-net method
    method = engine.nsganet(
        pop_size=args.pop_size, n_offsprings=args.n_offspring, eliminate_duplicates=True
    )

    res = minimize(
        problem,
        method,
        callback=do_every_generations,
        termination=("n_gen", args.n_gens),
    )

    return


if __name__ == "__main__":
    n_qubits = 4  # Dimension of the state vectors in CartPole
    n_actions = 2  # Number of actions in CartPole
    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)]  # Z_0*Z_1*Z_2*Z_3
    main(qubits, n_actions, observables)
