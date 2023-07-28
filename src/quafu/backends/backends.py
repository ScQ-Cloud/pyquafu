import requests
import json
import re
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from quafu.users.userapi import User


class Backend(object):
    def __init__(self, backend_info: dict):
        self.name = backend_info['system_name']
        self._valid_gates = backend_info['valid_gates']
        self.qubit_num = backend_info['qubits']
        self.system_id = backend_info['system_id']
        self.status = backend_info['status']
        self.qv = backend_info["QV"]
        # self.task_in_queue = backend_info["task_in_queue"]

    def get_chip_info(self, user=User()):
        # update api-token, a patch to be deleted in the future
        api_token = user._load_account_token()
        data = {"system_name": self.name.lower()}
        headers = {"api_token": api_token}
        chip_info = requests.post(url=User.chip_api, data=data,
                                  headers=headers)
        chip_info = json.loads(chip_info.text)
        json_topo_struct = chip_info["topological_structure"]
        qubits_list = []
        for gate in json_topo_struct.keys():
            qubit = gate.split('_')
            qubits_list.append(qubit[0])
            qubits_list.append(qubit[1])
        qubits_list = list(set(qubits_list))
        qubits_list = sorted(qubits_list, key=lambda x: int(re.findall(r"\d+", x)[0]))
        int_to_qubit = {k: v for k, v in enumerate(qubits_list)}
        qubit_to_int = {v: k for k, v in enumerate(qubits_list)}

        directed_weighted_edges = []
        weighted_edges = []
        edges_dict = {}
        clist = []
        for gate, name_fidelity in json_topo_struct.items():
            gate_qubit = gate.split('_')
            qubit1 = qubit_to_int[gate_qubit[0]]
            qubit2 = qubit_to_int[gate_qubit[1]]
            gate_name = list(name_fidelity.keys())[0]
            fidelity = name_fidelity[gate_name]['fidelity']
            directed_weighted_edges.append([qubit1, qubit2, fidelity])
            clist.append([qubit1, qubit2])
            gate_reverse = gate.split('_')[1] + '_' + gate.split('_')[0]
            if gate not in edges_dict and gate_reverse not in edges_dict:
                edges_dict[gate] = fidelity
            else:
                if fidelity < edges_dict[gate_reverse]:
                    edges_dict.pop(gate_reverse)
                    edges_dict[gate] = fidelity

        for gate, fidelity in edges_dict.items():
            gate_qubit = gate.split('_')
            qubit1, qubit2 = qubit_to_int[gate_qubit[0]], qubit_to_int[gate_qubit[1]]
            weighted_edges.append([qubit1, qubit2, np.round(fidelity, 3)])

        # draw topology
        G = nx.Graph()
        for key, value in int_to_qubit.items():
            G.add_node(key, name=value)

        G.add_weighted_edges_from(weighted_edges)

        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.9]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.9]
        elarge_labels = {(u, v): "%.3f" % d["weight"] for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.9}
        esmall_labels = {(u, v): "%.3f" % d["weight"] for (u, v, d) in G.edges(data=True) if d["weight"] < 0.9}

        pos = nx.spring_layout(G, seed=1)
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(G, pos, node_size=400, ax=ax)

        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=2, ax=ax)
        nx.draw_networkx_edges(
            G, pos, edgelist=esmall, width=2, alpha=0.5, style="dashed"
            , ax=ax)

        nx.draw_networkx_labels(G, pos, font_size=14, font_family="sans-serif", ax=ax)
        # edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, elarge_labels, font_size=12, font_color="green", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, esmall_labels, font_size=12, font_color="red", ax=ax)
        fig.set_figwidth(14)
        fig.set_figheight(14)
        fig.tight_layout()
        return {"mapping": int_to_qubit, "topology_diagram": fig, "full_info": chip_info}

    def get_valid_gates(self):
        return self._valid_gates
