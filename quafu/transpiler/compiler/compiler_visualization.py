import time
from copy import deepcopy
from pprint import pprint
from typing import List, Union

from quafu import QuantumCircuit

# (C) Copyright 2023 Beijing Academy of Quantum Information Sciences
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from quafu.dagcircuits.circuit_dag import dag_to_circuit
from quafu.dagcircuits.dag_circuit import DAGCircuit
from quafu.transpiler.passes.basepass import BasePass
from quafu.transpiler.passes.datadict import DataDict


class CompilerVis:
    def __init__(self, workflow: List[BasePass], initial_model=None):
        self.workflow = workflow
        if initial_model is None:
            self.model = DataDict()
        else:
            self.model = initial_model

    def set_model(self, new_model):
        self.model = new_model

    def compile_vis(self, circuit: Union[QuantumCircuit, DAGCircuit]):
        """
        Return the circuit and the information of each pass.
        The information of each pass is a dict, which contains:
        "Pass Name": pass_name,
        "Execution Time": pass_execution_time,
        "CNOT Before": cnot_count_before,
        "CNOT After": cnot_count_after,
        "Layout Before": layout_before,
        "Layout After": layout_after,
        "Circuit Before": circuit_before,
        "Circuit After": circuit_after
        "Pass Docstring": pass_docstring



        returns:
            circuit: the compiled circuit
            info: the information of each pass
            short_info: the short information of each pass. This is used to check the statistics of all passes
        """
        info = {}

        total_execution_time = 0.00  # ms
        all_layouts = {"initial_layout": deepcopy(self.model.get_layout()['initial_layout']),
                       "final_layout": None}

        for idx, pass_instance in enumerate(self.workflow):
            pass_name = pass_instance.__class__.__name__
            # print(f"Pass {idx}: {pass_name}")

            # get the pass docstring
            pass_docstring = pass_instance.__doc__

            # get the circuit, layout and others before the pass
            circuit_before = deepcopy(circuit)
            if hasattr(pass_instance, 'set_model'):
                pass_instance.set_model(self.model)
            layout_before = deepcopy(self.model.get_layout()['final_layout'])  # get the pass initial layout
            # cnot_count_before = circuit_before.count_ops().get("cx", 0)
            gates_count_before = len(circuit_before.gates)
            circuit_depth_before = len(circuit_before.layered_circuit().T) - 1
            mutil_qubit_gates_before = 0
            for g in circuit_before.gates:
                if isinstance(g.pos, list):
                    if len(g.pos) == 2:
                        mutil_qubit_gates_before += 1

            # timing the start time <--- start
            pass_start_time = time.perf_counter()  # us
            circuit = pass_instance.run(circuit)
            pass_end_time = time.perf_counter()  # us
            # timing the end time <--- end

            # # Determine whether layout is changed in pass
            # if layout_before != deepcopy(self.model.get_layout()['initial_layout']):
            #     layout_before = deepcopy(self.model.get_layout()['initial_layout'])

            pass_execution_time = (pass_end_time - pass_start_time) * 1000  # ms
            pass_execution_time = round(pass_execution_time, 2)  # round to 2 decimal places

            total_execution_time += pass_execution_time  # ms , add the execution time of each pass
            total_execution_time = round(total_execution_time, 2)  # round to 2 decimal places

            # Get the circuit , layout and others  after the pass
            circuit_after = deepcopy(circuit)
            if hasattr(pass_instance, 'get_model'):
                self.model = pass_instance.get_model()

            # Get the final layout of the pass, and we can compare layout_before and layout_after
            layout_after = deepcopy(self.model.get_layout()['final_layout'])
            circuit_depth_after = len(circuit_after.layered_circuit().T) - 1
            gates_count_after = len(circuit_after.gates)
            mutil_qubit_gates_after = 0
            for g in circuit_after.gates:
                if isinstance(g.pos, list):
                    if len(g.pos) == 2:
                        mutil_qubit_gates_after += 1

            # layout changed status
            if layout_before == layout_after:
                layout_changed_status = False
            else:
                if layout_before is None: # layout_after is not None
                    layout_changed_status = True
                elif layout_after is None: # layout_before is not None
                    layout_changed_status = True
                else:
                    dict_layout_before = layout_before.v2p
                    dict_layout_after = layout_after.v2p
                    layout_changed_status = (dict_layout_before != dict_layout_after)

            # circuit_before, transpiled_qasm = qiskit2quafu(circuit_before)
            # print(circuit_before.draw_circuit())
            # circuit_after, transpiled_qasm = qiskit2quafu(circuit_after)
            # circuit_after.draw_circuit()

            info[f"Pass_{idx}"] = {
                "Pass Name": pass_name,
                "Execution Time (ms)": pass_execution_time,
                "2qubit Gates Before": mutil_qubit_gates_before,
                "2qubit Gates After": mutil_qubit_gates_after,
                "Total Gates Before": gates_count_before,
                "Total Gates After": gates_count_after,
                "Depth Before": circuit_depth_before,
                "Depth After": circuit_depth_after,
                "Layout Before": layout_before,
                "Layout After": layout_after,
                "Layout changed status": layout_changed_status,
                "Circuit Before": circuit_before,
                "Circuit After": circuit_after,
                "Pass Docstring": pass_docstring
            }

        if isinstance(circuit, DAGCircuit):
            circuit = dag_to_circuit(circuit, circuit.circuit_qubits)

        info["Total"] = {
            "Total Execution Time (ms)": total_execution_time,
            "model": deepcopy(self.model)
        }

        # change the final layout of info's model
        all_layouts["initial_layout"] = deepcopy(info["Total"]["model"].get_layout()['initial_layout'])
        all_layouts["final_layout"] = deepcopy(info["Total"]["model"].get_layout()['final_layout'])
        info["Total"]["model"].set_layout(all_layouts)

        # make a copy of short info
        short_info = deepcopy(info)
        for key, value in short_info.items():
            if key == "Total":
                continue
            short_info[key].pop("Circuit Before", None)
            short_info[key].pop("Circuit After", None)
            short_info[key].pop("Pass Docstring", None)
            short_info[key].pop("Layout Before", None)
            short_info[key].pop("Layout After", None)
        short_info["Total"].pop("model", None)

        return circuit, info, short_info


def draw_allpass_circuits(info_dict, only_original_and_last=False):
    """Draw the circuit before and after each pass."""
    original_circuit = info_dict["Pass_0"]["Circuit Before"]

    if not only_original_and_last:
        for key, pass_info in info_dict.items():
            if key == "Total":
                break
            # draw the circuit before and after the pass
            # print the divider line
            print("*" * 100)
            print(f"Drawing the circuit before and after {key}:", pass_info["Pass Name"])
            print("Circuit Before:")
            pass_info["Circuit Before"].draw_circuit()
            print("Circuit After:")
            pass_info["Circuit After"].draw_circuit()
            print()
    last_circuit = info_dict[f"Pass_{len(info_dict) - 2}"]["Circuit After"]
    print("*" * 100)
    print("Drawing the original circuit and the last circuit:")
    print("Original Circuit:")
    original_circuit.draw_circuit()
    print("Last Circuit:")
    last_circuit.draw_circuit()
    print("*" * 100)


def draw_pass_info(info_dict, pass_idx):
    """Draw the info about the input index pass."""
    print("*" * 100)
    print(f"The information of Pass_{pass_idx}:")
    for key, value in info_dict[f"Pass_{pass_idx}"].items():
        if key == "Pass Docstring":
            print("*" * 100)
            print("Pass Usage", ":")
            print(value)
            print("*" * 100)
            continue
        if key == "Circuit Before":
            print(key, ":")
            value.draw_circuit()
            continue
        if key == "Circuit After":
            print(key, ":")
            value.draw_circuit()
            continue
        if key == "Layout Before":
            print(key, ":")
            if value is None:
                print('None')
                continue
            print('layout (v2p):', value.v2p)
            continue
        if key == "Layout After":
            print(key, ":")
            if value is None:
                print('None')
                continue
            print('layout (v2p):', value.v2p)
            continue
        if key == "Layout changed status":
            print(key, ":", end=" ")
            if value:
                print("Changed")
            else:
                print("Not Changed")
            continue

        print(key, ":", value)


    # print(f"Drawing the circuit before and after Pass_{pass_idx}:",end=" ")
    # print(info_dict[f"Pass_{pass_idx}"]["Pass Name"])
    # print("Circuit Before:")
    # info_dict[f"Pass_{pass_idx}"]["Circuit Before"].draw_circuit()
    # print("Circuit After:")
    # info_dict[f"Pass_{pass_idx}"]["Circuit After"].draw_circuit()
    # print("*" * 100)


def dynamic_draw(info_dict, short_info):
    """ Draw the circuit before and after the input index pass. here the input index is dynamic
    shot_info: to check the statistics of all passes
    info_dict: to check the model and the detail information of each pass
    """
    while True:
        print("Please input 'm' to check the model,\n"
              "or input 'q' to quit compilation information visualization,\n"
              "or input 'c' to check the statistics of all passes,\n"
              "or input the int (0-" + str(len(short_info) - 2) + ") index of the pass you want to draw:")

        input_str = input()
        if input_str == 'q':
            break
        elif input_str == 'c':
            # pprint(short_info, sort_dicts=False)
            from rich.table import Table
            from rich.console import Console

            print("*" * 100)
            print("quafu transpile passes information:")
            # console = Console()
            console = Console(width=200)
            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.width = 150
            table.add_column("Pass")
            table.add_column("Pass Name", width=25)
            table.add_column("Execution Time (ms)")
            table.add_column("Total Gates Before")
            table.add_column("Total Gates After")
            table.add_column("2qubit Gates Before")
            table.add_column("2qubit Gates After")
            table.add_column("Depth Before")
            table.add_column("Depth After")
            table.add_column("Layout changed status")

            for key, value in short_info.items():
                if key == "Total":
                    continue
                # add layout changed status
                table.add_row(key, value["Pass Name"], str(value["Execution Time (ms)"]),
                              str(value["Total Gates Before"]), str(value["Total Gates After"]),
                              str(value["2qubit Gates Before"]), str(value["2qubit Gates After"]),
                              str(value["Depth Before"]), str(value["Depth After"]),str(value["Layout changed status"]), end_section=True)
                # table.add_row(key, value["Pass Name"], str(value["Execution Time (ms)"]),
                #               str(value["Total Gates Before"]), str(value["Total Gates After"]),
                #               str(value["2qubit Gates Before"]), str(value["2qubit Gates After"]),
                #               str(value["Depth Before"]), str(value["Depth After"]), end_section=True)

            console.print(table)

            print("Total Execution Time (ms):", short_info["Total"]["Total Execution Time (ms)"])

            print("*" * 100)
            continue
        elif input_str == "m":  # print the model
            model = info_dict["Total"]["model"]
            print("*" * 100)
            print("The model:", model)
            print("The backend:")
            pprint(model.get_backend().get_all_properties())
            print("The layout:")
            # pprint(model.get_layout())
            layout = model.get_layout()
            if layout['initial_layout'] is None:
                print('initial_layout: None')
            else:
                print('initial_layout (v2p):', layout['initial_layout'].v2p)

            if layout['final_layout'] is None:
                print('final_layout: None')
            else:
                print('final_layout (v2p):', layout['final_layout'].v2p)
            print("*" * 100)
            continue
        else:
            try:
                input_idx = int(input_str)
                draw_pass_info(info_dict, input_idx)
                print("*" * 100)
                continue
            except:
                print("wrong input, please input again")
                continue


def dynamic_draw_tabulate(info_dict, short_info, tablefmt='fancy_grid'):
    """ Draw the circuit before and after the input index pass. here the input index is dynamic"""
    # tablefmt = ["plain", "simple", "github", "grid", "fancy_grid", "pipe", "orgtbl", "jira",
    #             "presto", "psql", "rst", "mediawiki", "moinmoin", "youtrack", "html", "latex",
    #             "latex_raw", "latex_booktabs", "textile"]

    while True:
        print("Please input 'm' to check the model,\n"
              "or input 'q' to quit compilation information visualization,\n"
              "or input 'c' to check the statistics of all passes,\n"
              "or input the int (0-" + str(len(short_info) - 2) + ") index of the pass you want to draw:")

        input_str = input()
        if input_str == 'q':
            break
        elif input_str == 'c':
            from tabulate import tabulate
            from rich.table import Table
            from rich.console import Console

            print("*" * 100)
            print("quafu transpile passes information:")

            table_header = ['Pass', 'Pass Name', 'Execution Time (ms)', 'Total Gates Before',
                            'Total Gates After', '2qubit Gates Before',
                            '2qubit Gates After', 'Depth Before', 'Depth After','Layout changed status']
            # table_header = ['Pass', 'Pass Name', 'Execution Time (ms)', 'Total Gates Before',
            #                 'Total Gates After', '2qubit Gates Before',
            #                 '2qubit Gates After', 'Depth Before', 'Depth After']

            table_data = []
            for key, value in short_info.items():
                if key == "Total":
                    continue
                # add layout changed status
                table_data.append((key, value["Pass Name"], str(value["Execution Time (ms)"]),
                                   str(value["Total Gates Before"]), str(value["Total Gates After"]),
                                   str(value["2qubit Gates Before"]), str(value["2qubit Gates After"]),
                                   str(value["Depth Before"]), str(value["Depth After"]),str(value["Layout changed status"])))
                # table_data.append([key, value["Pass Name"], str(value["Execution Time (ms)"]),
                #                       str(value["Total Gates Before"]), str(value["Total Gates After"]),
                #                       str(value["2qubit Gates Before"]), str(value["2qubit Gates After"]),
                #                       str(value["Depth Before"]), str(value["Depth After"])])

            print(tabulate(table_data, headers=table_header, tablefmt=tablefmt))

            print("Total Execution Time (ms):", short_info["Total"]["Total Execution Time (ms)"])

            print("*" * 100)
            continue
        elif input_str == "m":  # print the model
            model = info_dict["Total"]["model"]
            print("*" * 100)
            print("The model:", model)
            print("The backend:")
            pprint(model.get_backend().get_all_properties())
            print("The layout:")
            # pprint(model.get_layout())
            layout = model.get_layout()
            if layout['initial_layout'] is None:
                print('initial_layout: None')
            else:
                print('initial_layout (v2p):', layout['initial_layout'].v2p)

            if layout['final_layout'] is None:
                print('final_layout: None')
            else:
                print('final_layout (v2p):', layout['final_layout'].v2p)
            print("*" * 100)
            continue
        else:
            try:
                input_idx = int(input_str)
                draw_pass_info(info_dict, input_idx)
                print("*" * 100)
                continue
            except:
                print("wrong input, please input again")
                continue
