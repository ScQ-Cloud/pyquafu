from typing import Dict, Any, List, Union
import dataclasses 

@dataclasses.dataclass
class InstructionNode: 
    name:Any            # gate.name
    pos:Union[List[Any], Dict[Any,Any]]       # gate.pos |  Dict[Any,Any] for measure
    paras:List[Any]        # gate.paras
    # matrix:List[Any]   # for gate in [QuantumGate]
    duration: Union[float,int]  # for gate in [Delay,XYResonance,QuantumPulse] in quafu
    unit:str           # for gate in [Delay,XYResonance] in quafu
    channel:str        # for gate in [QuantumPulse] in quafu
    time_func: Any     # for gate in [QuantumPulse] in quafu
    label:str          # used for specifying the instruction node
        
    def __hash__(self):
        return hash((type(self.name), tuple(self.pos) ,self.label))
    
    def __str__(self):
        if self.name == 'measure':
            args = ','.join(str(q) for q in self.pos.keys())
            args += f'=>{",".join(str(c) for c in self.pos.values())}'
        else:    
            args = ','.join(str(q) for q in self.pos)
            
        if self.paras == None:
            return f'{self.label}{{{self.name}({args})}}' # no paras
        else:
            # if self.paras not a list, then make it a list  of str of .3f float
            if not isinstance(self.paras, list):
                formatted_paras = [f'{self.paras:.3f}']
            else:
                formatted_paras = [f'{p:.3f}' for p in self.paras]  
                
            formatted_paras_str = ','.join(formatted_paras)
            
            return f'{self.label}{{{self.name}({args})}}({formatted_paras_str})'
    
    def __repr__(self): 
        return str(self)