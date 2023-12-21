from collections import namedtuple

import numpy as np

Genotype = namedtuple("Genotype", "measure vpqc dpqc entangle")

# what you want to search should be defined here and in quantum_operations
PRIMITIVES = ["measurement", "variationalPQC", "dataencodingPQC", "entanglement"]


def convert2arch(bit_string):
    # convert bit-string to architecture
    measure = []
    vpqc = []
    dpqc = []
    entangle = []
    new_bit = 0
    for i in range(len(bit_string)):
        if PRIMITIVES[bit_string[i]] == "variationalPQC":
            vpqc.append((PRIMITIVES[bit_string[i]], i))
        elif PRIMITIVES[bit_string[i]] == "dataencodingPQC":
            dpqc.append((PRIMITIVES[bit_string[i]], i))
        elif PRIMITIVES[bit_string[i]] == "entanglement":
            entangle.append((PRIMITIVES[bit_string[i]], i))
        elif PRIMITIVES[bit_string[i]] == "measurement":
            measure.append((PRIMITIVES[bit_string[i]], i))
            if vpqc != [] and dpqc != [] and entangle != []:
                new_bit = bit_string[0 : (i + 1)]
                break
        else:
            raise NameError("Unknown quantum architecture.")
    if type(new_bit) == int:
        new_bit = bit_string
    new_bit = list(new_bit)
    m_num = new_bit.count(0)
    if m_num <= 1:
        g = Genotype(measure=measure, vpqc=vpqc, dpqc=dpqc, entangle=entangle)
    else:
        for i in range(m_num - 1):
            new_bit.remove(0)
        new_bit = np.array(new_bit)
        measure = []
        vpqc = []
        dpqc = []
        entangle = []
        for i in range(len(new_bit)):
            if PRIMITIVES[new_bit[i]] == "variationalPQC":
                vpqc.append((PRIMITIVES[new_bit[i]], i))
            elif PRIMITIVES[new_bit[i]] == "dataencodingPQC":
                dpqc.append((PRIMITIVES[new_bit[i]], i))
            elif PRIMITIVES[new_bit[i]] == "entanglement":
                entangle.append((PRIMITIVES[new_bit[i]], i))
            elif PRIMITIVES[new_bit[i]] == "measurement":
                measure.append((PRIMITIVES[new_bit[i]], i))
            else:
                raise NameError("Unknown quantum architecture.")
        g = Genotype(measure=measure, vpqc=vpqc, dpqc=dpqc, entangle=entangle)

    return new_bit, g
