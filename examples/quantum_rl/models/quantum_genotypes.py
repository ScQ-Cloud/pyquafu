from collections import namedtuple

Genotype = namedtuple('Genotype', 'measure vpqc dpqc entangle')

PRIMITIVES = [
    'measurement',
    'variationalPQC',
    'dataencodingPQC',
    'entanglement'
]

NSGANet_id10 = Genotype(
    measure=[
        ('measurement', 12)
    ], 
    vpqc=[
        ('variationalPQC', 1), 
        ('variationalPQC', 2), 
        ('variationalPQC', 4), 
        ('variationalPQC', 5), 
        ('variationalPQC', 7), 
        ('variationalPQC', 10)
    ], 
    dpqc=[
        ('dataencodingPQC', 3), 
        ('dataencodingPQC', 9), 
        ('dataencodingPQC', 11)
    ], 
    entangle=[
        ('entanglement', 0), 
        ('entanglement', 6), 
        ('entanglement', 8)
    ]
)

NSGANet_id21 = Genotype(
    measure=[
        ('measurement', 16)
    ], 
    vpqc=[
        ('variationalPQC', 1), 
        ('variationalPQC', 2), 
        ('variationalPQC', 5), 
        ('variationalPQC', 6), 
        ('variationalPQC', 11)
    ], 
    dpqc=[
        ('dataencodingPQC', 0), 
        ('dataencodingPQC', 3), 
        ('dataencodingPQC', 4), 
        ('dataencodingPQC', 7), 
        ('dataencodingPQC', 9), 
        ('dataencodingPQC', 10), 
        ('dataencodingPQC', 13)
    ], 
    entangle=[
        ('entanglement', 8), 
        ('entanglement', 12), 
        ('entanglement', 14), 
        ('entanglement', 15)
    ]
)

NSGANet_id97 = Genotype(
    measure=[
        ('measurement', 5)
    ], 
    vpqc=[
        ('variationalPQC', 1)
    ], 
    dpqc=[
        ('dataencodingPQC', 0), 
        ('dataencodingPQC', 3), 
        ('dataencodingPQC', 4)
    ], 
    entangle=[
        ('entanglement', 2)
    ]
)