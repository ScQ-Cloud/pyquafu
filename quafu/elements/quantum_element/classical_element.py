
class Cif:
    name = 'if'
    def __init__(self, cbit, condition, instructions):
        # cbit can be a list of cbit or just a cbit
        self.cbit = cbit
        self.cond = condition
        self.instructions = instructions
