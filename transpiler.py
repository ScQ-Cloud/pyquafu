
import copy
      

class OptSequence(object):
    def __init__(self, num, gatelist):
        self.nodes = []
        self.edges = []
        self.tails = [-1 for i in range(num)]
        self.deg = []
        self.gatelist = gatelist
        self.optlist = []
        self.zerodeg = []
        self.idx = 0

        self.singlequbit = 30
        self.twoqubit = 60
        self.opt = 10000000

    def addnode(self, pos):
        if self.tails[pos] == -1:
            self.zerodeg.append(self.idx)
            self.deg.append(0)
        else:
            self.edges[self.tails[pos]].append(self.idx)
            self.deg.append(1)
        self.edges.append([])
        self.tails[pos] = self.idx

    def addnode2(self, pos1, pos2):
        self.deg.append(0)
        if self.tails[pos1] == -1 and self.tails[pos2] == -1:
            self.zerodeg.append(self.idx)
        else:
            if self.tails[pos1] > -1:
                self.edges[self.tails[pos1]].append(self.idx)
                self.deg[-1] += 1
            if self.tails[pos2] > -1:
                self.edges[self.tails[pos2]].append(self.idx)
                self.deg[-1] += 1
        self.tails[pos1] = self.idx
        self.tails[pos2] = self.idx
        self.edges.append([])

    def add_gate(self, gate):
        self.nodes.append((*gate, self.idx))
        if gate[0] == 1:
            self.addnode(gate[-1])
        elif gate[0] == 2:
            self.addnode2(gate[-2], gate[-1])
        self.idx += 1
  
    def initial(self):
        for gate in self.gatelist:
            self.add_gate(gate)
    
    def run_opt(self, state, cur, curdeg, curlist):
        if not state:
            if cur < self.opt:
                self.opt = cur
                self.optlist = curlist
        else:
            l = len(state)
            zerodegnodes = []
            for i in state:
                zerodegnodes.append(self.nodes[i])
            zerodegnodes.sort()
            rec = 0
            for i in range(l):
                if zerodegnodes[i][0] == 2:
                    break
                else:
                    rec += 1
            for i in range(max(1, 2 ** rec - 1), 2 ** l):
                newstate = copy.copy(state)
                newcur = cur
                newlist = curlist[::]
                newdeg = curdeg[::]
                banset = set([])
                curstep = []
                steptime = self.singlequbit
                binlist = [i >> d & 1 for d in range(l)]
                used = []
                for j in range(l):
                    if binlist[j]:
                        node = zerodegnodes[j]
                        used.append(node[4])
                        if node[0] == 1:
                            pos = node[3]
                            banset.add(pos - 1)
                            banset.add(pos + 1)
                            aff = self.edges[node[4]]
                            for k in aff:
                                newdeg[k] -= 1
                                if newdeg[k] == 0:
                                    newstate.add(k)
                            if type(node[2]) == type([]):
                                newstr = "["
                                templist = []
                                for k in node[2]:
                                    templist.append("%.4f" % k)
                                newstr += ",".join(templist)
                                newstr += "]"
                                curstep.append("[\'%s\', %d, %s]" % (node[1], node[3], newstr))
                            else:
                                if node[1] == 'H':
                                    curstep.append("[\'%s\', %d, %d]" % (node[1], node[3], node[2]))
                                else:
                                    curstep.append("[\'%s\', %d, %.4f]" % (node[1], node[3], node[2]))

                        else:
                            if not (node[2] in banset or node[3] in banset):
                                curstep.append("[\'%s\', [%d, %d]]" % (node[1], node[2], node[3]))
                                banset.add(min(node[2], node[3]) - 1)
                                banset.add(max(node[2], node[3]) + 1)
                                steptime = self.twoqubit
                            else:
                                steptime = 10000000
                            aff = self.edges[node[4]]
                            for k in aff:
                                newdeg[k] -= 1
                                if newdeg[k] == 0:
                                    newstate.add(k)
                newcur += steptime
                newlist.append(curstep)
                newstate -= set(used)
                if cur < self.opt:
                    self.run_opt(newstate, newcur, newdeg, newlist)

