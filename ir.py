from pprint import pprint
from typing import Optional, Any, List, Set

import numpy as np

DEBUG = True


class Statement:
    """
    Statement occurring in MIR with concomitant data relevant for building CFG+BB IR
    """

    def __init__(self, lhs_location=None, rhs_location=None, rhs_value=None, rhs_type=None, op=None, bb_id=None):
        self.lhs_location: int = None
        self.rhs_location: Optional[int] = None
        self.rhs_value: Optional[Any] = None
        self.rhs_type: Optional[str] = None
        self.rhs_op: Optional[str] = None
        self.mutability: Optional[bool] = None


class FlagMatrix:
    """
    FlagMatrix describing the borrowing status of locations
    Transposed wrt. the paper to make it easier to work with
    """

    # symbols
    symbols = {'i': 2, 'm': 1, '0': 0}

    def __init__(self, max_size: int):
        self.matrix = np.array([[self.symbols['0']] * max_size] * max_size)

    # define whether matrix represent legal borrowing
    def legal(self) -> bool:
        # check matrix for each column, denoting a locations
        for c_i, c in enumerate(self.matrix):
            if DEBUG:
                print(f"column {c_i}: {c}")
            high_status = self.symbols['0']
            for s_i, s in enumerate(c):
                if DEBUG:
                    print(f"symbol {s_i}: {s}")
                high_status = max(high_status, s)
                if DEBUG:
                    print(f"max: {high_status}")
                if s < high_status:
                    return False
                # update
                last = s
        return True

    # pretty print matrix substituting self.symbols values with their keys
    def pprint(self):
        for c in self.matrix:
            print([list(self.symbols.keys())[list(self.symbols.values()).index(i)] for i in c])


class BasicBlock:
    def __init__(self, id: int):
        self.name: int = id
        self.stmts: List[Statement] = []
        self.succ: Set[BasicBlock] = []
        self.pred: Set[BasicBlock] = []
        self.livein = []
        self.liveout = []
        self.defs = []
        self.uses = []
        self.gen = []

        # check if id is positive and reject if not
        if id < 0:
            raise ValueError(f"Invalid BB id: {id}")

    def add_def(self, loc):
        self.defs.append(loc)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return not self.__lt__(other) and not self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)


class CFG:
    """
    Control flow graph data structure using BasicBlock as nodes, and edges by pred and succ on BasicBlock/linkedlist
    """

    # list of BB nodes
    bb = []
    # list of locations in the CFG
    locations = []
    # list of types of locations in the CFG k:v -> location:type
    types = {}

    def __init__(self):
        pass

    def index_of(self, node: BasicBlock):
        return self.bb.index(node)

    def add_bb(self, node: BasicBlock):
        self.bb.append(node)

    def add_edge(self, fro: BasicBlock, to: BasicBlock):
        # set succ of fro to to
        self.bb[self.index_of(fro)].succ.append(self.index_of(to))
        # set pred of to to fro
        self.bb[self.index_of(to)].pred.append(self.index_of(fro))

    def pprint(self):
        for n in self.bb:
            print(f"BB {n.name}:")
            print(f"  succ: {n.succ}")
            print(f"  pred: {n.pred}")


class CFGUDChain(CFG):
    """
    CFG data structure with Use-Define chain computation
    """

    # list of UD chains
    ud_chains = []

    def __init__(self):
        super().__init__()

    def compute_ud_chains(self):
        # compute UD chains
        for n in self.bb:
            # compute defs
            n.defs = [i for i in n.instr if i[0] == 'def']
            # compute uses
            n.uses = [i for i in n.instr if i[0] == 'use']
            # compute gen
            n.gen = [i for i in n.instr if i[0] == 'gen']
            # compute livein
            n.livein = [i for i in n.instr if i[0] == 'livein']
            # compute liveout
            n.liveout = [i for i in n.instr if i[0] == 'liveout']

        # compute UD chains
        for n in self.bb:
            for u in n.uses:
                # find def
                for d in n.defs:
                    if u[1] == d[1]:
                        self.ud_chains.append((u, d))
                        break
                # find gen
                for g in n.gen:
                    if u[1] == g[1]:
                        self.ud_chains.append((u, g))
                        break
                # find livein
                for l in n.livein:
                    if u[1] == l[1]:
                        self.ud_chains.append((u, l))
                        break
                # find liveout
                for l in n.liveout:
                    if u[1] == l[1]:
                        self.ud_chains.append((u, l))
                        break

    def pprint(self):
        super().pprint()
        for c in self.ud_chains:
            print(f"UD chain: {c}")


if __name__ == '__main__':
    a = FlagMatrix(2)
    b = FlagMatrix(2)
    a.matrix[0][0] = FlagMatrix.symbols['i']
    a.matrix[0][1] = FlagMatrix.symbols['m']
    a.matrix[1][0] = FlagMatrix.symbols['m']
    a.matrix[1][1] = FlagMatrix.symbols['i']
    # a.matrix[0][1] = FlagMatrix.symbols['m']
    print(a.legal())
    print(b.legal())
    pprint(a.matrix)
    a.pprint()

    exit()

    pprint(a.matrix)
    pprint(b.matrix)
    print()
    print(a.matrix[0, :])
