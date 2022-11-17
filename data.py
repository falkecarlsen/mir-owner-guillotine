from pprint import pprint

import numpy as np
import sympy as sp

DEBUG = True


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
        self.instr = []
        self.succ = []
        self.pred = []
        self.livein = []
        self.liveout = []
        self.defs = []
        self.uses = []
        self.gen = []

        # check if id is positive and reject if not
        if id < 0:
            raise ValueError(f"Invalid BB id: {id}")

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

    def __init__(self):
        # list of BB nodes
        self.nodes = []
        # list of locations in the CFG
        self.locations = []
        # list of types of locations in the CFG k:v -> location:type
        self.types = {}

    def index_of(self, node: BasicBlock):
        return self.nodes.index(node)

    def add_node(self, node: BasicBlock):
        self.nodes.append(node)

    def add_edge(self, fro: BasicBlock, to: BasicBlock):
        # set succ of fro to to
        self.nodes[self.index_of(fro)].succ.append(self.index_of(to))
        # set pred of to to fro
        self.nodes[self.index_of(to)].pred.append(self.index_of(fro))

    def pprint(self):
        for n in self.nodes:
            print(f"BB {n.name}:")
            print(f"  succ: {n.succ}")
            print(f"  pred: {n.pred}")


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
