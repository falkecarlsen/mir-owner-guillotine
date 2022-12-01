from dataclasses import dataclass, field
from enum import Enum, auto
from pprint import pprint
from typing import Optional, Any, List, Set, Dict, Tuple

import numpy as np

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


class StatementType(Enum):
    """
    Statement type
    """

    ASSIGN = auto()
    FUNCTION_CALL = auto()
    PRIMITIVE = auto()
    UNREACHABLE = auto()
    RETURN = auto()
    GOTO = auto()


class ValueType(Enum):
    """
    Value type
    """

    CONST = auto()
    LOCATION = auto()
    BORROW = auto()
    DEREF = auto()
    CALL = auto()


@dataclass(kw_only=True)
class Statement:
    """
    Statement occurring in MIR with concomitant data relevant for building CFG+BB IR
    """

    stmt_type: Optional[StatementType] = None
    lhs_location: int = None
    mutability: Optional[bool] = False
    value_type: Optional[ValueType] = None
    rhs_location: Optional[int] = None
    rhs_op: Optional[str] = None
    rhs_value: Optional[Any] = None

    def __repr__(self):
        return (
            f"\tStatement(\n"
            f"\t\tstmt_type={self.stmt_type},\n"
            f"\t\tlhs_location={self.lhs_location}, \n"
            f"\t\trhs_location={self.rhs_location}, \n"
            f"\t\trhs_value={self.rhs_value}, \n"
            f"\t\trhs_type={self.value_type}, \n"
            f"\t\trhs_op={self.rhs_op}, \n"
            f"\t\tmutability={self.mutability}\n"
        )


class Mode(Enum):
    """
    Mode of ownership transfer (move or copy)
    """

    NONE = auto()
    MOVE = auto()
    COPY = auto()


@dataclass(kw_only=True)
class FunctionArg:
    """
    Function argument, mode (move, etc), location, constant, type of
    """

    mode: Mode = Mode.NONE
    location: [int | None] = None
    # constant (value, type)
    constant: Optional[Tuple[Any, str]] = None
    type: Optional[str] = None

    def __repr__(self):
        return (
            f"\tFunctionArg(\n"
            f"\t\tmode={self.mode},\n"
            f"\t\tlocation={self.location}, \n"
            f"\t\tconstant={self.constant}, \n"
            f"\t\ttype={self.type}\n"
        )


@dataclass(kw_only=True)
class FunctionStatement(Statement):
    """
    Function statement occurring in MIR with concomitant data relevant for building CFG+BB IR
    """

    function_method: Optional[str] = None
    function_type: Optional[str] = None
    function_args: Optional[List[FunctionArg]] = field(default_factory=list)
    function_bb_goto: Optional[int] = None

    def __repr__(self):
        return (
            f"\tFunctionStatement(\n"
            f"\t\tfunction_method={self.function_method},\n"
            f"\t\tfunction_type={self.function_type}, \n"
            f"\t\tfunction_args={self.function_args}, \n"
            f"\t\tfunction_bb_goto={self.function_bb_goto}, \n"
        )


@dataclass(kw_only=True)
class PrimitiveFunctionStatement(Statement):
    """
    Primitive statement occurring in MIR with concomitant data relevant for building CFG+BB IR
    """

    primitive_type: str = None
    primitive_args: Optional[List[FunctionArg]] = field(default_factory=list)
    primitive_bb_goto: Optional[List[int]] = None

    # repr
    def __repr__(self):
        return (
            f"\tPrimitiveFunctionStatement(\n"
            f"\t\tprimitive_type={self.primitive_type},\n"
            f"\t\tprimitive_args=\n{self.primitive_args}, \n"
            f"\t\tprimitive_bb_goto={self.primitive_bb_goto}\n"
        )


@dataclass
class BasicBlock:
    name: int
    stmts: List[Statement] = field(default_factory=list)
    succ: Set['BasicBlock'] = field(default_factory=set)
    pred: Set['BasicBlock'] = field(default_factory=set)
    livein: Set = field(default_factory=set)
    liveout: Set = field(default_factory=set)
    defs: List = field(default_factory=list)
    uses: List = field(default_factory=list)
    gen: List = field(default_factory=list)
    kill: List = field(default_factory=list)

    def __repr__(self):
        return (
            f"\tBasicBlock(id={self.name}, \n"
            f"\tstmts=\n{self.stmts}, \n"
            f"\tsucc={self.succ}, \n"
            f"\tpred={self.pred}, \n"
            f"\tlivein={self.livein}, \n"
            f"\tliveout={self.liveout}, \n"
            f"\tdefs={self.defs}, \n"
            f"\tuses={self.uses}, \n"
            f"\tgen={self.gen}, \n"
            f"\tkill={self.kill})\n"
        )

    def add_statements(self, stmts: List[Statement]):
        self.stmts.extend(stmts)


class CFG:
    """
    Control flow graph data structure using BasicBlock as nodes, and edges by pred and succ on BasicBlock/linkedlist
    """

    # list of BB nodes
    bbs: List[BasicBlock] = []
    # list of locations in the CFG
    locations = []
    # list of types of locations in the CFG k:v -> location:type
    types: Dict[int, str] = {}

    def __init__(self):
        self.bbs = []
        self.locations = []
        self.types = {}

    def __repr__(self):
        return f"CFG(bbs={self.bbs},\n locations={self.locations},\n types={self.types})"

    def index_of(self, node: BasicBlock):
        return self.bbs.index(node)

    def add_stmt_bb(self, bb_id: int, stmt: Statement):
        self.bbs[bb_id].stmts.append(stmt)

    def add_bb(self, node: BasicBlock):
        self.bbs.append(node)
        # sort bbs by name (int)
        self.bbs.sort(key=lambda x: x.name)

    def add_edge(self, fro: BasicBlock, to: BasicBlock):
        # set succ of fro to to
        self.bbs[self.index_of(fro)].succ.append(self.index_of(to))
        # set pred of to to fro
        self.bbs[self.index_of(to)].pred.append(self.index_of(fro))

    def pprint(self):
        for n in self.bbs:
            print(f"BB {n.name}:\n\tSucc: {n.succ}\n\tPred: {n.pred}\n\tStmts (num: {len(n.stmts)}: {n.stmts}")


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
