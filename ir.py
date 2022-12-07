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
    # liveness analysis variables
    # in[i] = gen[i] U (out[i] \ kill[i])
    live_in: Set[int] = field(default_factory=set)
    # out[i] = U in[j] where j is successor of i
    live_out: Set[int] = field(default_factory=set)
    # gen[i] = {x | x is used in Statement() i}
    gen: Set[int] = field(default_factory=set)
    # kill[i] = {x | x is defined in Statement() i}
    kill: Set[int] = field(default_factory=set)

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
    NOT_MOVE = auto()
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
    bb_goto: Optional[int] = None

    def __repr__(self):
        return (
            f"\tFunctionStatement(\n"
            f"\t\tfunction_method={self.function_method},\n"
            f"\t\tfunction_type={self.function_type}, \n"
            f"\t\tfunction_args={self.function_args}, \n"
            f"\t\tbb_goto={self.bb_goto}, \n"
        )


@dataclass(kw_only=True)
class PrimitiveFunctionStatement(Statement):
    """
    Primitive statement occurring in MIR with concomitant data relevant for building CFG+BB IR
    """

    primitive_type: str = None
    primitive_args: Optional[List[FunctionArg]] = field(default_factory=list)
    bb_goto: Optional[List[int]] = None

    # repr
    def __repr__(self):
        return (
            f"\tPrimitiveFunctionStatement(\n"
            f"\t\tprimitive_type={self.primitive_type},\n"
            f"\t\tprimitive_args=\n{self.primitive_args}, \n"
            f"\t\tbb_goto={self.bb_goto}\n"
        )


@dataclass
class BasicBlock:
    name: int
    stmts: List[Statement] = field(default_factory=list)
    succ: Set[int] = field(default_factory=set)
    pred: Set[int] = field(default_factory=set)
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
        self.bbs: List[BasicBlock] = []
        # edges as list of tuples (idx of pred, succ)
        self.edges: List[Tuple[int, int]] = []

    def add_edge(self, pred: int, succ: List[int]):
        # if only one succ, add it
        if isinstance(succ, int):
            if pred > succ:
                pred, succ = succ, pred
            self.edges.append((pred, succ))
            # give blocks pred and succ
            self.bbs[pred].succ.add(succ)
        else:
            # for each succ, add them to set of bb succs
            for s in succ:
                self.edges.append((pred, s))
                # give blocks pred and succ
                self.bbs[pred].succ.add(s)

    def fill_in_pred_bb(self):
        for bb in self.bbs:
            for succ_index in bb.succ:
                self.bbs[succ_index].pred.add(self.index_of(bb))

    def index_of(self, node: BasicBlock) -> int:
        return self.bbs.index(node)

    def add_bb(self, node: BasicBlock):
        self.bbs.append(node)
        # sort bbs by name (int)
        self.bbs.sort(key=lambda x: x.name)

    def __repr__(self):
        return f"CFG(bbs={self.bbs},\n locations={self.locations},\n types={self.types})"

    def pprint(self):
        for n in self.bbs:
            print(f"BB {n.name}:\n\tSucc: {n.succ}\n\tPred: {n.pred}\n\tStmts (num: {len(n.stmts)}: {n.stmts}")
        print(f"CFG edges: {self.edges}")
        # print succs and preds for each bb
        for bb in self.bbs:
            print(f"BB {bb.name}:\n\tSucc: {bb.succ}\n\tPred: {bb.pred}")


class CFGUDChain(CFG):
    """
    Use-Define chain computation on CFG class. Dataflow analysis.
    Will need Reaching Definitions analysis to be done first. We only have unambiguous definitions in MIR (correct?)
        Reaching Definitions pseudocode:
        variables:
            IN is a set of definitions that reach the current statement
            OUT is a set of definitions that are reachable from the current statement
            GEN is a set of definitions that are generated by the current statement
            KILL is a set of definitions that are killed by the current statement
        algorithm:
            IN[entry] = {}
            OUT[entry] = {}
            for each node n in CFG:
                IN[n] = {}
                OUT[n] = {}
            while True:
                change = False
                for each node n in CFG:
                    IN[n] = U OUT[p] for each p in pred[n]
                    OUT[n] = gen[n] U (IN[n] - kill[n])
                    if IN[n] != old_IN[n] or OUT[n] != old_OUT[n]:
                        change = True
                if not change:
                    break

        Using previously computed Reaching Definitions, compute Use-Define chains:




    """

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"CFGUDChain(bbs={self.bbs},\n locations={self.locations},\n types={self.types})"

    def compute_reaching_defs(self):
        """
        Compute reaching definitions for each statement in each BB, then compute livein and liveout for each BB
        """
        # compute reaching definitions for each stmt in each bb
        for bb in self.bbs:
            # init reaching definitions to empty set
            bb.reaching_defs = [set() for _ in range(len(bb.stmts))]
            # compute reaching definitions for each stmt in bb
            for i in range(len(bb.stmts)):
                # init reaching definitions for stmt i to empty set
                bb.reaching_defs[i] = set()
                # add reaching definitions of all preds of bb to stmt i reaching definitions
                for pred in bb.pred:
                    bb.reaching_defs[i] = bb.reaching_defs[i].union(self.bbs[pred].reaching_defs[-1])
                # remove reaching definitions of stmt i from reaching definitions of stmt i
                bb.reaching_defs[i] = bb.reaching_defs[i].difference(bb.stmts[i].locations)
                # add reaching definitions of stmt i to reaching definitions of stmt i
                bb.reaching_defs[i] = bb.reaching_defs[i].union(bb.stmts[i].locations)
        # compute livein and liveout for each bb
        for bb in self.bbs:
            # init livein and liveout to empty set
            bb.livein = set()
            bb.liveout = set()
            # compute liveout for each bb
            for succ in bb.succ:
                bb.liveout = bb.liveout.union(self.bbs[succ].livein)
            # compute livein for each bb
            bb.livein = bb.liveout.difference(bb.defs).union(bb.uses)



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
