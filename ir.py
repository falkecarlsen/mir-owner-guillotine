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
    UNWRAP = auto()


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
    pred: Set[Tuple[Optional[int], int]] = field(default_factory=set)
    succ: Set[Tuple[Optional[int], int]] = field(default_factory=set)
    # liveness analysis variables
    # in[i] = gen[i] U (out[i] \ kill[i])
    live_in: Set[int] = field(default_factory=set)
    # out[i] = U in[j] where j is successor of i
    live_out: Set[int] = field(default_factory=set)
    # gen[i] = {x | x is used in Statement() i}
    gen: Set[int] = field(default_factory=set)
    # kill[i] = {x | x is defined in Statement() i}
    kill: Set[int] = field(default_factory=set)

    # return locations that are generated by this statement
    def generates(self):
        return {self.lhs_location}

    # return locations that are killed by this statement
    def kills(self):
        # kills every use, consts are not killed, if assigning a location, kill the location
        if self.rhs_location:
            return {self.rhs_location}
        else:
            return set()

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
            f"\t\tgen={self.gen}\n"
            f"\t\tkill={self.kill}\n"
            f"\t)"

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

    def kills(self):
        kill_set = set()
        for arg in self.function_args:
            if arg.location is not None:
                if arg.mode in [Mode.MOVE, Mode.NONE]:
                    if arg.location:
                        kill_set.add(arg.location)
        return kill_set

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

    def generates(self):
        return set()

    def kills(self):
        kill_set = set()
        for arg in self.primitive_args:
            # TODO: fix this, happens somewhere in valueargs, where arg is nested for some reason
            # TODO: HACK: if arg is list, unwrap to one FunctionArg
            if isinstance(arg, list):
                arg = arg[0]

            if arg.location is not None:
                if arg.mode in [Mode.MOVE, Mode.NONE]:
                    if arg.location:
                        kill_set.add(arg.location)
        return kill_set

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
    # list of BB edges
    edges: List[Tuple[int, int]] = []
    # entry and exit nodes, referring to bb ids
    entry: int = None
    exit: int = None
    # list of types of locations in the CFG k:v -> location:type
    _types: Dict[int, str] = {}

    def find_and_set_entry_exit(self):
        """
        Find entry and exit nodes in the CFG
        """
        for bb in self.bbs:
            if not bb.pred:
                self.entry = bb.name
            if not bb.succ:
                self.exit = bb.name

    def add_edge(self, pred: int, succ: List[int]):
        # if only one succ, add it
        if isinstance(succ, int):
            if pred > succ:
                pred, succ = succ, pred
            self.edges.append((pred, succ))
        else:
            # for each succ, add them to set of bb succs
            for s in succ:
                self.edges.append((pred, s))

    def fill_in_bb_pred_succ(self):
        # for each bb, add pred and succ according to self.edges
        for e in self.edges:
            pred, succ = e
            for bb in self.bbs:
                if bb.name == pred:
                    bb.succ.add(succ)
                if bb.name == succ:
                    bb.pred.add(pred)


        for bb in self.bbs:
            for succ_index in bb.succ:
                pred = self.index_of(bb)
                pass
                # self.bbs[succ_index].pred.add(pred)

    def finalise_cfg(self):
        self.fill_in_bb_pred_succ()
        # compute succ and pred for each stmt in each bb
        for bb in self.bbs:
            for i, stmt in enumerate(bb.stmts):
                # fixme; hacky, as succs are bb-indicies not stmt indicies
                # use tuples (bb_index, stmt_index), bb_index None if within same bb
                # bb_index points to succ/pred bb_index if traversing basic blocks

                # last stmt in bb
                if i == len(bb.stmts) - 1:
                    stmt.pred.add((None, i - 1))
                    for succ in bb.succ:
                        # (bb_index, stmt_index) always the first in a succ
                        stmt.succ.add((succ, 0))

                # first stmt in bb
                if i == 0:
                    stmt.succ.add((None, i + 1))
                    for pred in bb.pred:
                        # (bb_index, stmt_index) always the last in a pred
                        stmt.pred.add((pred, len(self.bbs[pred].stmts) - 1))

                # middle stmt in bb
                else:
                    stmt.succ.add((None, i + 1))
                    stmt.pred.add((None, i - 1))

                # run gens and kills on each stmt
                stmt.gen = stmt.generates()
                stmt.kill = stmt.kills()

        # set entry and exit nodes
        self.find_and_set_entry_exit()

        # assert that all succ and preds in CFG are of the type Set[Tuple[Optional[int], int]]
        for bb in self.bbs:
            for stmt in bb.stmts:
                assert all(isinstance(s, tuple) for s in stmt.succ)
                assert all(isinstance(p, tuple) for p in stmt.pred)

    def index_of(self, elem) -> int:
        # if elem is a bb, return index of bb in bbs
        if isinstance(elem, BasicBlock):
            return self.bbs.index(elem)
        # if elem is a stmt, return index of bb in bbs
        elif isinstance(elem, Statement):
            for bb in self.bbs:
                if elem in bb.stmts:
                    return self.bbs.index(bb)

    def add_bb(self, node: BasicBlock):
        self.bbs.append(node)
        # sort bbs by name (int)
        self.bbs.sort(key=lambda x: x.name)

    def __repr__(self):
        return f"CFG(bbs={self.bbs},\n locations={self.locations},\n types={self._types})"

    def pprint(self):
        for n in self.bbs:
            print(f"BB {n.name}:\n\tSucc: {n.succ}\n\tPred: {n.pred}\n\tStmts (num: {len(n.stmts)}: {n.stmts}")
        print(f"CFG edges: {self.edges}")
        # print succs and preds for each bb
        for bb in self.bbs:
            print(f"BB {bb.name}:\n\tSucc: {bb.succ}\n\tPred: {bb.pred}")
        print(f"CFG entry: {self.entry}, exit: {self.exit}")


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
        return f"CFGUDChain(bbs={self.bbs},\n locations={self.locations},\n types={self._types})"

    def compute_reaching_definitions(self):
        """
        Compute reaching definitions for each statement in the CFG
        """
        # for each bb, compute reaching definitions
        # collect list of stmts into a flattened list
        # computing succs and defs for each
        count = 0
        changed = True
        while changed:
            for b_i, bb in enumerate(self.bbs):
                for s_i, stmt in enumerate(bb.stmts):
                    print(f"count {count}: {stmt}")
                    # remember old IN and OUT to check change, stopping at fixed point (sup/inf?)
                    old_in = stmt.live_in
                    old_out = stmt.live_out
                    # compute IN[n] = union OUT[p] for each p in pred[n]
                    # check if we are not at start node, then compute IN
                    if not (b_i == 0 and s_i == 0):
                        if len(stmt.pred) == 1:
                            # just one incoming edge, so IN = OUT[pred]
                            stmt.live_in = self.bbs[b_i].stmts[s_i - 1].live_out
                        # else, use last stmt in pred bb
                        else:
                            # multiple incoming edges, so IN = intersection OUT[pred]
                            for (bb_index, stmt_index) in stmt.pred:
                                # if bb_index is None, then within same bb
                                if bb_index is None:
                                    stmt.live_in = stmt.live_in.union(self.bbs[b_i].stmts[stmt_index].live_out)
                                else:
                                    # FIXME
                                    stmt.live_in = stmt.live_in.union(self.bbs[bb_index].stmts[-1].live_out)

                    # compute OUT[n] = gen[n] U (IN[n] - kill[n])
                    gen = stmt.gen
                    kill = stmt.kill
                    union = gen.union(stmt.live_in.difference(kill))
                    # remove erratic None value from union
                    if None in union:
                        union.remove(None)

                    stmt.live_out = union

                    # check if changed, repeating if change
                    changed = stmt.live_in != old_in or stmt.live_out != old_out
                    count += 1

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
