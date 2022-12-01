import re
from typing import List

from sly import Lexer, Parser

import ir

"""
Rust MIR simplified grammar for borrow-checking and CFG generation 

<mir> ::= <function>*
<function> ::= FN <name> TO "(" <param> ")" "{" <block>+ "}"
<param> ::= (<name> ("," <name>)*)+
<block> ::= <name> ":" "{" <statement>* "}"
?<statements> ::= <statement> <statements> | <statement>
<statement> ::= <assignment> | <borrow> | <mut_borrow> | <return> | <call>
<assignment> ::= <expr> ";"
<borrow> ::= "&" <expr> ";"
<mut_borrow> ::= "&mut" <expr> ";"
<return> ::= RETURN ";"
<call> ::= <name> "(" <expr> ")" TO <expr> ;"
<expr> ::= <name> | <literal>
<name> ::= [a-zA-Z_][a-zA-Z0-9_]*
<literal> ::= [0-9]+
"""


# noinspection PyUnresolvedReferences,PyUnboundLocalVariable
class MirLexer(Lexer):
    # Tokens
    literals = {'+', '-', '*', '/', '(', ')', '=', ':', ';', ',', '[', ']', '{', '}', '_', '<', '>'}
    tokens = {
        LOCATION,
        FN,
        NAME,
        LET,
        LETMUT,
        REF,
        REFMUT,
        BB,
        RETURN,
        DEREF,
        TYPENAMES,
        METHODNAMES,
        CONST,
        MOVE,
        NUMBER,
        UNREACHABLE,
        GOTO,
        ARROW,
        COLONTWICE,
        STRING,
        OTHERWISE,
        PRIMITIVES,
        AS,
    }
    ignore = ' \t'
    # ignore // comments
    ignore_comment = r'//.*'

    # Tokens
    LOCATION = r'\_\d+'
    FN = r'fn'
    LET = r'let'
    LETMUT = r'letmut'
    BB = r'bb\d+'

    ARROW = r'->'
    GOTO = r'goto'
    COLONTWICE = r'::'
    REFMUT = r'&mut'
    REF = r'&'
    DEREF = r'\*'
    # s = ["Hashmap", "u32", "String", "string", "str", "Some", "std", "Index"]
    # m = ["insert", "get", "index", "from", "discriminant", "switchInt"]
    # stucts -> '|'.join(s) + '|'.join(['&'+n for n in s]) + '|'.join(['&mut'+n for n in s])
    # methods -> '|'.join(m)
    TYPENAMES = r'|'.join(
        [
            r'i32',
            r'u32',
            r'i64',
            r'u64',
            r'f32',
            r'f64',
            r'isize',
            r'bool',
            r'char',
            r'str',
            r'string',
            r'String',
            r'Some',
            r'HashMap',
            r'Index',
            r'From',
            r'discriminant',
        ]
    )
    AS = r'as'

    METHODNAMES = r'|'.join(["index", "insert", "from", "get"])
    PRIMITIVES = r'|'.join(["switchInt"])
    CONST = r'const'
    MOVE = r'move'
    RETURN = r'return'
    UNREACHABLE = r'unreachable'
    OTHERWISE = r'otherwise'
    STRING = r'\".*?\"'

    # Ignored pattern
    ignore_newline = r'\n+'

    # Extra action for newlines
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    # This decorator allows us to add a logic before returning the matched token.
    @_(r"(0|[1-9][0-9]*)")
    def NUMBER(self, t):
        t.value = int(t.value)
        return t

    def error(self, t):
        print("Illegal character '%s'" % t.value[0])
        self.index += 1


# noinspection PyUnresolvedReferences
class MirParser(Parser):
    tokens = MirLexer.tokens
    debugfile = 'parser.out'
    start = 'block'

    def __init__(self):
        # legacy
        self.curr_bb_id: int = -1
        self.locations = {}
        self.types = {}
        self.names = {}
        # data-ir
        self.cfg: ir.CFG = ir.CFG()
        self.temp_stmts: List[ir.Statement] = []
        self.stmt: ir.Statement = ir.Statement()

    @staticmethod
    def get_loc_or_bb_int(string):
        # get int from location string
        if string is list:
            return string
        return int(re.sub(r'\D', '', string))

    def add_curr_stmt_and_reset(self):
        self.temp_stmts.append(self.stmt)
        self.stmt = ir.Statement()

    # function fixme
    @_('FN NAME "(" "," ")" "{" BB "}"')
    def function(self, p):
        print('function', p.NAME, p.params, p.blocks)

    # MIR type definitions
    @_('LET LOCATION ":" TYPENAMES ";"')
    def location_type_immut(self, p):
        self.types[p.LOCATION] = p.TYPENAMES
        return p.LOCATION

    @_('LETMUT LOCATION ":" TYPENAMES ";"')
    def location_type_mut(self, p):
        self.types[p.LOCATION] = p.TYPENAMES
        return p.LOCATION

    # block
    @_('BB ":" "{" stmtlist "}"')
    def block(self, p):
        try:
            self.curr_bb_id = self.get_loc_or_bb_int(p.BB)
            # create BasicBlock and add to CFG
            bb = ir.BasicBlock(self.curr_bb_id)
            # add temp statements to BasicBlock
            bb.add_statements(self.temp_stmts)
            self.cfg.add_bb(bb)

        except ValueError:
            print('ERROR: Invalid BB id', p.BB)
            exit(1)

        print(f'block{self.curr_bb_id} end')
        print(f"flushing {len(self.temp_stmts)} temp_stmts")
        self.temp_stmts.clear()
        return self.cfg

    @_('')
    def block_start(self, _p):
        print("block start")

    # bblist
    @_('block bblist')
    def bblist(self, p):
        return p.block + p.bblist

    @_('')
    def bblist(self, _p):
        return []

    # stmtlist -> stmtlist statement | statement
    @_('stmtlist statement')
    def stmtlist(self, p):
        print('stmtlist, statment', p.statement)
        return p.stmtlist

    @_('statement')
    def stmtlist(self, p):
        print('stmtlist', p.statement)
        return [p.statement]

    # statement -> LOCATION = stmttype ; | GOTO ARROW BB ; | UNREACHABLE ; | RETURN | primitives ;
    @_('LOCATION "=" stmttype ";"')
    def statement(self, p):
        curr_stmt_id = self.get_loc_or_bb_int(p.LOCATION)
        last_stmt = self.stmt
        # if stmttype is function or primitive, just return p.stmttype
        stmttype = p.stmttype
        if stmttype is ir.FunctionStatement or stmttype is ir.PrimitiveFunctionStatement:
            return stmttype
        # if last stmt is an assignment, then we need to assign the curr_stmt_id
        # fixme, not needed?
        match last_stmt.stmt_type:
            # do location assignment
            case ir.StatementType.ASSIGN | ir.StatementType.FUNCTION_CALL:
                # check location is not in set of seen locations of temp_stmts for current bb
                seen = [n.lhs_location for n in temp_stmts]
                if curr_stmt_id in seen:
                    print(f'ERROR: curr_stmt_id {curr_stmt_id} already used')
                    exit(1)
                else:
                    last_stmt.lhs_location = curr_stmt_id
                    print(f"set lhs_location of last_stmt to {curr_stmt_id}")
            case ir.StatementType.UNREACHABLE | ir.StatementType.RETURN:
                pass
            case ir.StatementType.GOTO:
                pass
        # add curr to temp and reinitialise self.curr_stmt
        self.add_curr_stmt_and_reset()

        print('statement', p.LOCATION, p.stmttype)
        return p.stmttype

    @_('GOTO ARROW BB ";"')
    def statement(self, p):
        print('goto', p.BB)
        self.stmt.stmt_type = ir.StatementType.GOTO
        self.stmt.bb_target = self.get_loc_or_bb_int(p.BB)
        self.add_curr_stmt_and_reset()

    @_('UNREACHABLE ";"')
    def statement(self, p):
        print('unreachable', p.UNREACHABLE)
        self.stmt.stmt_type = ir.StatementType.UNREACHABLE
        self.add_curr_stmt_and_reset()

    @_('RETURN ";"')
    def statement(self, p):
        print('return', p.RETURN)
        self.stmt.stmt_type = ir.StatementType.RETURN
        self.add_curr_stmt_and_reset()

    # stmttype -> LOCATION | constant | borrow | goto | unreachable | return | function_call
    @_('LOCATION')
    def stmttype(self, p):
        print('stmttype location', p.LOCATION)
        # create statement
        self.stmt.stmt_type = ir.StatementType.ASSIGN
        self.stmt.rhs_location = self.get_loc_or_bb_int(p.LOCATION)

    @_('constant')
    def stmttype(self, p):
        print('stmttype', p.constant)
        self.stmt.stmt_type = ir.StatementType.ASSIGN
        self.stmt.rhs_value = p.constant[0]
        self.stmt.value_type = p.constant[0]
        self.stmt.rhs_value = ir.ValueType.CONST
        return p.constant

    # constant -> CONST NUMBER _ TYPE
    @_('CONST NUMBER "_" TYPENAMES')
    def constant(self, p):
        print('constant item', p.CONST, p.NUMBER, p.TYPENAMES)
        return p.NUMBER, p.TYPENAMES

    @_('borrow')
    def stmttype(self, p):
        print('stmttype', p.borrow)
        self.stmt.stmt_type = ir.StatementType.ASSIGN
        return p.borrow

    # borrow -> REF source | REFMUT source
    @_('REF source')
    def borrow(self, p):
        print('borrow', p.source)
        self.stmt.value_type = ir.ValueType.BORROW
        self.stmt.mutability = False
        return p.source

    @_('REFMUT source')
    def borrow(self, p):
        print('borrow', p.source)
        self.stmt.value_type = ir.ValueType.BORROW
        self.stmt.mutability = True
        return p.source

    # source -> ( source ) | LOCATION | DEREF LOCATION
    @_('"(" source ")"')
    def source(self, p):
        print('source parens', p.source)
        return p.source

    @_('LOCATION')
    def source(self, p):
        print('source location', p.LOCATION)
        self.stmt.rhs_location = self.get_loc_or_bb_int(p.LOCATION)
        return p.LOCATION

    @_('DEREF LOCATION')
    def source(self, p):
        print('source deref location', p.LOCATION)
        self.stmt.rhs_location = self.get_loc_or_bb_int(p.LOCATION)
        return p.LOCATION

    @_('PRIMITIVES "(" valueargs ")" goto_cond_block ";"')
    def statement(self, p):
        print('stmt primitive', p.PRIMITIVES, p.valueargs, p.goto_cond_block if p.goto_cond_block else '')
        self.stmt = ir.PrimitiveFunctionStatement()

        self.stmt.primitive_type = p.PRIMITIVES
        self.stmt.primitive_args = p.valueargs
        self.stmt.primitive_bb_goto = p.goto_cond_block if p.goto_cond_block else None
        self.add_curr_stmt_and_reset()

    @_('function_call')
    def stmttype(self, p):
        print('stmttype function_call', p.function_call)
        return p.function_call

    # todo: function
    """              |-| method call on Struct
    Struct::<A,B,C>::new(x, y) -> bb1;
    ^ struct ^ type-args ^value args
    """

    # issue is recognising all:
    #   Type::<Type, Type>::method(args)
    #   Type::<Type, Type>::method::<Type, Type>(args) fixme
    # these due to:
    # <HashMap<u32, String> as Index<&u32>>::index(move _2, move _3) -> bb69
    # <String as From<&str>>::from(const "init") -> bb69

    # todo; update grammar to reflect impl, and get rid of 7 shift/reduce conflicts
    # function_call -> generic COLONTWICE method_call ( valueargs ) goto_block
    # method_call -> METHODNAME | METHODNAME turbofish
    # turbofish -> COLONTWICE "<" typeargs ">"
    # generic -> generic COLONTWICE "<" typeargs ">" cast
    #            | TYPENAMES "<" typeargs ">" cast
    #            | TYPENAMES
    #            | generic cast
    #            | "<" generic ">"
    # cast -> AS TYPENAMES < typeargs > | empty
    # typeargs -> typearg "," typeargs | typearg
    # typearg -> TYPENAMES | REF TYPENAMES | REFMUT TYPENAMES

    @_(
        'generic COLONTWICE method_call "(" valueargs ")" goto_block',
        'generic COLONTWICE method_call "(" valueargs ")"',
    )
    def function_call(self, p):
        print('function_call', p.generic, p.method_call, p.valueargs, p.goto_block if p.goto_block else '')
        self.stmt: ir.FunctionStatement = ir.FunctionStatement()
        self.stmt.function_type = p.generic
        self.stmt.function_method = p.method_call
        self.stmt.function_args = p.valueargs
        self.stmt.function_bb_goto = p.goto_block if p.goto_block else None
        self.add_curr_stmt_and_reset()
        return ir.FunctionStatement

    # HACK: primitive MIR-builtins can assign, but are functions
    @_('TYPENAMES "(" valueargs ")"')
    def function_call(self, p):
        print('function_call', p.TYPENAMES, p.valueargs)
        self.stmt: ir.PrimitiveFunctionStatement = ir.PrimitiveFunctionStatement()
        self.stmt.primitive_type = p.TYPENAMES
        self.stmt.function_args = p.valueargs
        self.add_curr_stmt_and_reset()
        return ir.PrimitiveFunctionStatement

    @_('generic COLONTWICE "<" typeargs ">" cast')
    def generic(self, p):
        print('generic', p.generic, p.typeargs)
        return p.generic, p.typeargs, p.cast if p.cast else None

    @_('TYPENAMES "<" typeargs ">" cast')
    def generic(self, p):
        print('generic', p.TYPENAMES, p.typeargs)
        return p.TYPENAMES, p.typeargs, p.cast if p.cast else None

    @_('TYPENAMES')
    def generic(self, p):
        print('generic', p.TYPENAMES)
        return p.TYPENAMES, None, None

    @_('generic cast')
    def generic(self, p):
        print('generic', p.generic)
        return p.generic, None, p.cast if p.cast else None

    @_('"<" generic ">"')
    def generic(self, p):
        print('generic', p.generic)
        return p.generic, None, None

    @_('AS TYPENAMES "<" typeargs ">"')
    def cast(self, p):
        print('cast', p.TYPENAMES, p.typeargs)
        return p.TYPENAMES, p.typeargs

    @_('')
    def empty(self, p):
        pass

    @_('empty')
    def cast(self, p):
        pass

    # method_call -> METHODNAME | METHODNAME turbofish
    @_('METHODNAMES')
    def method_call(self, p):
        print('method_call', p.METHODNAMES)
        return p.METHODNAMES

    @_('METHODNAMES turbofish')
    def method_call(self, p):
        print('method_call', p.METHODNAMES, p.turbofish)
        return p.METHODNAMES, p.turbofish

    # turbofish -> COLONTWICE "<" typeargs ">"
    @_('COLONTWICE "<" typeargs ">"')
    def turbofish(self, p):
        print('turbofish', p.typeargs)
        return p.typeargs

    # typeargs -> typearg "," typeargs | typearg
    # typearg -> TYPENAMES | REF TYPENAMES | REFMUT TYPENAMES

    @_('typearg "," typeargs')
    def typeargs(self, p):
        print('typeargs', p.typearg, p.typeargs)
        return [p.typearg] + p.typeargs

    @_('typearg')
    def typeargs(self, p):
        print('typeargs', p.typearg)
        return [p.typearg]

    @_('TYPENAMES')
    def typearg(self, p):
        print('typearg', p.TYPENAMES)
        return p.TYPENAMES

    @_('REF TYPENAMES')
    def typearg(self, p):
        print('typearg', p.TYPENAMES)
        return p.REF + p.TYPENAMES

    @_('REFMUT TYPENAMES')
    def typearg(self, p):
        print('typearg', p.TYPENAMES)
        return p.REFMUT + p.TYPENAMES

    # valueargs -> valueargs "," valuearg | valuearg
    @_('valueargs "," valuearg')
    def valueargs(self, p):
        print('valueargs', p.valuearg)
        return [p.valueargs] + [p.valuearg]

    @_('valuearg')
    def valueargs(self, p):
        print('valuearg', p.valuearg)
        return p.valuearg

    # valuearg -> LOCATION | MOVE LOCATION | valuearg_constant (reuse from stmt) | CONST STRING
    @_('LOCATION')
    def valuearg(self, p):
        print('valuearg loc', p.LOCATION)
        return ir.FunctionArg(location=p.LOCATION)

    @_('MOVE LOCATION')
    def valuearg(self, p):
        print('valuearg move loc', p.LOCATION)
        return ir.FunctionArg(mode=ir.Mode.MOVE, location=p.LOCATION)

    @_('constant')
    def valuearg(self, p):
        print('valuearg constant', p.constant)
        # p.constant[0] = number, p.constant[1] = type
        return ir.FunctionArg(constant=(p.constant[0], p.constant[1]))

    @_('CONST STRING')
    def valuearg(self, p):
        print('valuearg', p.STRING)
        # from method(const "init"), must be String due to typing of function call
        return ir.FunctionArg(constant=(p.STRING, "String"))

    # goto_block -> ARROW BB
    @_('ARROW BB')
    def goto_block(self, p):
        print('goto_block', p.BB)
        return self.get_loc_or_bb_int(p.BB)

    # goto_cond_block -> ARROW "[" goto_params "]"
    @_('ARROW "[" goto_params "]"')
    def goto_cond_block(self, p):
        print('goto_cond_block', p.goto_params)
        return p.goto_params

    # goto_params -> goto_params "," goto_param | goto_param
    @_('goto_params "," goto_param')
    def goto_params(self, p):
        print('goto_params', p.goto_params, p.goto_param)

        # neatly:tm: collect bb gotos in list
        if p.goto_params is None:
            return [self.get_loc_or_bb_int(p.goto_param)]
        elif type(p.goto_params) is list:
            return p.goto_params + [self.get_loc_or_bb_int(p.goto_param)]
        else:
            return [self.get_loc_or_bb_int(p.goto_params)] + [self.get_loc_or_bb_int(p.goto_param)]

    @_('goto_param')
    def goto_params(self, p):
        print('goto_param', p.goto_param)
        return p.goto_param

    # goto_param -> NUMBER "_" TYPENAMES ":" BB | OTHERWISE ":" BB
    @_('NUMBER "_" TYPENAMES ":" BB')
    def goto_param(self, p):
        print('goto_param', p.NUMBER, p.TYPENAMES, p.BB)
        return p.BB

    @_('OTHERWISE ":" BB')
    def goto_param(self, p):
        print('goto_param', p.BB)
        return p.BB


def parse(mir_program):
    mir_lexer = MirLexer()
    mir_parser = MirParser()
    return mir_parser.parse(mir_lexer.tokenize(mir_program))


if __name__ == '__main__':
    bold = '\033[1m'
    unbold = '\033[0m'
    header = "=" * 80

    text = open('mir-input-grammar/pass/functions.mir', 'r').read()

    print(f"{header}\nlexing: ")
    mir_lexer = MirLexer()
    for tok in mir_lexer.tokenize(text=text):
        print(
            f"type= {bold}{tok.type:<11s}{unbold} "
            f"value= {bold}{tok.value:<11}{unbold} "
            f"lineno={tok.lineno:<10} "
            f"index={tok.index:<10} "
            f"end={tok.end:<10}"
        )

    print(f"{header}\nparsing: ")
    res = parse(text)
    print(f"{header}\nparsing result: ")

    print(f"{header}\ncfg pprint: ")
    res.pprint()
