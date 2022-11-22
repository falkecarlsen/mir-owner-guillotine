import re
from pprint import pprint

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

cfg = ir.CFG()
curr_bb_id = -1


# noinspection PyUnresolvedReferences,PyUnboundLocalVariable
class MirLexer(Lexer):
    # Tokens
    literals = {'+', '-', '*', '/', '(', ')', '=', ':', ';', ',', '[', ']', '{', '}', '_'}
    tokens = {
        LOCATION,
        FN,
        NAME,
        LET,
        LETMUT,
        REF,
        REFMUT,
        TYPE,
        BB,
        PARAMS,
        EXPR,
        RETURN,
        STMT,
        DEREF,
        TYPES,
        FUNCTIONS,
        CONSTANTS,
        MOVE,
        STATEMENT,
        NUMBER,
        UNREACHABLE,
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

    TYPE = r'[a-zA-Z_][a-zA-Z0-9_] | \(\)*'
    PARAMS = r'\((\s*[a-zA-Z_][a-zA-Z0-9_]*\s*,)*\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)'
    EXPR = r'\((\s*[a-zA-Z_][a-zA-Z0-9_]*\s*,)*\s*[a-zA-Z_][a-zA-Z0-=9_]*\s*\)'

    # statements; assignment, borrow, mutborrow, return, call
    REFMUT = r'&mut'
    REF = r'&'
    DEREF = r'\*'
    # TYPES = [r'i32', r'u32', r'i64', r'u64', r'f32', r'f64', r'bool', r'char', r'str', r'String', r'()']
    TYPES = r'i32'
    FUNCTIONS = ["HashMap", "get"]
    CONSTANTS = r'const'
    MOVE = r'move'
    RETURN = r'return'
    UNREACHABLE = r'unreachable'

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

    precedence = (
        ('left', STMT),
        ('left', LOCATION),
    )

    def __init__(self):
        # legacy
        self.locations = {}
        self.types = {}
        self.names = {}
        # data-ir

    # function
    @_('FN NAME "(" PARAMS ")" "{" BB "}"')
    def function(self, p):
        print('function', p.NAME, p.params, p.blocks)

    # MIR type definitions
    @_('LET LOCATION ":" TYPE ";"')
    def location_type_immut(self, p):
        self.types[p.LOCATION] = p.TYPE
        return p.LOCATION

    @_('LETMUT LOCATION ":" TYPE ";"')
    def location_type_mut(self, p):
        self.types[p.LOCATION] = p.TYPE
        return p.LOCATION

    # block
    @_('BB ":" "{" stmtlist "}"')
    def block(self, p):
        global curr_bb_id
        try:
            curr_bb_id = int(p.BB[-1])
            # create BasicBlock and add to CFG
            bb = data.BasicBlock(curr_bb_id)
            cfg.add_bb(bb)

        except ValueError:
            print('ERROR: Invalid BB id', p.BB)
            exit(1)

        print(f'block{curr_bb_id} end')
        return p.stmtlist

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

    # stmtlist
    @_('stmtlist statement')
    def stmtlist(self, p):
        # do something with p.statement
        print('stmtlist, statment', p.statement)
        return p.stmtlist

    @_('statement')
    def stmtlist(self, p):
        print('stmtlist', p.statement)
        return [p.statement]

    # multiple statements
    @_('statement ";" statement')
    def statements(self, p):
        # NOP, do rules:
        # stmts -> stmt | stmts stmt
        print('multiple statements', p.statement0, p.statement1)
        return p.statement0, p.statement1

    @_('statement ";"')
    def statements(self, p):
        print('single statement', p.statement)
        return p.statement

    # assignment of location to location
    @_('LOCATION "=" LOCATION ";"')
    def statement(self, p):
        assign_id = int(p.LOCATION0[1:])
        value_id = int(p.LOCATION1[1:])

        print('statement location assignment ', assign_id, ' value ', value_id)
        return p.LOCATION0, p.LOCATION1

    # assignment constant to location
    @_('LOCATION "=" CONSTANTS NUMBER "_" TYPES ";"')
    def statement(self, p):
        print('const statement', p.LOCATION, p.NUMBER, p.TYPES)
        location_id = int(p.LOCATION[1:])
        self.locations[location_id] = p.NUMBER
        # maybe do check if fn-defined type corresponds with currently seen type
        self.types[location_id] = p.TYPES
        return p.LOCATION, p.NUMBER, p.TYPES

    # mut borrow location statement
    @_('LOCATION "=" REFMUT  "(" DEREF LOCATION ")" ";"')
    def statement(self, p):
        borrower_id = int(p.LOCATION0[1:])
        borrowee_id = int(p.LOCATION1[1:])
        print('mut borrow statement', borrower_id, borrowee_id)

        self.types[borrower_id] = "&mut_" + str(self.types[borrowee_id])
        self.locations[borrower_id] = self.locations[borrowee_id]
        # do cfg borrow mut location
        return borrower_id, borrowee_id

    # deref location statement
    @_('DEREF LOCATION')
    def statement(self, p):
        print('deref statement', p.LOCATION)
        # do cfg deref location

        return p.LOCATION

    """
    # parens location statement
    @_('"(" LOCATION ")"')
    def statement(self, p):
        print('parens statement', p.LOCATION)
        return p.LOCATION
    """

    @_('RETURN ";"')
    def statement(self, p):
        print('statement return')
        # close block scope?

    # unreachable statement
    @_('UNREACHABLE ";"')
    def statement(self, p):
        print('statement unreachable')
        # do cfg unreachable

    # types
    @_('TYPES')
    def types(self, p):
        print("type: ", p.TYPES)
        return p.TYPES


def parse(mir_program):
    mir_lexer = MirLexer()
    mir_parser = MirParser()
    return mir_parser.parse(mir_lexer.tokenize(mir_program))


if __name__ == '__main__':
    bold = '\033[1m'
    unbold = '\033[0m'

    lexer = MirLexer()
    parser = MirParser()
    text = open('mir-input-grammar/pass/test.mir', 'r').read()

    for tok in lexer.tokenize(text):
        print(
            f"type= {bold}{tok.type:<10s}{unbold} value= {bold}{tok.value:<10}{unbold} lineno={tok.lineno:<10} index={tok.index:<10} end={tok.end:<10}"
        )
    print("=" * 50)

    res = parser.parse(lexer.tokenize(text))
    print("result: ")
    pprint(res)
    print("=" * 50)

    print(f"locations: {parser.locations}")
    print(f"types: {parser.types}")
    print(f"names: {parser.names}")

    print("=" * 50)
    cfg.pprint()
