import re
from pprint import pprint

from sly import Lexer, Parser

import data

"""
Rust MIR simplified grammar for borrow-checking and CFG generation 

<mir> ::= <function>*
<function> ::= FN <name> TO "(" <param> ")" "{" <block>+ "}"
<param> ::= (<name> ("," <name>)*)+
<block> ::= <name> ":" "{" <statement>* "}"
?<statements> ::= <statement> <statements> | <statement>
<statement> ::= <assignment> | <borrow> | <mut_borrow> | <return> | <call>
<assignment> ::= <name> "=" <expr> ";"
<borrow> ::= <name> "=" "&" <expr> ";"
<mut_borrow> ::= <name> "=" "&mut" <expr> ";"
<return> ::= RETURN ";"
<call> ::= <name> "(" <expr> ")" TO <expr> ;"
<expr> ::= <name> | <literal>
<name> ::= [a-zA-Z_][a-zA-Z0-9_]*
<literal> ::= [0-9]+
"""

cfg = data.CFG()


class CalcLexer(Lexer):
    # Tokens
    literals = {'+', '-', '*', '/', '(', ')', '=', ':', ';', ',', '[', ']', '{', '}', '_'}
    tokens = {LOCATION, FN, NAME, LET, LETMUT, REF, REFMUT, TYPE, BB, PARAMS, EXPR, RETURN, STMT, DEREF,
              TYPES, FUNCTIONS, CONSTANTS, MOVE, STATEMENT, NUMBER}
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


class CalcParser(Parser):
    tokens = CalcLexer.tokens
    debugfile = 'parser.out'
    start = 'block'

    precedence = (
        ('left', STMT),
        ('left', LOCATION),
    )

    def __init__(self):
        self.locations = {}
        self.types = {}
        self.names = {}

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
        try:
            bb_id = int(p.BB[-1])
            # create BasicBlock and add to CFG
            bb = data.BasicBlock(bb_id)
            cfg.add_node(bb)
        except ValueError:
            print('ERROR: Invalid BB id', p.BB)
            exit(1)

        print('block', p.BB, p.stmtlist)
        return p.stmtlist

    # stmtlist
    @_('stmtlist statement')
    def stmtlist(self, p):
        # do something with p.statement
        print('statement', p.statement)
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

    # assignment constant to location
    @_('LOCATION "=" CONSTANTS NUMBER "_" TYPES ";"')
    def statement(self, p):
        print('const statement', p.LOCATION, p.NUMBER, p.TYPES)
        location_id = int(p.LOCATION[1:])
        self.locations[location_id] = p.NUMBER
        # maybe do check if fn-defined type corresponds with currently seen type
        self.types[p.LOCATION] = p.TYPES
        return p.LOCATION, p.NUMBER, p.TYPES

    # types
    @_('TYPES')
    def types(self, p):
        print("type: ", p.TYPES)
        return p.TYPES

    @_('LOCATION "=" REF EXPR ";"')
    def statement(self, p):
        print('statement', p.LOCATION, p.expr)
        return p.expr

    @_('LOCATION "=" REFMUT EXPR ";"')
    def statement(self, p):
        print('statement', p.LOCATION, p.expr)

    @_('RETURN ";"')
    def statement(self, p):
        print('statement', p.RETURN)
        return 42
        # close block scope

    # constant expression
    @_('NUMBER "_" TYPE')
    def expr(self, p):
        print('expr', p.NUMBER, p.TYPE)
        return p.NUMBER


if __name__ == '__main__':
    bold = '\033[1m'
    unbold = '\033[0m'

    lexer = CalcLexer()
    parser = CalcParser()
    text = open('mir-input-grammar/pass/test.mir', 'r').read()

    for tok in lexer.tokenize(text):
        print(
            f"type= {bold}{tok.type:<10s}{unbold} value= {bold}{tok.value:<10}{unbold} lineno={tok.lineno:<10} index={tok.index:<10} end={tok.end:<10}")
    print("=" * 50)

    res = parser.parse(lexer.tokenize(text))
    print("result: ")
    print(res)
    print("=" * 50)

    print(f"locations: {parser.locations}")
    print(f"types: {parser.types}")
    print(f"names: {parser.names}")

    print("=" * 50)
    cfg.pprint()
