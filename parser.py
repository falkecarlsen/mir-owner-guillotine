import re
from pprint import pprint
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

cfg = ir.CFG()
curr_bb_id = -1
temp_stmts: List[ir.Statement] = []


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
        GOTO,
        ARROW,
        COLONTWICE,
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

    # TYPE = r'[a-zA-Z_][a-zA-Z0-9_] | \(\)*'
    PARAMS = r'\((\s*[a-zA-Z_][a-zA-Z0-9_]*\s*,)*\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)'
    EXPR = r'\((\s*[a-zA-Z_][a-zA-Z0-9_]*\s*,)*\s*[a-zA-Z_][a-zA-Z0-=9_]*\s*\)'

    ARROW = r'->'
    GOTO = r'goto'
    COLONTWICE = r'::'
    REFMUT = r'&mut'
    REF = r'&'
    DEREF = r'\*'
    TYPES = r'|'.join(
        [r'i32', r'u32', r'i64', r'u64', r'f32', r'f64', r'bool', r'char', r'str', r'String', r'\(\)', r'HashMap',
         r'Index'])
    # TYPES = r'i32'
    FUNCTIONS = r'|'.join(["index", "insert", "from", "get"])
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
    @_('LET LOCATION ":" TYPES ";"')
    def location_type_immut(self, p):
        self.types[p.LOCATION] = p.TYPE
        return p.LOCATION

    @_('LETMUT LOCATION ":" TYPES ";"')
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
            bb = ir.BasicBlock(curr_bb_id)
            # add temp statements to BasicBlock
            bb.add_statements(temp_stmts)
            cfg.add_bb(bb)

        except ValueError:
            print('ERROR: Invalid BB id', p.BB)
            exit(1)

        print(f'block{curr_bb_id} end')
        print(f"flushing {len(temp_stmts)} temp_stmts")
        temp_stmts.clear()
        return cfg

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
        # do something with p.statement
        print('stmtlist, statment', p.statement)
        return p.stmtlist

    @_('statement')
    def stmtlist(self, p):
        print('stmtlist', p.statement)
        return [p.statement]

    # assigment -> LOCATION = assigntype ;
    # assigntype -> LOCATION | constant | borrow | function
    # function -> TurboFish fun_goto | TurboFish
    # fun_goto -> ARROW fun_goto_location
    # fun_goto_location -> LOCATION | "[" cond_goto_loc "]"
    # cond_goto_loc -> cond_goto_loc "," cond_goto_loc | cond_goto_loc
    #                | NUMBER "_" types ":" LOCATION | OTHERWISE ":" LOCATION

    # types -> TYPE
    # constant -> CONSTANT NUMBER _ TYPE
    # borrow -> REF SOURCE | REFMUT SOURCE
    # source -> LOCATION | ( source )
    # source -> LOCATION | DEREF LOCATION

    # <GenericType<u32, String>>::index(move _1, move _2) #-> bb42;
    # HashMap::<u32, String>::get::<u32>(move _1, move _2)#-> bb42;
    # std::string::String
    # &std::string::String
    # &std::collections::HashMap<u32, std::string::String>

    # TurboFish -> GenericType :: Function
    # GenericType -> < GenericType Types > | Types
    # ConvertedType -> GenericType "as" GenericType
    # Types -> Type | ConvertedType
    # Function -> FUNCTIONS "(" PARAMS ")"
    # PARAMS -> PARAMS "," PARAM | PARAM
    # PARAM -> LOCATION | MODE LOCATION
    # MODE -> move

    # goto -> GOTO ARROW LOCATION ;




    #####
    # types
    @_('TYPES')
    def types(self, p):
        return p.types


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
        # create Statement
        stmt = ir.Statement(
            lhs_location=location_id,
            value_type=ir.ValueType.CONST,
            rhs_value=p.NUMBER,
            stmt_type=ir.StatementType.ASSIGN,
        )
        # add to temp stmts
        temp_stmts.append(stmt)

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

        # create Statement
        stmt = ir.Statement(
            lhs_location=borrower_id,
            value_type=ir.ValueType.BORROW,
            rhs_value=borrowee_id,
            stmt_type=ir.StatementType.ASSIGN,
        )
        # add to temp stmts
        temp_stmts.append(stmt)

        # return borrower_id, borrowee_id

    # deref location statement
    @_('DEREF LOCATION')
    def statement(self, p):
        print('deref statement', p.LOCATION)
        # do cfg deref location

        return p.LOCATION

    @_('RETURN ";"')
    def statement(self, p):
        print('statement return')
        # add statement to temp stmts
        temp_stmts.append(ir.Statement(
            stmt_type=ir.StatementType.RETURN,
        ))

        # todo: close off current bb, or handled by block end action code?

    # unreachable statement
    @_('UNREACHABLE ";"')
    def statement(self, p):
        print('statement unreachable')
        temp_stmts.append(ir.Statement(
            stmt_type=ir.StatementType.UNREACHABLE,
        ))


    # todo GOTO EBNF:

    # goto statment
    @_('GOTO LOCATION ";"')
    def goto(self, p):
        print('goto statement', p.LOCATION)
        # add statement to temp stmts
        temp_stmts.append(ir.Statement(
            stmt_type=ir.StatementType.GOTO,
            rhs_value=p.LOCATION,
        ))
        return p.LOCATION

    # <GenericType<u32, String>>::index(move _1, move _2) #-> bb42;
    # HashMap::<u32, String>::get::<u32>(move _1, move _2)#-> bb42;
    # std::string::String
    # &std::string::String
    # &std::collections::HashMap<u32, std::string::String>

    # TurboFish -> GenericType :: Function
    # GenericType -> < GenericType Types > | Types
    # ConvertedType -> GenericType "as" GenericType
    # Types -> Type | ConvertedType
    # Function -> FUNCTIONS "(" PARAMS ")"
    # PARAMS -> PARAMS "," PARAM | PARAM
    # PARAM -> LOCATION | MODE LOCATION
    # MODE -> move
    """
    # function call
    @_('LOCATION "=" types COLONTWICE FUNCTIONS "(" PARAMS ")" GOTO_FUNCTION ";"')
    def statement(self, p):
        print('function call statement', p.LOCATION, p.types, p.FUNCTIONS, p.PARAMS, p.GOTO)
        return p.types, p.FUNCTIONS, p.PARAMS


    @_('TYPES "<" types ">"')
    def types(self, p):
        return p.types
    """

def parse(mir_program):
    mir_lexer = MirLexer()
    mir_parser = MirParser()
    return mir_parser.parse(mir_lexer.tokenize(mir_program))


if __name__ == '__main__':
    bold = '\033[1m'
    unbold = '\033[0m'
    header = "=" * 80

    text = open('mir-input-grammar/pass/test.mir', 'r').read()

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
    # lpprint(res)

    print(f"{header}\ncfg pprint: ")
    cfg.pprint()
