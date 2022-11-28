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
        CONSTANT,
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
        [
            r'i32',
            r'u32',
            r'i64',
            r'u64',
            r'f32',
            r'f64',
            r'bool',
            r'char',
            r'str',
            r'String',
            r'\(\)',
            r'HashMap',
            r'Index',
        ]
    )
    # TYPES = r'i32'
    FUNCTIONS = r'|'.join(["index", "insert", "from", "get"])
    CONSTANT = r'const'
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
        self.stmt_id = -1
        self.curr_stmt: ir.Statement = None

    def get_curr_stmt_id(self):
        if self.stmt_id == -1:
            raise Exception("stmt_id not defined when used")
        else:
            # return curr and reset to -1
            ret = self.stmt_id
            self.stmt_id = -1
            return ret

    @staticmethod
    def get_loc_int(loc):
        # get int from location string
        return int(re.sub(r'\D', '', loc))

    def add_curr_stmt_and_reset(self):
        global temp_stmts
        temp_stmts.append(self.curr_stmt)
        self.curr_stmt = ir.Statement()

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
    @_('BB block_start ":" "{" stmtlist "}"')
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
        print("block start, setup curr_stmt")
        # setup curr_stmt
        self.curr_stmt = ir.Statement()

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

    # statement -> LOCATION = stmttype ; | GOTO ARROW LOCATION ; | UNREACHABLE ; | RETURN ;
    # stmttype -> LOCATION | constant | borrow | function
    # ctrlflow -> goto | unreachable | return
    # function -> TurboFish fun_goto | TurboFish
    # fun_goto -> ARROW fun_goto_location
    # fun_goto_location -> LOCATION | "[" cond_goto_loc "]"
    # cond_goto_loc -> cond_goto_loc "," cond_goto_loc | cond_goto_loc
    #                | NUMBER "_" types ":" LOCATION | OTHERWISE ":" LOCATION

    # types -> TYPE
    # constant -> CONSTANT NUMBER _ TYPE
    # borrow -> REF source | REFMUT source
    # source -> ( source ) | LOCATION | DEREF LOCATION

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

    #####

    # statement -> LOCATION = stmttype | goto | unreachable | return;
    @_('LOCATION "=" stmttype ";"')
    def statement(self, p):
        curr_stmt_id = self.get_loc_int(p.LOCATION)
        last_stmt = self.curr_stmt
        # if last stmt is an assignment, then we need to assign the curr_stmt_id
        match last_stmt.stmt_type:
            case ir.StatementType.ASSIGN:
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
        temp_stmts.append(self.curr_stmt)
        self.curr_stmt = ir.Statement()

        print('statement', p.LOCATION, p.stmttype)
        return p.stmttype

    # statement -> GOTO ARROW BB ;
    @_('GOTO ARROW BB ";"')
    def statement(self, p):
        print('goto', p.BB)
        self.curr_stmt.stmt_type = ir.StatementType.GOTO
        self.curr_stmt.bb_target = self.get_loc_int(p.BB)
        self.add_curr_stmt_and_reset()

    # statement -> UNREACHABLE ;
    @_('UNREACHABLE ";"')
    def statement(self, p):
        print('unreachable', p.UNREACHABLE)
        self.curr_stmt.stmt_type = ir.StatementType.UNREACHABLE
        self.add_curr_stmt_and_reset()

    # statement -> RETURN ;
    @_('RETURN ";"')
    def statement(self, p):
        print('return', p.RETURN)
        self.curr_stmt.stmt_type = ir.StatementType.RETURN
        self.add_curr_stmt_and_reset()

    # stmttype -> LOCATION | constant | borrow | goto | unreachable | return | function
    @_('LOCATION')
    def stmttype(self, p):
        print('stmttype location', p.LOCATION)
        # create statement
        self.curr_stmt.stmt_type = ir.StatementType.ASSIGN
        self.curr_stmt.rhs_location = self.get_loc_int(p.LOCATION)

    @_('constant')
    def stmttype(self, p):
        print('stmttype', p.constant)
        return p.constant

    # constant -> CONSTANT NUMBER _ TYPE
    @_('CONSTANT NUMBER "_" TYPES')
    def constant(self, p):
        print('constant', p.CONSTANT)
        self.curr_stmt.stmt_type = ir.StatementType.ASSIGN
        self.curr_stmt.rhs_value = p.NUMBER
        self.curr_stmt.value_type = p.TYPES
        self.curr_stmt.rhs_value = ir.ValueType.CONST

    @_('borrow')
    def stmttype(self, p):
        print('stmttype', p.borrow)
        self.curr_stmt.stmt_type = ir.StatementType.ASSIGN
        return p.borrow

    # borrow -> REF source | REFMUT source
    @_('REF source')
    def borrow(self, p):
        print('borrow', p.source)
        self.curr_stmt.value_type = ir.ValueType.BORROW
        self.curr_stmt.mutability = False
        return p.source

    @_('REFMUT source')
    def borrow(self, p):
        print('borrow', p.source)
        self.curr_stmt.value_type = ir.ValueType.BORROW
        self.curr_stmt.mutability = True
        return p.source

    # source -> ( source ) | LOCATION | DEREF LOCATION
    @_('"(" source ")"')
    def source(self, p):
        print('source parens', p.source)
        return p.source

    @_('LOCATION')
    def source(self, p):
        print('source location', p.LOCATION)
        self.curr_stmt.rhs_location = self.get_loc_int(p.LOCATION)
        return p.LOCATION

    @_('DEREF LOCATION')
    def source(self, p):
        print('source deref location', p.LOCATION)
        self.curr_stmt.rhs_location = self.get_loc_int(p.LOCATION)
        return p.LOCATION

    # function -> TurboFish fun_goto | TurboFish
    # fun_goto -> ARROW fun_goto_location
    # fun_goto_location -> LOCATION | "[" cond_goto_loc "]"
    # cond_goto_loc -> cond_goto_loc "," cond_goto_loc | cond_goto_loc
    #                | NUMBER "_" types ":" LOCATION | OTHERWISE ":" LOCATION

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
