from ir import *
from parser import *


def ud_cfg_borrow_analysis(cfg: CFG):
    # compute UD chains
    cfg.compute_ud_chains()
    # check if borrow is valid
    for c in cfg.ud_chains:
        # check if borrow is valid
        if c[0][1] == c[1][1]:
            print(f"borrow is valid: {c}")
        else:
            print(f"borrow is invalid: {c}")
    # compute flag matrix
    cfg.compute_flag_matrix()
    # print flag matrix
    cfg.pprint()


if __name__ == '__main__':
    # lex and parse input
    text = open('mir-input-grammar/pass/test.mir', 'r').read()
    parser = MirParser()
    parser.parse()
    # build cfg
    cfg = CFG(parser.locations, parser.types)
    cfg.build()
    # compute borrow analysis
    ud_cfg_borrow_analysis(cfg)
