from enum import IntEnum


class PosTag(IntEnum):
    ADJ = 84  # adjective, e.g.big, old, green, incomprehensible, first
    ADP = 85  # adposition, e.g. in, to, during
    ADV = 86  # adverb, e.g.very, tomorrow, down, where, there
    AUX = 87  # auxiliary, e.g. is, has(done), will(do), should(do)
    CONJ = 88  # conjunction, e.g. and, or, but
    CCONJ = 89  # coordinating conjunction, e.g. and, or, but
    DET = 90  # determiner, e.g.a, an, the
    INTJ = 91  # interjection, e.g.psst, ouch, bravo, hello
    NOUN = 92  # noun, e.g.girl, cat, tree, air, beauty
    NUM = 93  # numeral, e.g.  1, 2017, one, seventy - seven, IV, MMXIV
    PART = 94  # particle, e.g. ’s, not,
    PRON = 95  # pronoun, e.g I, you, he, she, myself, themselves, somebody
    PROPN = 96  # proper noun, e.g.Mary, John, London, NATO, HBO
    PUNCT = 97  # punctuation, e.g.., (,), ?
    SCONJ = 98  # subordinating conjunction, e.g. if, while , that
    SYM = 99  # symbol, e.g. $, %, §, ©, +, −, ×, ÷, =,:), 😝
    VERB = 100  # verb, e.g.run, runs, running, eat, ate, eating
    X = 101  # other, e.g.sfpksdpsxmsa
    SPACE = 103  # space, e.g.
