"""Reading a kernel's source as code rather than as prose.

Several checks in this suite have to be textual: they assert something about
a kernel that only a Neuron device could otherwise prove -- that a loop is
not unrolled, that a buffer is not reallocated per pass, that a dispatch
does not index weights by token.

Those checks are worthless if they read comments. Twice now a check has
passed on the very comment explaining the defect it was written to catch,
because the comment quotes the code it warns about. Strip comments and
docstrings first, and the check reads what runs.

Tokens are joined with single spaces, so match against spaced forms:
``"expert_weights [ chosen ]"``, not ``"expert_weights[chosen]"``.
"""

import io
import tokenize
import typing

# A docstring is the first statement of a module, class or function, so it
# follows a *logical* line end or an indent.
#
# NL is deliberately not a separator: it marks a non-logical newline, the
# kind inside brackets, and treating it as a statement boundary made every
# dict key on a continuation line look like a docstring and vanish.
#
#     {"key": 1,        ->    { : 1 ,
#      "other": 2}             : 2 }
#
# Which silently broke every check written against a dict key -- an `in`
# assertion could not pass, and worse, a `not in` assertion passed for the
# wrong reason. Found 2026-09-10 when an assertion about "implied_tflops"
# failed against source that demonstrably contained it.
#
# But dropping NL from the separator set was not enough on its own, and the
# fix for one bug was the cause of the next. A module docstring preceded by
# a shebang or a licence comment follows COMMENT then NL, so with NL no
# longer a separator it stopped looking like a docstring and survived into
# the filtered output. `tools/compare_matmul_paths.py` starts with a
# shebang, and an ordering assertion over its filtered source matched text
# inside the docstring instead of the code -- which is precisely the class
# of accident this module exists to prevent, produced by this module.
#
# So the question is not "what was the previous token" but "have we seen a
# real token since the last statement boundary". COMMENT and NL are
# non-logical: they do not open a statement and they do not close one.
_SEPARATORS = (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)

# Tokens that neither begin a statement nor end one, so a docstring can
# still follow them.
_TRANSPARENT = (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING)


def code_only(source: str) -> str:
    """Return ``source`` with comments and docstrings removed."""
    kept: typing.List[str] = []
    # True while the next real token would be the first of a statement,
    # which is the only position a string is a docstring rather than an
    # operand.
    at_statement_start = True
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type in _TRANSPARENT:
            kept.append(token.string)
            continue
        if token.type == tokenize.STRING and at_statement_start:
            continue
        kept.append(token.string)
        at_statement_start = token.type in _SEPARATORS
    return " ".join(kept)


def flat(source: str) -> str:
    """Collapse all whitespace, so a match cannot depend on line wrapping.

    ``code_only`` joins tokens with single spaces but keeps the newlines
    the tokenizer emits, so a call split across lines does not match the
    same call written on one. Four assertions in this suite have broken
    or passed on that difference -- a check about how code is formatted
    rather than what it does, which is the thirteenth entry in
    docs/checks_that_pass_by_accident.md.

    Use this whenever the assertion is about a construct rather than a
    single identifier.
    """
    return " ".join(source.split())


def flat_function_code(function) -> str:
    """``function_code`` with the wrapping flattened out."""
    return flat(function_code(function))


def function_code(function) -> str:
    """``code_only`` over a function's own source.

    Tokenising the whole function and slicing afterwards is deliberate:
    slicing the raw text first leaves a fragment whose indentation
    ``tokenize`` refuses.
    """
    import inspect

    return code_only(inspect.getsource(function))
