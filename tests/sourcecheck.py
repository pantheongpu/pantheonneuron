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

_SEPARATORS = (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT)


def code_only(source: str) -> str:
    """Return ``source`` with comments and docstrings removed."""
    kept: typing.List[str] = []
    previous: typing.Optional[int] = None
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        # A bare string at the start of a logical line is a docstring, not
        # an operand.
        if token.type == tokenize.STRING and previous in (None, *_SEPARATORS):
            continue
        kept.append(token.string)
        previous = token.type
    return " ".join(kept)


def function_code(function) -> str:
    """``code_only`` over a function's own source.

    Tokenising the whole function and slicing afterwards is deliberate:
    slicing the raw text first leaves a fragment whose indentation
    ``tokenize`` refuses.
    """
    import inspect

    return code_only(inspect.getsource(function))
