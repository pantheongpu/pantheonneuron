"""The transformer family: ten workloads that had no test of their own.

`transformer_ops`, `llm_inference`, `inference_mix`, `encoders` and
`transformer_compute` are the largest block of code in this suite and were
the only kernel modules no test imported. That gap hid a real defect:
`omni_virus` called `_read_back` without defining or importing it, so a
NameError fired at the end of `run()` -- after a full-duration stress run on
a rented device -- and turned a completed run into a FAIL row. Nothing off
hardware could have caught it, because nothing off hardware imported the
module.

So this file covers what can be checked without a Neuron device: the FLOP
arithmetic, the patch geometry, the serving interleave, the readback and
verification helpers, and the import-level integrity of every module the
orchestrator dispatches to. The kernels' device behaviour still needs
hardware; their arithmetic no longer does.
"""

import importlib
import math

import pytest

import sourcecheck
from kernels import (encoders, inference_mix, llm_inference, omni_virus,
                     registry, transformer_compute, transformer_ops)


PROBLEMS = {w.name: w.problem for w in registry.WORKLOADS}


# -- module integrity --------------------------------------------------------
#
# The omni_virus bug was a name that did not exist, in a module nothing
# imported. These tests are cheap and they close that class of defect for
# every kernel module at once.

KERNEL_MODULES = (
    "allocation_fragmentation", "collectives", "cores", "encoders",
    "graph_replay", "inference_mix", "llm_inference", "memory_agg",
    "memory_read", "memory_write", "nki_backend", "omni_virus",
    "pcie_bandwidth", "profiler", "pulse_virus", "registry", "tensor_virus",
    "tiling", "transformer_compute", "transformer_ops",
)


@pytest.mark.parametrize("name", KERNEL_MODULES)
def test_every_kernel_module_imports(name):
    """A kernel that cannot be imported cannot be run."""
    assert importlib.import_module(f"kernels.{name}") is not None


def _undefined_globals(source: str) -> set:
    """Names loaded in ``source`` that are neither defined, imported nor builtin.

    Compiling to bytecode does not catch these: a name that does not exist
    raises only when its line executes, which for these kernels is at the
    end of a timed run on hardware that bills by the hour. That is exactly
    how ``omni_virus`` shipped a call to a ``_read_back`` it did not have.
    """
    import ast
    import builtins

    tree = ast.parse(source)
    # Module-level dunders are real at runtime and absent from builtins.
    defined = set(dir(builtins)) | {
        "__file__", "__name__", "__doc__", "__package__", "__spec__",
        "__loader__", "__builtins__",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            defined.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spec = node.args
                for arg in (spec.args + spec.posonlyargs + spec.kwonlyargs):
                    defined.add(arg.arg)
                for arg in (spec.vararg, spec.kwarg):
                    if arg is not None:
                        defined.add(arg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)

    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - defined


def test_the_undefined_name_check_catches_the_omni_virus_defect():
    """The checker must fail the code it was written for, or it proves nothing.

    This is the shape of the bug verbatim: a helper called but never defined
    or imported, inside a function that only runs on hardware.
    """
    guilty = """
from . import transformer_ops

def run(problem, duration):
    return {"warning": transformer_ops.verify_output_is_a_number(
        _read_back(sink), "chain output")}
"""
    assert "_read_back" in _undefined_globals(guilty)

    innocent = guilty.replace("_read_back(sink)",
                              "transformer_ops.read_back(None)")
    assert not _undefined_globals(innocent)


@pytest.mark.parametrize("name", KERNEL_MODULES)
def test_no_kernel_module_references_an_undefined_global(name):
    """Catch the omni_virus defect anywhere it recurs."""
    module = importlib.import_module(f"kernels.{name}")
    with open(module.__file__, encoding="utf-8") as handle:
        missing = _undefined_globals(handle.read())
    assert not missing, f"{name} uses undefined name(s): {sorted(missing)}"


def test_read_back_has_exactly_one_definition():
    """It was copied into four modules and missing from a fifth."""
    import pathlib

    root = pathlib.Path(transformer_ops.__file__).parent
    definitions = [
        path.name for path in root.glob("*.py")
        if "def read_back(" in path.read_text(encoding="utf-8")
        or "def _read_back(" in path.read_text(encoding="utf-8")
    ]
    assert definitions == ["transformer_ops.py"], definitions


# -- readback and verification -----------------------------------------------

def test_read_back_returns_none_for_none():
    assert transformer_ops.read_back(None) is None


def test_read_back_returns_none_when_materialisation_fails():
    """A tensor that raises on read leaves the run unverified, not failed."""

    class Unreadable:
        def reshape(self, *_):
            raise RuntimeError("device is gone")

    assert transformer_ops.read_back(Unreadable()) is None


def test_read_back_takes_the_first_element():
    class Fake:
        def __init__(self, values):
            self.values = values

        def reshape(self, *_):
            return self.values

    assert transformer_ops.read_back(Fake([2.5, 9.0])) == 2.5


def test_a_real_number_verifies():
    assert transformer_ops.verify_output_is_a_number(1.5) is None
    assert transformer_ops.verify_output_is_a_number(0.0) is None
    assert transformer_ops.verify_output_is_a_number(-3.0) is None


def test_infinity_verifies_on_purpose():
    """All-ones weights with no normalisation saturate bf16, and that is fine.

    The check exists to catch a graph that produced nothing readable, not to
    police numerics. An inf still proves the graph ran, so failing it here
    would fail every deep run for doing exactly what it was built to do.
    """
    assert transformer_ops.verify_output_is_a_number(math.inf) is None
    assert transformer_ops.verify_output_is_a_number(-math.inf) is None


def test_nan_does_not_verify():
    message = transformer_ops.verify_output_is_a_number(math.nan, "loss")
    assert message is not None
    assert "loss" in message and "NaN" in message


def test_an_unreadable_output_says_so_rather_than_passing():
    message = transformer_ops.verify_output_is_a_number(None, "block output")
    assert message is not None
    assert "block output" in message and "unverified" in message


# -- FLOP arithmetic ---------------------------------------------------------

def test_block_flops_matches_the_arithmetic_it_documents():
    """Four projections, two attention matmuls, a 4x MLP; 2 FLOPs per MAC."""
    hidden, seq, batch = 8, 4, 2
    projections = 4 * (2 * batch * seq * hidden * hidden)
    attention = 2 * (2 * batch * seq * seq * hidden)
    mlp = 2 * (2 * batch * seq * hidden * 4 * hidden)
    assert transformer_ops.block_flops(hidden, seq, batch) == (
        projections + attention + mlp
    )


def test_block_flops_is_linear_in_batch():
    single = transformer_ops.block_flops(512, 128, 1)
    assert transformer_ops.block_flops(512, 128, 4) == 4 * single


def test_the_attention_term_is_quadratic_in_sequence():
    """Doubling the sequence must more than double the cost.

    The projections and MLP are linear in sequence and the attention term is
    quadratic, so the total grows faster than linearly. That is the property
    that makes prefill compute-bound, and a refactor that dropped the
    quadratic term would leave a plausible-looking number behind.
    """
    hidden = 512
    short = transformer_ops.block_flops(hidden, 128)
    long = transformer_ops.block_flops(hidden, 256)
    assert long > 2 * short


def test_decode_step_flops_is_linear_in_context():
    """Decode attends to a cache, so its cost is linear where prefill's is not."""
    hidden = 512
    base = transformer_ops.decode_step_flops(hidden, 0)
    step = (transformer_ops.decode_step_flops(hidden, 1024)
            - transformer_ops.decode_step_flops(hidden, 512))
    doubled = (transformer_ops.decode_step_flops(hidden, 2048)
               - transformer_ops.decode_step_flops(hidden, 1536))
    assert base > 0
    assert step == doubled


def test_prefill_and_decode_differ_by_orders_of_magnitude():
    """The registry's stated reason for keeping them separate workloads.

    The README claims decode is "roughly three orders of magnitude less
    arithmetic per step" than prefill. Both figures come from these two
    functions applied to the pinned problems, so the claim is checkable --
    and if a change ever made the two converge, that would be the signal
    that they had become one measurement wearing two names.
    """
    prefill = PROBLEMS["llm_prefill"]
    decode = PROBLEMS["llm_decode"]

    per_prefill = transformer_ops.block_flops(
        prefill["hidden"], prefill["prompt"], prefill["batch"])
    per_decode = transformer_ops.decode_step_flops(
        decode["hidden"], decode["context"], decode["batch"])

    assert per_decode > 0
    ratio = per_prefill / per_decode
    assert 100 < ratio < 100_000, ratio


def test_flop_counts_are_whole_numbers():
    """They are counts. A float here would leak into implied_tflops."""
    assert isinstance(transformer_ops.block_flops(128, 128), int)
    assert isinstance(transformer_ops.decode_step_flops(128, 128), int)


# -- vision encoder geometry -------------------------------------------------

def test_patch_plan_for_the_pinned_problem():
    plan = encoders.patch_plan(PROBLEMS["vision_encoder"])
    assert plan["per_side"] == 16          # 224 / 14
    assert plan["patches"] == 256          # 16 x 16
    assert plan["patch_dim"] == 14 * 14 * 3
    assert plan["batch"] == 64


def test_patch_plan_rejects_a_resolution_the_patch_does_not_divide():
    """An indivisible resolution would silently drop a strip of every image."""
    with pytest.raises(ValueError, match="not divisible"):
        encoders.patch_plan({"resolution": 225, "patch": 14, "batch": 1})


@pytest.mark.parametrize("problem", [
    {"resolution": 0, "patch": 14, "batch": 1},
    {"resolution": 224, "patch": 0, "batch": 1},
    {"resolution": -224, "patch": 14, "batch": 1},
])
def test_patch_plan_rejects_invalid_geometry(problem):
    with pytest.raises(ValueError, match="invalid geometry"):
        encoders.patch_plan(problem)


def test_patch_plan_rejects_a_non_positive_batch():
    with pytest.raises(ValueError, match="batch must be positive"):
        encoders.patch_plan({"resolution": 224, "patch": 14, "batch": 0})


def test_patches_scale_with_the_square_of_the_side():
    coarse = encoders.patch_plan({"resolution": 224, "patch": 28, "batch": 1})
    fine = encoders.patch_plan({"resolution": 224, "patch": 14, "batch": 1})
    assert fine["patches"] == 4 * coarse["patches"]


# -- serving interleave ------------------------------------------------------

def test_the_pinned_serving_ratio_is_exact():
    """0.2 is 1/5, so the deterministic interleave reproduces it exactly."""
    ratio = PROBLEMS["serving_mix"]["prefill_ratio"]
    assert inference_mix.interleave_period(ratio) == 5

    requests = 1000
    prefills = sum(1 for n in range(requests) if n % 5 == 0)
    assert prefills / requests == pytest.approx(ratio)


@pytest.mark.parametrize("ratio,period", [
    (1.0, 1),      # every request is a prefill
    (0.5, 2),
    (0.25, 4),
    (0.2, 5),
    (0.0, 0),      # no prefills at all
])
def test_interleave_period_for_representative_ratios(ratio, period):
    assert inference_mix.interleave_period(ratio) == period


def test_a_ratio_that_is_not_one_over_n_quantises():
    """Documented behaviour, asserted so it stays documented.

    A period is a whole number of requests, so 0.6 becomes period 2 and the
    run actually serves 0.5. The row reports observed_prefill_ratio beside
    the requested one precisely because of this.
    """
    assert inference_mix.interleave_period(0.6) == 2


@pytest.mark.parametrize("ratio", [-0.1, 1.5])
def test_interleave_period_rejects_a_ratio_outside_the_unit_interval(ratio):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        inference_mix.interleave_period(ratio)


# -- dispatch integrity ------------------------------------------------------
#
# The orchestrator dispatches by name to a function in one of these modules
# and reads one key out of the dict it returns. Neither half is checked by
# the type system, and a rename on either side fails only on hardware.

FAMILY_ENTRY_POINTS = {
    "llm_prefill": (llm_inference, "run_prefill", "prompt_tokens_per_s"),
    "llm_decode": (llm_inference, "run_decode", "tokens_per_s"),
    "kv_cache_churn": (llm_inference, "run_cache_churn", "cache_updates_per_s"),
    "transformer_virus": (transformer_compute, "run_virus", "analytic_tflops"),
    "transformer_train_step": (transformer_compute, "run_train_step",
                               "train_steps_per_s"),
    "fused_attention": (inference_mix, "run_fused_attention",
                        "attention_tiles_per_s"),
    "quantized_gemm": (inference_mix, "run_quantized_gemm",
                       "quantized_ops_per_s"),
    "moe_router": (inference_mix, "run_moe_router", "routed_tokens_per_s"),
    "speculative_decode": (inference_mix, "run_speculative_decode",
                           "verified_tokens_per_s"),
    "serving_mix": (inference_mix, "run_serving_mix", "requests_per_s"),
    "rag_embedding": (encoders, "run_rag_embedding", "embedding_vectors_per_s"),
    "vision_encoder": (encoders, "run_vision_encoder", "image_tiles_per_s"),
    "omni_virus": (omni_virus, "run", "analytic_tflops"),
}


@pytest.mark.parametrize("name", sorted(FAMILY_ENTRY_POINTS))
def test_each_workload_has_the_entry_point_the_orchestrator_calls(name):
    module, function, _ = FAMILY_ENTRY_POINTS[name]
    assert callable(getattr(module, function, None)), f"{name} -> {function}"


@pytest.mark.parametrize("name", sorted(FAMILY_ENTRY_POINTS))
def test_the_orchestrator_reads_the_key_this_kernel_returns(name):
    """The Score key must appear in the kernel that is supposed to produce it.

    A textual check, because calling the kernel needs a device. It still
    catches the failure that matters: renaming the key in the kernel and not
    in the dispatch, which raises KeyError at the end of a timed run.
    """
    import inspect

    module, _, key = FAMILY_ENTRY_POINTS[name]
    assert f'"{key}"' in inspect.getsource(module), f"{name}: {key}"

    dispatch = inspect.getsource(
        importlib.import_module("pantheon_neuron")._execute)
    assert f'result["{key}"]' in dispatch, f"{name}: {key} not read by dispatch"


@pytest.mark.parametrize("name", sorted(FAMILY_ENTRY_POINTS))
def test_every_family_workload_is_declared_implemented(name):
    pantheon_neuron = importlib.import_module("pantheon_neuron")
    assert name in pantheon_neuron.IMPLEMENTED
    assert name in PROBLEMS


@pytest.mark.parametrize("name", sorted(FAMILY_ENTRY_POINTS))
def test_every_family_workload_verifies_that_its_graph_ran(name):
    """Each kernel must judge its output *and* let the verdict reach the row.

    The paired form is the point. Calling `verify_output_is_a_number` and
    storing the message under "warning" is what these kernels used to do,
    and it left three workloads reporting PASS with a published Score while
    their output was NaN: the check fired and nothing acted on it.
    `output_check` returns the message and `score_invalid` together, so the
    orchestrator cannot fail to notice.

    Which paired helper is used is not the invariant, and asserting on
    `output_check(` by name was. fused_attention now uses
    `attention_check`, which is stricter -- its inputs are all ones, so
    softmax is exactly uniform and the answer is 1.0 rather than merely
    finite -- and the old assertion failed it for being better. A check
    written against one implementation of a rule enforces the
    implementation, not the rule.
    """
    module, function, _ = FAMILY_ENTRY_POINTS[name]
    source = sourcecheck.function_code(getattr(module, function))

    paired = [helper for helper in dir(transformer_ops)
              if helper.endswith("_check")]
    assert paired, "no paired helpers found -- the naming convention moved"
    # Either a shared helper directly, or a local one that wraps it --
    # transformer_train_step pairs the loss verdict with a second verdict
    # about whether the optimiser moved anything, and the requirement is
    # that a paired check runs, not that it is called by a particular
    # name from a particular module.
    called = (any(f"transformer_ops . {helper} (" in source
                  for helper in paired)
              or "_check (" in source)
    assert called, f"{name} uses no paired check helper"
    assert "read_back" in source, name
    assert '"warning" : transformer_ops . verify_' not in source, (
        f"{name} stores the message without the verdict"
    )


def test_output_check_pairs_the_message_with_the_verdict():
    assert transformer_ops.output_check(1.0) == {
        "warning": None, "score_invalid": False}

    nan = transformer_ops.output_check(math.nan, "loss")
    assert nan["score_invalid"] is True
    assert "loss" in nan["warning"]

    unread = transformer_ops.output_check(None, "block output")
    assert unread["score_invalid"] is True

    # inf is deliberately fine, so it must not invalidate a Score.
    assert transformer_ops.output_check(math.inf)["score_invalid"] is False


def test_weights_are_scaled_by_fan_in():
    """1/fan_in, not 1 and not a single 1/hidden.

    Ones grow activations by `hidden` per layer. A single 1/hidden fixes
    the square projections but leaves `w2` -- which contracts over 4*hidden
    -- four times too large, and trn1.2xlarge measured llm_prefill as NaN
    with exactly that scaling.
    """
    code = sourcecheck.function_code(transformer_ops.weights)
    assert "1.0 / rows" in code
    assert "torch . ones (" not in code

    hidden = PROBLEMS["llm_prefill"]["hidden"]
    # w1 contracts over hidden, w2 over 4*hidden. Under one shared 1/hidden
    # the MLP would multiply by four every block.
    assert (4 * hidden) * (1.0 / hidden) == 4.0
    assert (4 * hidden) * (1.0 / (4 * hidden)) == 1.0


def test_the_block_normalises_its_branch_inputs():
    """Scaling alone cannot keep a deep stack alive; normalisation can.

    Attention scores are quadratic in activation magnitude -- a dot product
    over head_dim of values that are themselves growing -- so a residual
    stream that grows multiplicatively overflows fp32 inside softmax long
    before bf16 runs out. Pre-norm makes every matmul input unit scale, so
    the stream grows additively instead.
    """
    code = sourcecheck.function_code(transformer_ops.block)
    assert code.count("rms_norm (") == 2, "pre-norm on both branches"

    # Pre-norm, not post-norm: the residual is taken before normalising.
    assert code.index("residual = hidden_states") < code.index("rms_norm (")


def test_normalisation_is_what_keeps_the_pinned_depth_in_range():
    """The arithmetic, so the claim is checkable rather than asserted."""
    hidden = PROBLEMS["llm_prefill"]["hidden"]
    layers = PROBLEMS["llm_prefill"]["layers"]
    head_dim = PROBLEMS["fused_attention"]["head_dim"]
    fp32_max = 3.4e38

    def overflows_at(per_block_gain):
        magnitude = 1.0
        for layer in range(1, layers + 1):
            magnitude *= per_block_gain
            if head_dim ** 0.5 * magnitude ** 2 > fp32_max:
                return layer
        return None

    # Unscaled ones, then one shared 1/hidden, then fan_in scaling.
    assert overflows_at(hidden) is not None
    assert overflows_at(10.0) == 19, "what shipped, and what NaN'd"
    # Even correct fan_in scaling only just survives, which is why the fix
    # is normalisation rather than a better constant.
    assert overflows_at(4.0) in (layers, None)

    # Pre-norm: the stream grows additively, roughly 2 per block.
    additive = 2 * layers
    assert head_dim ** 0.5 * additive ** 2 < fp32_max / 1e30


# -- MoE dispatch ------------------------------------------------------------
#
# The 2026-09-08 full-coverage run refused to compile this workload at all:
# `expert_weights[chosen]` gathered a [hidden, hidden] weight matrix per
# token, materialising 137 GB against a 0.3 GB weight table. Fixing that
# exposed a second defect underneath -- the routing itself was degenerate.

def test_expert_capacity_divides_the_routed_slots():
    problem = PROBLEMS["moe_router"]
    capacity = inference_mix.expert_capacity(
        problem["tokens"], problem["top_k"], problem["experts"])
    assert capacity * problem["experts"] == problem["tokens"] * problem["top_k"]


def test_capacity_is_never_zero():
    """A degenerate problem must still compile to something runnable."""
    assert inference_mix.expert_capacity(1, 1, 64) == 1


def test_the_pinned_routing_uses_every_expert_exactly_once_over():
    """Balanced by construction, so no expert idles and nothing is dropped."""
    problem = PROBLEMS["moe_router"]
    capacity = inference_mix.expert_capacity(
        problem["tokens"], problem["top_k"], problem["experts"])
    balance = inference_mix.routing_balance(
        problem["tokens"], problem["experts"], problem["top_k"])

    assert len(balance) == problem["experts"]
    assert set(balance.values()) == {capacity}, balance
    assert sum(balance.values()) == problem["tokens"] * problem["top_k"]


def test_uniform_inputs_would_have_idled_six_of_eight_experts():
    """What the previous version measured, as arithmetic.

    All-ones activations through an all-ones gate give every token
    identical logits, so top-k picks the same two experts for all of them.
    Six experts receive nothing and three quarters of the routed slots hit
    the capacity limit and are dropped -- while the Score counts every
    token as routed.
    """
    problem = PROBLEMS["moe_router"]
    experts, tokens, top_k = (problem["experts"], problem["tokens"],
                              problem["top_k"])
    capacity = inference_mix.expert_capacity(tokens, top_k, experts)

    # Ties in topk resolve to the lowest indices, so every token picks 0..top_k-1.
    degenerate = {e: (tokens if e < top_k else 0) for e in range(experts)}
    idle = [e for e, n in degenerate.items() if n == 0]
    served = sum(min(n, capacity) for n in degenerate.values())

    assert len(idle) == experts - top_k == 6
    assert served == capacity * top_k
    assert served < tokens * top_k / 2, "most routed slots were dropped"

    # The pattern actually used has neither property.
    balanced = inference_mix.routing_balance(tokens, experts, top_k)
    assert not [e for e, n in balanced.items() if n == 0]
    assert sum(min(n, capacity) for n in balanced.values()) == tokens * top_k


def test_the_dispatch_gathers_activations_not_weight_matrices():
    """The compile failure, guarded against return.

    `expert_weights[chosen]` with one index per token materialises
    [tokens, hidden, hidden]. At the pinned problem that is 137 GB, and the
    compiler refused it: 4,194,304 instructions against a limit of 150,000.

    Read as code, not as text: the comment explaining this defect quotes
    it, and a naive substring check passes on the explanation.
    """
    code = sourcecheck.function_code(inference_mix.run_moe_router)

    assert "expert_weights [ chosen ]" not in code
    assert "index_select" in code, "the gather must be over token vectors"
    assert "expert_weights [ expert ]" in code, "one weight matrix per expert"


def test_the_gathered_weight_table_stays_small():
    """Arithmetic for why the old dispatch could not work."""
    problem = PROBLEMS["moe_router"]
    tokens, hidden, experts = (problem["tokens"], problem["hidden"],
                               problem["experts"])

    per_token_gather = tokens * hidden * hidden * 2      # bf16
    whole_table = experts * hidden * hidden * 2
    assert per_token_gather > 100e9
    assert whole_table < 1e9
    assert per_token_gather > 400 * whole_table


# -- kv_cache_churn measured the runtime, not the cache ----------------------
#
# trn1.2xlarge 2026-09-08: 16 KiB per step in 115 microseconds, which is
# 1,794x longer than HBM needs for 16 KiB and 0.056% of the part's
# bandwidth. cache-updates/s looks identical whether each update moved a
# cache or a register, which is why the row now carries the bandwidth too.

def test_the_pinned_cache_fits_on_a_neuroncore():
    plan = llm_inference.cache_plan(PROBLEMS["kv_cache_churn"])
    assert plan["resident_bytes"] == 128 * 1024**2
    assert plan["resident_bytes"] < 16 * 1024**3 / 2


def test_a_step_costs_a_whole_cache_copy_not_a_slice():
    """The finding: XLA has no in-place update.

    cache[:, a:b, :] = entry lowers to dynamic-update-slice, which produces
    a new tensor. So an append reads the cache and writes a new one, and
    the traffic is twice the resident size regardless of how few tokens
    were appended. Counting the slice would report a sixteenth of what the
    hardware moves at the pinned size.
    """
    plan = llm_inference.cache_plan(PROBLEMS["kv_cache_churn"])

    assert plan["bytes_per_step"] == 2 * plan["resident_bytes"]
    assert plan["bytes_per_step"] > plan["slice_bytes"]
    # The gap is the size of the mistake, and it grows with the cache.
    assert plan["bytes_per_step"] / plan["slice_bytes"] == plan["ring_slots"] * 2


def test_a_step_writes_every_layer_not_one():
    """A KV cache is per layer; an append touches all of them."""
    problem = PROBLEMS["kv_cache_churn"]
    plan = llm_inference.cache_plan(problem)
    one_layer_one_token = 2 * problem["hidden"] * 2
    assert plan["slice_bytes"] == (
        plan["tokens_per_step"] * plan["layers"] * one_layer_one_token)


def test_the_pinned_step_is_large_enough_to_time_the_write():
    """About 1 ms of bandwidth per step.

    Large enough that the copy dominates, small enough that its graphs
    compile in something like a minute. The 2 GiB cache tried before this
    took roughly seven minutes per ring slot, eight slots, and never
    produced a number at all.
    """
    plan = llm_inference.cache_plan(PROBLEMS["kv_cache_churn"])
    at_hbm_ms = plan["bytes_per_step"] / (
        llm_inference.MEASURED_HBM_GBPS * 1e9) * 1000
    assert 0.5 < at_hbm_ms < 5.0

    # The cache that could not finish, for contrast.
    huge = llm_inference.cache_plan(
        {"hidden": 4096, "context": 4096, "layers": 32,
         "ring_slots": 8, "dtype": "bf16"})
    assert huge["bytes_per_step"] / (
        llm_inference.MEASURED_HBM_GBPS * 1e9) * 1000 > 15.0


def test_the_ring_compiles_one_graph_per_slot_and_no_more():
    """Each slot is a static slice, so the count is the compile cost.

    A position per token would be `context` graphs; 8 slots is 8.
    """
    plan = llm_inference.cache_plan(PROBLEMS["kv_cache_churn"])
    assert plan["ring_slots"] == 8
    assert plan["ring_slots"] * plan["tokens_per_step"] == plan["context"]
    assert plan["ring_slots"] < 16, "each slot costs a compile"


@pytest.mark.parametrize("bad", [
    {"hidden": 0, "context": 4096, "dtype": "bf16"},
    {"hidden": 4096, "context": 0, "dtype": "bf16"},
    {"hidden": 4096, "context": 4096, "layers": 0, "dtype": "bf16"},
    {"hidden": 4096, "context": 4096, "ring_slots": 0, "dtype": "bf16"},
])
def test_cache_plan_rejects_impossible_geometry(bad):
    with pytest.raises(ValueError, match="must be positive"):
        llm_inference.cache_plan(bad)


def test_the_ring_must_divide_the_context():
    """A partial slot would make one step write a different amount."""
    with pytest.raises(ValueError, match="whole ring slots"):
        llm_inference.cache_plan({"hidden": 4096, "context": 100,
                                  "ring_slots": 8, "dtype": "bf16"})


def test_the_write_is_a_static_slice_not_a_scatter():
    """The primitive is the finding: scatter cost 54x the bytes it moved.

    Read as code, because the comment explaining the defect names it.
    """
    code = sourcecheck.function_code(llm_inference.run_cache_churn)
    assert "index_copy_" not in code, "a runtime index means a scatter"
    assert "cache_k [ : , start : stop , : ] = entry" in code
    # Slot bounds are Python ints closed over per writer, so each slot is
    # its own graph rather than an index reaching the device.
    assert "start , stop = slot * tokens" in code


def test_the_old_measurement_would_now_be_flagged():
    """0.143 GB/s against 256.2: the run that started this."""
    message = llm_inference.verify_memory_bound(0.143e9)
    assert message is not None
    assert "timed dispatch, not memory" in message


def test_a_memory_bound_run_is_not_flagged():
    assert llm_inference.verify_memory_bound(134e9) is None


def test_an_untouched_cache_is_flagged():
    assert "never touched" in llm_inference.verify_memory_bound(0.0)


def test_the_floor_is_where_it_is_documented():
    hbm = llm_inference.MEASURED_HBM_GBPS * 1e9
    assert llm_inference.verify_memory_bound(hbm * 0.04) is not None
    assert llm_inference.verify_memory_bound(hbm * 0.06) is None


# -- serving_mix counted blocks and called them requests ---------------------
#
# It ran one transformer block per "request". A real prefill runs `layers`
# blocks and a real 256-token decode request runs decode * layers of them --
# 32x and 8,192x at the pinned problem. It also reported
# requested_decode_length: 256 beside a loop that never produced a second
# token.

def test_a_scheduler_step_runs_every_layer():
    plan = inference_mix.serving_plan(PROBLEMS["serving_mix"])
    assert plan["blocks_per_step"] == plan["layers"] == 32
    assert plan["blocks_per_step"] > 1, "one block is not a model pass"


def test_a_decode_request_takes_many_steps_to_finish():
    """256 tokens at batch 8 is 32 steps, not one."""
    plan = inference_mix.serving_plan(PROBLEMS["serving_mix"])
    assert plan["decode_steps_per_request"] == 32
    assert plan["decode_steps_per_request"] * plan["batch"] >= plan["decode"]


def test_the_old_accounting_overstated_by_these_factors():
    """The size of the defect, as arithmetic."""
    plan = inference_mix.serving_plan(PROBLEMS["serving_mix"])
    layers, decode = plan["layers"], plan["decode"]

    assert layers == 32                       # a prefill was 1/32 of itself
    assert decode * layers == 8192            # a decode request 1/8192


def test_a_prefill_step_costs_far_more_than_a_decode_step():
    """They were counted as the same unit; they are not the same work."""
    plan = inference_mix.serving_plan(PROBLEMS["serving_mix"])
    assert plan["prefill_flops"] > 100 * plan["decode_flops"]


def test_completed_requests_come_from_tokens_not_steps():
    """The accounting that replaced one-request-per-block, in the open."""
    plan = inference_mix.serving_plan(PROBLEMS["serving_mix"])
    batch, decode = plan["batch"], plan["decode"]

    for decode_steps in (0, 16, 32, 64, 100):
        tokens = decode_steps * batch
        assert tokens // decode == max(0, decode_steps // 32)


def test_the_step_runs_the_stack_not_a_block():
    """Asserted against the source, since running it needs a device."""
    code = sourcecheck.function_code(inference_mix.run_serving_mix)
    assert "for _ in range ( layers )" in code
    assert "state = transformer_ops . block ( state , params )" in code


@pytest.mark.parametrize("bad", [
    {"prefill_ratio": 0.2, "batch": 0, "prompt": 8, "decode": 8},
    {"prefill_ratio": 0.2, "batch": 8, "prompt": 0, "decode": 8},
    {"prefill_ratio": 0.2, "batch": 8, "prompt": 8, "decode": 0},
    {"prefill_ratio": 0.2, "batch": 8, "prompt": 8, "decode": 8, "layers": 0},
])
def test_serving_plan_rejects_impossible_problems(bad):
    with pytest.raises(ValueError, match="must be positive"):
        inference_mix.serving_plan(bad)


def test_serving_plan_still_rejects_a_bad_ratio():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        inference_mix.serving_plan(
            {"prefill_ratio": 1.5, "batch": 8, "prompt": 8, "decode": 8})


def test_speculative_decode_verifies_through_the_target_model():
    """Verification runs every layer, not one block.

    Speculative decoding exists to amortise the target model's cost across
    several drafted tokens. Verifying through a single block makes that
    cost a thirty-second of itself, so the workload would report the
    technique winning against a target it never ran.
    """
    code = sourcecheck.function_code(inference_mix.run_speculative_decode)
    assert "for _ in range ( layers )" in code
    assert "state = transformer_ops . block ( state , target )" in code


def test_the_pinned_speculative_problem_names_its_target_depth():
    problem = PROBLEMS["speculative_decode"]
    assert problem["layers"] == 32
    assert problem["draft_len"] == 4


def test_verification_dominates_drafting():
    """The economic premise: drafting must be cheap relative to verifying.

    If it is not, speculative decoding is a slower way to decode, and a
    workload whose draft costs as much as its verify is measuring
    something other than the technique.
    """
    problem = PROBLEMS["speculative_decode"]
    hidden, draft_len, layers = (problem["hidden"], problem["draft_len"],
                                 problem["layers"])

    verify = layers * transformer_ops.block_flops(hidden, draft_len)
    # The draft runs at a quarter width, one matmul per token.
    draft = draft_len * 2 * (hidden // 4) ** 2

    assert verify > 100 * draft


# -- the encoders ran a fraction of the models they name ---------------------
#
# Four workloads in a row had this: serving_mix and speculative_decode ran
# one block where a model runs `layers`, and so did vision_encoder.
# rag_embedding ran no blocks at all -- two matmuls and a normalise.

def test_rag_embedding_runs_an_encoder_stack():
    code = sourcecheck.function_code(encoders.run_rag_embedding)
    assert "for _ in range ( layers )" in code
    assert "transformer_ops . block ( state , params )" in code
    # It embeds a token sequence and pools it, rather than starting from a
    # vector that has already been pooled.
    assert "state . mean ( dim = 1 )" in code


def test_vision_encoder_runs_every_layer():
    code = sourcecheck.function_code(encoders.run_vision_encoder)
    assert "for _ in range ( layers )" in code


@pytest.mark.parametrize("name", ["rag_embedding", "vision_encoder"])
def test_the_encoders_pin_a_depth(name):
    assert PROBLEMS[name]["layers"] == 12, "ViT-B and BERT-base are twelve"


def test_the_embedder_reads_a_sequence_not_a_vector():
    problem = PROBLEMS["rag_embedding"]
    assert problem["seq"] == 128, "retrieval documents are token sequences"
    assert problem["batch"] == 64


def test_the_encoder_flop_counts_follow_the_pinned_depth():
    """The arithmetic that makes the old numbers implausible.

    A projection is two matmuls; a 12-layer encoder is three orders of
    magnitude more work, and the reported rate should differ accordingly.
    """
    problem = PROBLEMS["rag_embedding"]
    dim, seq, batch, layers = (problem["dim"], problem["seq"],
                               problem["batch"], problem["layers"])

    encoder = layers * transformer_ops.block_flops(dim, seq, batch)
    projection = 2 * (2 * batch * dim * dim)      # what it used to run
    assert encoder > 1000 * projection


# -- omni_virus advertised a shape it did not run ----------------------------

def test_omni_virus_declares_when_it_cut_the_shape_down():
    """`problem` is the comparison contract; a row must not overstate it.

    The kernel runs min(m, 2048) while the registry pins 8192^3, so a
    cross-platform comparison joining on Problem would put a GPU's 8192^3
    against a Neuron 2048^3. Same defect as serving_mix's
    requested_decode_length: a field naming work that did not happen.
    """
    cut = omni_virus._shape_warning(2048, 8192, 8192, 8192, {"warning": None})
    assert "ran 2048^3" in cut["warning"]
    assert "8192x8192x8192" in cut["warning"]

    full = omni_virus._shape_warning(8192, 8192, 8192, 8192, {"warning": None})
    assert full["warning"] is None


def test_the_shape_note_does_not_displace_an_output_failure():
    """A NaN still has to reach the row, and still has to fail it."""
    checked = {"warning": "chain output is NaN", "score_invalid": True}
    both = omni_virus._shape_warning(2048, 8192, 8192, 8192, checked)
    assert "NaN" in both["warning"] and "ran 2048^3" in both["warning"]
    assert both["score_invalid"] is True


def test_omni_virus_runs_the_pinned_shape_by_default():
    """The cap was a workaround for a limitation that no longer exists.

    Measured on trn1.2xlarge 2026-09-08: 22.53 TFLOPS at 2048, 34.83 at
    4096, 48.13 at the pinned 8192 -- so the cap cost more than half the
    throughput as well as making the row advertise a shape it did not run.
    """
    import sourcecheck

    code = sourcecheck.function_code(omni_virus.run)
    assert 'PANTHEON_NEURON_OMNI_TILE" , m' in code, "the default is the pin"
    assert '"PANTHEON_NEURON_OMNI_TILE" , 2048' not in code


def test_the_cap_survives_as_an_override():
    """A bring-up on a new part wants a shape that compiles in seconds."""
    import sourcecheck

    code = sourcecheck.function_code(omni_virus.run)
    assert "os . environ . get" in code
    assert "min ( m ," in code


def test_a_run_that_finishes_no_decode_request_says_so():
    """Otherwise requests/s is prefills wearing the name of a mix.

    A decode request needs decode/batch steps -- 32 at the pinned 256
    tokens -- so a short run produces decode tokens and completes no decode
    request. trn1.2xlarge 2026-09-08 reported 0.0312 requests/s from a
    20-second run on that basis.
    """
    message = inference_mix.verify_requests_completed(6, 26, 0)
    assert message is not None
    assert "prefills only" in message

    assert inference_mix.verify_requests_completed(6, 64, 2) is None
    # A pure-prefill mix is not incomplete, it is a different mix.
    assert inference_mix.verify_requests_completed(6, 0, 0) is None


def test_the_mix_note_does_not_displace_an_output_failure():
    checked = {"warning": "serving output is NaN", "score_invalid": True}
    both = inference_mix._mix_warning(10, 5, 6, 26, 0, checked)
    assert "NaN" in both["warning"] and "prefills only" in both["warning"]
    assert both["score_invalid"] is True


def test_how_long_the_pinned_mix_needs_to_mean_anything():
    """Recorded so the duration is a decision rather than a surprise."""
    plan = inference_mix.serving_plan(PROBLEMS["serving_mix"])
    steps_per_request = plan["decode_steps_per_request"]
    assert steps_per_request == 32

    # The scheduler sustained about one step a second on trn1.
    measured_steps_per_s = 1.0
    seconds_for_one_request = steps_per_request / measured_steps_per_s
    assert seconds_for_one_request > 20, "a 20s run cannot finish one"


def test_a_run_too_short_to_sample_the_mix_says_so():
    """The hole the narrower guard left.

    trn1.2xlarge 2026-09-09: a 20-second run completed ONE scheduler step,
    reported 0.031 requests/s from that single prefill, and carried no
    caveat -- because verify_requests_completed only fires once a decode
    step has run, and none had.
    """
    assert inference_mix.verify_mix_was_observed(1, 5) is not None
    assert "never completed one cycle" in inference_mix.verify_mix_was_observed(1, 5)
    assert inference_mix.verify_mix_was_observed(5, 5) is None
    assert inference_mix.verify_mix_was_observed(100, 5) is None
    # A ratio of zero means no interleave to sample.
    assert inference_mix.verify_mix_was_observed(1, 0) is None


def test_both_mix_caveats_can_fire_together():
    both = inference_mix._mix_warning(1, 5, 1, 0, 0, {"warning": None})
    assert "never completed one cycle" in both["warning"]

    tokens_no_request = inference_mix._mix_warning(
        10, 5, 2, 8, 0, {"warning": None})
    assert "prefills only" in tokens_no_request["warning"]

    healthy = inference_mix._mix_warning(100, 5, 20, 80, 2, {"warning": None})
    assert healthy["warning"] is None


def test_serving_mix_warms_both_graph_shapes():
    """A prefill and a decode step are different graphs.

    Warming only decode left the prefill to compile inside the timed
    region -- the defect memory_read documents, where a mismatched warm-up
    measured seven minutes of compilation as bandwidth. Measured here: one
    scheduler step in twenty seconds, that step taking about 32.
    """
    code = sourcecheck.function_code(inference_mix.run_serving_mix)
    warmup = code[:code.index("while time . perf_counter")]
    assert "for shape in ( decode_batch , prompt_batch )" in warmup


# -- a rate with nothing beside it cannot be checked --------------------------

@pytest.mark.parametrize("function", ["run_fused_attention", "run_moe_router"])
def test_the_attention_and_router_rates_carry_an_implied_flops(function):
    """A tile or token count says nothing about whether the arithmetic
    happened. Every defect found in this suite came from one number
    disagreeing with another, and these two had nothing to disagree with.
    """
    code = sourcecheck.function_code(getattr(inference_mix, function))
    assert '"implied_tflops"' in code
    assert '"flops_issued"' in code


def test_the_router_counts_its_passes():
    """implied_tflops needs a pass count, and this loop had none."""
    code = sourcecheck.function_code(inference_mix.run_moe_router)
    assert "passes = 0" in code
    assert "passes += 1" in code


def test_the_implied_rates_land_where_the_hardware_says_they_should():
    """The arithmetic here, checked against what the kernel computed there.

    Measured on trn1.2xlarge 2026-09-10, 30s each:

        fused_attention  6043.18 tiles/s          implied_tflops 12.978
        moe_router     255288.26 routed-tokens/s  implied_tflops 17.132

    against tensor_virus at 26.1 TFLOPS dense on the same part. Attention
    is softmax-bound and dispatch is gather-bound, so both landing under
    the dense figure is the expected shape -- and either exceeding it
    would mean the FLOP count is wrong.

    Recomputing both from the published rate is the point: it is the
    off-hardware arithmetic held against the on-hardware arithmetic, and
    the two agreeing is what makes implied_tflops a check rather than
    another number nothing can contradict.
    """
    attention = PROBLEMS["fused_attention"]
    hidden = attention["heads"] * attention["head_dim"]
    passes = 6043.176933016786 / attention["heads"]
    implied = passes * 2 * (2 * attention["seq"] ** 2 * hidden) / 1e12
    assert implied == pytest.approx(12.97762364562434, rel=1e-6)
    assert implied < 26.1

    router = PROBLEMS["moe_router"]
    capacity = inference_mix.expert_capacity(
        router["tokens"], router["top_k"], router["experts"])
    assert capacity == 1024, capacity
    passes = 255288.26031910875 / router["tokens"]
    implied = passes * router["experts"] * 2 * capacity * router["hidden"] ** 2 / 1e12
    assert implied == pytest.approx(17.132105142551666, rel=1e-6)
    assert implied < 26.1


# -- a vector or tile count is not evidence of arithmetic --------------------

@pytest.mark.parametrize("function", ["run_rag_embedding",
                                      "run_vision_encoder"])
def test_the_encoder_rates_carry_an_implied_flops(function):
    """Same gap fused_attention and moe_router had, in the same shape.

    A kernel that skips the encoder still produces vectors and still
    produces tiles. Every defect found in this suite came from one number
    disagreeing with another, and neither of these had anything to
    disagree with.
    """
    code = sourcecheck.function_code(getattr(encoders, function))
    assert '"implied_tflops"' in code
    assert '"flops_issued"' in code


def test_the_vision_flop_count_is_computed_once():
    """It was inline in the dict, which is why adding a rate needed it out.

    A figure computed twice is a figure that can be changed in one place.
    """
    code = sourcecheck.function_code(encoders.run_vision_encoder)
    assert code.count("block_flops") == 1


def test_speculative_decode_reports_a_total_not_only_a_per_cycle_cost():
    """A constant cannot disagree with a measurement.

    ``verify_flops_per_cycle`` was already reported and is the same number
    on every run -- it describes the pinned problem, not the execution. A
    run that never reached the target model produces an identical cycle
    count and an identical per-cycle cost; only the total and the rate
    move.
    """
    code = sourcecheck.function_code(inference_mix.run_speculative_decode)
    assert '"flops_issued"' in code
    assert '"implied_tflops"' in code
    assert '"verify_flops_per_cycle"' in code
    # And computed once, so the total and the per-cycle figure cannot drift
    # apart.
    assert code.count("block_flops") == 1


# -- the one kernel whose answer is known in advance -------------------------

def test_uniform_attention_accepts_the_value_it_must_produce():
    assert transformer_ops.verify_attention_is_uniform(1.0) is None
    assert transformer_ops.verify_attention_is_uniform(0.995) is None
    assert transformer_ops.verify_attention_is_uniform(1.005) is None


@pytest.mark.parametrize("wrong,why", [
    (0.0, "a mask applied by mistake, or a saturated softmax at the far end"),
    (2048.0, "weights summing to seq instead of to one"),
    (0.000488, "the mean taken over the wrong axis"),
    (float("inf"), "saturation -- finite is not the test here"),
])
def test_uniform_attention_rejects_finite_wrong_answers(wrong, why):
    """Every one of these passed the previous check, which asked only
    whether the number was readable and not NaN.

    That is the point of the change: attention over constant inputs has a
    known answer, so a plausibility check was standing where a
    correctness check could stand.
    """
    message = transformer_ops.verify_attention_is_uniform(wrong)
    assert message is not None, why


def test_a_nan_is_still_caught():
    """The old check's job, kept."""
    message = transformer_ops.verify_attention_is_uniform(float("nan"))
    assert message is not None
    assert "NaN" in message


def test_an_unreadable_output_is_not_silently_accepted():
    message = transformer_ops.verify_attention_is_uniform(None)
    assert message is not None
    assert "could not be read" in message


def test_the_verdict_and_the_flag_travel_together():
    """Separating them is how three NaN runs published Scores."""
    good = transformer_ops.attention_check(1.0)
    assert good["warning"] is None and good["score_invalid"] is False
    bad = transformer_ops.attention_check(0.0)
    assert bad["warning"] is not None and bad["score_invalid"] is True


def test_fused_attention_uses_the_stricter_check():
    code = sourcecheck.function_code(inference_mix.run_fused_attention)
    assert "attention_check" in code
    assert "output_check" not in code


def test_the_expected_value_follows_from_the_pinned_problem():
    """Softmax over identical scores is uniform whatever the shape, so
    the context is v's fill value -- which the kernel sets to one.

    Asserted rather than assumed, because if the kernel ever fills v with
    something else the expected value moves with it and this check would
    otherwise start failing for the right reason with the wrong message.
    """
    code = sourcecheck.function_code(inference_mix.run_fused_attention)
    assert "v = torch . ones (" in code
    assert "attention_check ( transformer_ops . read_back ( sink ) )" in code


# -- the stack's answer follows from its depth -------------------------------

def test_the_per_block_increment_is_one_plus_gelu_of_one():
    """Derived from the block, and independently from rms_norm's own note.

    rms_norm's docstring says the residual stream "grows additively
    rather than multiplicatively -- about 2 per block instead of a factor
    of 4", reasoning that predates this check. 1.8413 is that 2.
    """
    step = (transformer_ops.stacked_block_output(1)
            - transformer_ops.stacked_block_output(0))
    assert step == pytest.approx(1.8413447, rel=1e-6)
    assert "about 2 per block" in transformer_ops.rms_norm.__doc__


@pytest.mark.parametrize("layers,expected", [
    (0, 1.0), (1, 2.8413447), (4, 8.3653790), (32, 59.9230319)])
def test_the_stack_output_is_linear_in_depth(layers, expected):
    """Pre-norm is what makes it depth-independent: the input to every
    matmul is unit scale however deep the stack, so neither branch's
    contribution depends on x. A stack that had lost its normalisation
    would grow multiplicatively and miss this by orders of magnitude.
    """
    assert transformer_ops.stacked_block_output(layers) == pytest.approx(
        expected, rel=1e-6)


def test_a_stack_that_ran_the_declared_depth_passes():
    for slack in (1.0, 0.95, 1.05):
        assert transformer_ops.verify_stack_computed_its_depth(
            59.923 * slack, 32) is None


def test_a_stack_that_ran_half_its_depth_is_caught():
    """The check nothing else in the suite makes.

    flops_issued multiplies by `layers` whether or not that many ran, so
    a stack executing half its depth reports the full arithmetic at twice
    the throughput and reads as good news.
    """
    message = transformer_ops.verify_stack_computed_its_depth(
        transformer_ops.stacked_block_output(16), 32)
    assert message is not None
    assert "about 16" in message


@pytest.mark.parametrize("wrong,why", [
    (1.0, "no block ran at all"),
    (2.8413, "one block ran where 32 were declared"),
    (1e6, "multiplicative growth -- the normalisation is gone"),
    (float("inf"), "saturation; finite is not the test here"),
])
def test_finite_wrong_answers_are_rejected(wrong, why):
    assert transformer_ops.verify_stack_computed_its_depth(wrong, 32), why


def test_a_nan_and_an_unreadable_output_are_still_caught():
    assert "NaN" in transformer_ops.verify_stack_computed_its_depth(
        float("nan"), 32)
    assert "could not be read" in transformer_ops.verify_stack_computed_its_depth(
        None, 32)


def test_the_tolerance_admits_bf16_drift_and_nothing_larger():
    """bf16 carries about three decimal digits and the sum reaches ~60
    over 32 accumulations, so a few percent is expected. The failures
    this catches are factors, not percentages.
    """
    expected = transformer_ops.stacked_block_output(32)
    assert transformer_ops.verify_stack_computed_its_depth(
        expected * 1.09, 32) is None
    assert transformer_ops.verify_stack_computed_its_depth(
        expected * 1.11, 32) is not None


@pytest.mark.parametrize("module,function", [
    ("llm_inference", "run_prefill"),
    ("transformer_compute", "run_virus"),
])
def test_the_stacks_use_the_depth_check(module, function):
    import importlib
    target = importlib.import_module(f"kernels.{module}")
    code = sourcecheck.function_code(getattr(target, function))
    assert "stack_check" in code
    assert "output_check" not in code


def test_the_tolerance_is_five_times_the_drift_bf16_actually_produces():
    """The threshold is measured, not guessed.

    Simulating the residual walk in bf16 -- the rounding lands on the two
    additions per block, since the matmuls accumulate in fp32 -- gives
    -1.96% at 32 layers. A first estimate from worst-case ulp
    accumulation said 7%, which would have left almost no headroom.
    Round-to-nearest does far better than worst case because the errors
    do not share a sign.

    This matters in one direction: the check sets score_invalid, so a
    false positive turns a working run into a FAIL, and a check that
    fails good runs gets widened until it means nothing.
    """
    def bf16(value):
        if value == 0:
            return 0.0
        ulp = 2.0 ** (math.floor(math.log2(abs(value))) - 7)
        return round(value / ulp) * ulp

    gelu_bf16 = bf16(0.8413447460685429)
    for layers in (1, 4, 32):
        walked = 1.0
        for _ in range(layers):
            walked = bf16(walked + 1.0)
            walked = bf16(walked + gelu_bf16)
        exact = transformer_ops.stacked_block_output(layers)
        drift = abs(walked / exact - 1.0)
        assert drift < 0.02, (layers, walked, exact, drift)
        # And the simulated value must pass the real check.
        assert transformer_ops.verify_stack_computed_its_depth(
            walked, layers) is None, (layers, walked)


def test_decode_uses_the_depth_check_too():
    """Its recursion is prefill's: rms_norm makes every branch input unit
    scale, attention over a constant cache is uniform whatever the
    context length, and both residuals add 1 and gelu(1).
    """
    code = sourcecheck.function_code(llm_inference.run_decode)
    assert "stack_check" in code
    assert "output_check" not in code


def test_decode_omits_the_softmax_scale_and_nothing_can_see_it():
    """A limit of constant-input verification, asserted so it stays known.

    transformer_ops.attention divides scores by sqrt(head_dim); decode's
    inline attention does not. With all-ones inputs every score is
    identical and softmax is uniform either way, so the omission changes
    no observable value -- not the output, not the FLOP count, not the
    rate. It is a real structural difference from prefill that no check
    built on constant inputs can distinguish.

    This test does not fix that. It records it, so the next person to
    reconcile the two paths finds the reason rather than the divergence.
    """
    shared = sourcecheck.function_code(transformer_ops.attention)
    assert "head_dim ** 0.5" in shared

    decode = sourcecheck.function_code(llm_inference.run_decode)
    assert "torch . softmax" in decode, "decode still inlines its attention"
    assert "head_dim" not in decode, (
        "decode now references head_dim -- if it has adopted the scale, "
        "this test should be deleted rather than updated")


# -- an encoder whose blocks could not move its output -----------------------

def _bf16(value):
    if value == 0:
        return 0.0
    ulp = 2.0 ** (math.floor(math.log2(abs(value))) - 7)
    return round(value / ulp) * ulp


def _walk(start, layers):
    """The residual walk in bf16, where the rounding actually lands."""
    x, gelu = start, _bf16(0.8413447460685429)
    for _ in range(layers):
        x = _bf16(x + 1.0)
        x = _bf16(x + gelu)
    return x


def test_an_unscaled_patch_projection_makes_the_blocks_invisible():
    """The defect, reproduced in arithmetic.

    patch_projection was torch.ones, so the embedding was patch_dim --
    588 at the pinned 14x14x3. bf16's ulp at 588 is 4.0, and each block
    adds 1 + gelu(1) = 1.84. Every contribution rounded away.

    The arithmetic ran: the FLOPs were issued and the throughput was
    real. But the output was bit-identical to running zero blocks, so no
    check on it could have distinguished a working twelve-block encoder
    from a broken one.
    """
    patch_dim = 14 * 14 * 3
    assert _walk(float(patch_dim), 12) == float(patch_dim), (
        "the simulation no longer reproduces the defect")
    assert 2.0 ** (math.floor(math.log2(patch_dim)) - 7) == 4.0


def test_the_scaled_projection_puts_the_stack_where_it_can_be_checked():
    """1/patch_dim gives an embedding of 1.0, like every other stack."""
    walked = _walk(1.0, 12)
    assert walked == pytest.approx(23.1, abs=0.1)
    assert transformer_ops.verify_stack_computed_its_depth(walked, 12) is None


def test_the_vision_encoder_scales_its_patch_projection():
    code = sourcecheck.function_code(encoders.run_vision_encoder)
    assert "torch . full (" in code
    assert '1.0 / plan [ "patch_dim" ]' in code
    assert "patch_projection = torch . ones (" not in code


@pytest.mark.parametrize("function", ["run_rag_embedding",
                                      "run_vision_encoder"])
def test_both_encoders_check_their_depth(function):
    code = sourcecheck.function_code(getattr(encoders, function))
    assert "stack_check" in code
    assert "output_check" not in code


def test_the_defect_would_now_be_caught():
    """The check has to actually reject the old behaviour, or scaling the
    projection just moved the problem somewhere unobserved.
    """
    unscaled = _walk(float(14 * 14 * 3), 12)
    message = transformer_ops.verify_stack_computed_its_depth(unscaled, 12)
    assert message is not None


# -- the property all three unscaled-weight defects violated -----------------

def test_a_block_is_observable_at_unit_scale():
    """Which is why every stack in this suite must start near one."""
    for magnitude in (1.0, 2.84, 23.1, 59.92):
        assert transformer_ops.increment_is_observable(magnitude, 1.8413), \
            magnitude


@pytest.mark.parametrize("magnitude,workload", [
    (588.0, "vision_encoder's unscaled patch projection"),
    (1024.0 ** 4 * 1024, "speculative_decode's unscaled draft chain"),
])
def test_a_block_vanishes_at_the_magnitudes_the_defects_produced(
        magnitude, workload):
    """Both silent. The arithmetic ran and the output did not depend on it.

    llm_prefill is the same defect from the other end -- unscaled weights
    grew the residual until it left bf16's range and the output was NaN.
    That one was loud, which is why it was found first.
    """
    assert not transformer_ops.increment_is_observable(magnitude, 1.8413), \
        workload


def test_the_ulp_helper_matches_bfloat16():
    """8 explicit mantissa bits, so the spacing doubles every octave."""
    assert transformer_ops.ulp(1.0) == 2.0 ** -7
    assert transformer_ops.ulp(2.0) == 2.0 ** -6
    assert transformer_ops.ulp(588.0) == 4.0
    assert transformer_ops.ulp(0.0) == 0.0
    # fp32 for comparison: 24 significand bits.
    assert transformer_ops.ulp(1.0, mantissa_bits=24) == 2.0 ** -23


def test_every_stack_in_the_suite_starts_where_its_blocks_are_visible():
    """The guard for the class, rather than a fix for each instance.

    Three separate kernels shipped an unscaled weight. A fourth will, so
    the invariant is stated once: every stack starts at unit scale, its
    increment is observable there, and it stays observable through the
    whole depth.
    """
    for layers in (1, 12, 32):
        for step in range(layers + 1):
            magnitude = transformer_ops.stacked_block_output(step)
            assert transformer_ops.increment_is_observable(
                magnitude, 1.8413), (layers, step, magnitude)


def test_speculative_decode_scales_its_draft_chain():
    code = sourcecheck.function_code(inference_mix.run_speculative_decode)
    assert "1.0 / draft_hidden" in code
    assert "draft_w = torch . ones (" not in code
    assert "project_up = torch . ones (" not in code
    assert "stack_check" in code


def test_an_unmovable_output_is_reported_as_such_not_as_a_wrong_value():
    """"Wrong value" is the wrong complaint when no value was possible.

    At a magnitude where a block's 1.84 is below half an ulp, the output
    is bit-identical to running zero blocks -- so it is not evidence
    about the stack in either direction, and saying "expected 610, got
    588" invites someone to adjust the expectation.
    """
    message = transformer_ops.verify_stack_computed_its_depth(588.0, 12)
    assert message is not None
    assert "zero blocks" in message
    assert "says nothing" in message
    assert "must produce" not in message, "reported as a wrong value"


def test_the_speculative_magnitude_is_reported_the_same_way():
    message = transformer_ops.verify_stack_computed_its_depth(1.1e15, 32)
    assert message is not None
    assert "zero blocks" in message


def test_a_zero_layer_stack_is_not_accused_of_being_unmovable():
    """With no blocks there is no increment to be invisible."""
    assert transformer_ops.verify_stack_computed_its_depth(1.0, 0) is None


def test_an_infinite_output_is_a_message_not_an_exception():
    """It raised OverflowError from inside a verdict function.

    verify_output_is_a_number lets infinity pass on purpose -- these
    stacks were expected to saturate before they were normalised -- so an
    inf reached the ulp arithmetic through the NaN gate, inf not being
    NaN, and math.floor(inf) raised from a function whose entire job is
    to return a message rather than throw.
    """
    message = transformer_ops.verify_stack_computed_its_depth(
        float("inf"), 32)
    assert message is not None
    assert "saturated" in message

    assert transformer_ops.ulp(float("inf")) == float("inf")
    assert not transformer_ops.increment_is_observable(float("inf"), 1.8413)


# -- an output that its own normalisation erased -----------------------------

def test_l2_normalisation_makes_the_embedding_independent_of_the_stack():
    """rag_embedding's published output cannot see its own encoder.

    Every element of the returned vector is 1/sqrt(dim), and it is
    1/sqrt(dim) whether twelve blocks ran, one ran, or none did --
    because L2 normalisation is scale-invariant and every element of the
    stack output is identical over constant inputs.

    Fourth workload found producing an output that does not depend on its
    own arithmetic, after vision_encoder, speculative_decode and, from
    the other end, llm_prefill. Unlike those three it is not a bug: L2
    normalising is what a retrieval embedder does. The answer is to check
    the value the normalisation erased.
    """
    dim = 1024
    for layers in (0, 1, 12):
        stack_value = transformer_ops.stacked_block_output(layers)
        # mean over a sequence of identical values, then L2 over dim of them
        pooled = stack_value
        norm = math.sqrt(dim * pooled * pooled)
        assert pooled / norm == pytest.approx(1.0 / math.sqrt(dim)), layers

    assert 1.0 / math.sqrt(dim) == 0.03125


def test_rag_embedding_checks_the_stack_and_not_only_the_vector():
    code = sourcecheck.function_code(encoders.run_rag_embedding)
    assert "stack_output" in code
    assert '"expected_embedding_element"' in code
    # The depth check must read the stack, not the normalised vector.
    assert "stack_check (\n" not in code
    assert 'read_back ( stack_output . get ( "state" ) )' in code


# -- a training step that does not train -------------------------------------

def test_an_unmoved_parameter_is_reported():
    """train-steps/s measures the cost of a step, not that one happened.

    Parameters are bf16 at 1/fan_in; q sits at 2.44e-4 where bf16's ulp
    is 1.91e-6. The loss is a mean over batch*seq*hidden elements, so a
    weight's gradient is about 2.05e-4 and an SGD step at lr=1e-4 moves
    it by 2.05e-8 -- 2% of a half-ulp. SGD applies each update
    independently rather than into an accumulator, so every step rounds
    back and the weights are bit-identical after a thousand steps.
    """
    message = transformer_ops.verify_optimiser_moved_the_model(
        6.104e-05, 6.104e-05)
    assert message is not None
    assert "unchanged" in message
    assert "does not measure training" in message


def test_a_moved_parameter_says_nothing():
    """The control -- if the optimiser works this must stay quiet."""
    assert transformer_ops.verify_optimiser_moved_the_model(
        6.104e-05, 6.105e-05) is None


def test_an_unreadable_parameter_is_not_treated_as_unmoved():
    """Absent is not equal. Inventing a verdict from a failed read would
    accuse a working optimiser."""
    for pair in ((None, 1.0), (1.0, None), (None, None)):
        message = transformer_ops.verify_optimiser_moved_the_model(*pair)
        assert message is not None
        assert "could not read" in message


def test_it_warns_rather_than_invalidating():
    """The Score is right: forward, backward and step all executed and
    the cost is real. It is the name that misleads, so this is a warning.
    """
    correct = transformer_ops.stacked_block_output(4)
    paired = transformer_compute._train_step_check(correct, 5.0, 5.0, 4)
    assert paired["warning"] is not None
    assert paired["score_invalid"] is False


def test_a_nan_loss_still_invalidates_and_is_reported_first():
    """A run with both problems says the more serious one first."""
    paired = transformer_compute._train_step_check(float("nan"), 5.0, 5.0, 4)
    assert paired["score_invalid"] is True
    assert paired["warning"].index("NaN") < paired["warning"].index("unchanged")


def test_the_train_step_samples_a_parameter_before_and_after():
    code = sourcecheck.function_code(transformer_compute.run_train_step)
    assert 'sampled = params [ 0 ] [ "w2" ]' in code
    assert '"parameter_moved"' in code
    # Sampled before the warm-up step, or the first update would be missed.
    assert code.index("before = transformer_ops . read_back") < code.index(
        "warm = one_step ( )")


# -- a cache write that could not be seen ------------------------------------

def test_the_cache_and_the_entry_are_different_values():
    """They were both ones, so every write copied ones into ones.

    The only check on this kernel reads an element back, and that element
    read 1.0 for a working ring, a ring that never wrote, and a graph the
    compiler had elided entirely.

    Which matters more here than anywhere else: this is the workload that
    established XLA has no in-place write, after the index_copy_ version
    cost 455 ms to move 32 MiB. The whole finding is about whether the
    write reaches the cache, and nothing verified that it does.
    """
    code = sourcecheck.function_code(llm_inference.run_cache_churn)
    assert "CACHE_FILL , ENTRY_FILL = 1.0 , 2.0" in code
    assert "cache_k = torch . ones (" not in code
    assert "entry = torch . ones (" not in code


def test_a_cache_still_holding_its_own_fill_invalidates_the_score():
    code = sourcecheck.function_code(llm_inference.run_cache_churn)
    assert "observed == CACHE_FILL" in code
    assert '"score_invalid" ] = True' in code
    assert "wrote nothing the device kept" in code


def test_the_write_check_does_not_replace_a_nan_message():
    """Replacing it would trade the more serious finding for the more
    specific one."""
    code = sourcecheck.function_code(llm_inference.run_cache_churn)
    marker = code.index("observed == CACHE_FILL")
    after = code[marker:marker + 400]
    # The existing message has to be an input to the new one. Matching the
    # separator literal is what the first version did -- and it looked for
    # `" ; " . join`, spacing the tokenizer does not insert inside a string
    # literal. A check about what the code does, not how it is spaced.
    assert ". join (" in after, "the message is assigned, not joined"
    assert 'result . get ( "warning" )' in after, (
        "the existing message is not an input to the joined one")


def test_the_bandwidth_warning_still_yields_to_it():
    """A run that wrote nothing should not also be lectured about
    bandwidth -- the bytes it reports were never written."""
    code = sourcecheck.function_code(llm_inference.run_cache_churn)
    assert code.index("observed == CACHE_FILL") < code.index(
        "verify_memory_bound ( bytes_per_s )")


# -- a scatter that landed nowhere -------------------------------------------

def test_a_zero_router_output_is_the_untouched_destination():
    """moe_router builds torch.zeros_like and scatters into it.

    A dispatch that routed nothing, scattered to the wrong positions, or
    was elided leaves that tensor untouched -- and output_check passed
    it, because 0.0 is a number.
    """
    message = transformer_ops.verify_scatter_landed(0.0, "router output")
    assert message is not None
    assert "exactly zero" in message
    assert "scattered nothing" in message


def test_a_real_router_output_says_nothing():
    """The expert matmul sums `hidden` terms of a non-zero input, so any
    element the dispatch touches is far from zero."""
    assert transformer_ops.verify_scatter_landed(51.2) is None
    assert transformer_ops.verify_scatter_landed(-51.2) is None


def test_a_nan_and_an_unreadable_scatter_are_still_caught():
    assert "NaN" in transformer_ops.verify_scatter_landed(float("nan"))
    assert "could not be read" in transformer_ops.verify_scatter_landed(None)


def test_moe_router_uses_the_scatter_check():
    code = sourcecheck.function_code(inference_mix.run_moe_router)
    assert "scatter_check" in code
    assert "output_check" not in code


def test_the_loss_and_the_parameter_verdict_constrain_each_other():
    """Either both move or neither does, and a row showing one alone
    means one of the two checks is measuring the wrong thing.

    Inputs are ones and parameters are 1/fan_in, so a forward pass
    through `layers` blocks gives stacked_block_output(layers) and the
    loss is its mean over identical elements. That is the value on the
    first step; it holds on later steps only if the parameters have not
    moved -- which is exactly what the other verdict reports.
    """
    expected = transformer_ops.stacked_block_output(4)

    # Consistent: nothing moved, and the loss is where a static model
    # leaves it.
    steady = transformer_compute._train_step_check(expected, 5.0, 5.0, 4)
    assert steady["score_invalid"] is False
    assert "unchanged" in steady["warning"]

    # Consistent the other way: the model moved and the loss moved with
    # it, so neither verdict fires.
    trained = transformer_compute._train_step_check(
        expected * 0.5, 5.0, 5.1, 4)
    assert trained["score_invalid"] is True   # the loss check, correctly
    assert "unchanged" not in (trained["warning"] or "")


def test_the_quantized_answer_follows_from_K_and_the_scale():
    """4096 terms of all-ones int8, dequantised by 0.02, is 81.92."""
    from kernels import inference_mix as mix
    problem = PROBLEMS["quantized_gemm"]
    k = problem["shape"][2]
    assert k * mix.SCALE == pytest.approx(81.92)
    assert transformer_ops.verify_equals(81.92, k * mix.SCALE, "q") is None
    for wrong in (0.0, 4096.0, 81.92 * 4, 40.96):
        assert transformer_ops.verify_equals(wrong, k * mix.SCALE, "q")


def test_the_scale_is_named_once():
    """It was a literal in the kernel and would have been a second
    literal in the expected value -- two places to change one number."""
    from kernels import inference_mix as mix
    code = sourcecheck.function_code(mix.run_quantized_gemm)
    assert "0.02" not in code
    assert "SCALE" in code


def test_serving_mix_checks_its_depth_and_would_have_caught_its_own_defect():
    """A scheduler running one block per "request" was the defect here.

    One block leaves 2.84 where thirty-two leave 59.92, so the depth
    check separates them by a factor of twenty -- and the finiteness
    check that was there could not tell them apart at all.
    """
    code = sourcecheck.flat_function_code(inference_mix.run_serving_mix)
    assert "stack_check ( transformer_ops . read_back ( sink ) , layers )" in code
    assert '"expected_output"' in code

    one_block = transformer_ops.stacked_block_output(1)
    full = transformer_ops.stacked_block_output(32)
    assert full / one_block > 20
    # And the check rejects the shallow answer against the deep expectation.
    assert transformer_ops.verify_stack_computed_its_depth(one_block, 32)


def test_the_depth_check_takes_layers_not_a_label():
    """stack_check's second argument is a depth; output_check's was a
    name. Swapping one for the other without changing the argument would
    have compared against a string.
    """
    import inspect
    signature = inspect.signature(transformer_ops.stack_check)
    assert list(signature.parameters) == ["observed", "layers"]


# -- a chain link that could not change the chain ----------------------------

def test_tanh_saturates_so_the_vector_stage_was_invisible():
    """omni_virus drives four engines in a dependent chain. One link was
    not dependent.

    Both operands were ones, so the matmul produced `tile` -- 8192 at the
    pinned shape -- and tanh saturates to exactly 1.0 for any input above
    about nine. The vector stage's `* 1.0001 + 0.5` therefore changed the
    output by zero: deleting the line would have given bit-identical
    results.
    """
    saturated = math.tanh(8192.0)
    with_vector = math.tanh(8192.0 * 1.0001 + 0.5)
    assert saturated == with_vector == 1.0


def test_the_scaled_chain_carries_the_vector_stage():
    """1.0 in, 1.5001 after the vector stage, and tanh separates them."""
    scaled = math.tanh(1.0 * 1.0001 + 0.5)
    without = math.tanh(1.0)
    assert abs(scaled - without) / without > 0.15


def test_the_chain_output_follows_from_the_tile():
    """0.905166 * tile * (tile + 1) / 2, and nothing else."""
    for tile in (2048, 4096, 8192):
        expected = math.tanh(1.5001) * tile * (tile + 1) / 2.0
        assert omni_virus._chain_output(tile) == pytest.approx(expected)


def test_the_bf16_cast_of_the_cumsum_averages_out():
    """The cumsum runs in fp32 and is cast once; the closing matmul
    accumulates in fp32. So per-element rounding does not accumulate, and
    the tolerance is about correctness rather than arithmetic.
    """
    def bf16(value):
        ulp = 2.0 ** (math.floor(math.log2(abs(value))) - 7)
        return round(value / ulp) * ulp

    tile, per_step = 8192, math.tanh(1.5001)
    walked = sum(bf16(per_step * j) for j in range(1, tile + 1))
    assert abs(walked / omni_virus._chain_output(tile) - 1.0) < 1e-4


def test_omni_virus_scales_its_left_operand_and_checks_the_answer():
    code = sourcecheck.flat_function_code(omni_virus.run)
    assert "torch . full ( ( tile , tile ) , 1.0 / tile" in code
    assert "lhs = torch . ones (" not in code
    assert "equals_check" in code
    assert "output_check" not in code


def test_read_back_detaches_before_taking_a_scalar():
    """transformer_train_step samples a parameter, and a parameter carries
    requires_grad=True.

    torch warned "Converting a tensor with requires_grad=True to a scalar
    may lead to unexpected behavior" on trn1.2xlarge 2026-09-10. The read
    was correct and the warning was right to fire -- taking a scalar off
    the autograd graph is exactly what read_back does, and saying so is
    the difference between meaning it and getting away with it.

    It only appears on hardware, so pytest's warnings-as-errors would
    never have caught it; it was caught by reading a verification log.
    """
    code = sourcecheck.flat_function_code(transformer_ops.read_back)
    assert "detach" in code


def test_read_back_still_works_on_a_plain_object():
    """detach() is guarded, because read_back is also handed values from
    kernels that return something without one."""
    class Flat:
        def reshape(self, _):
            return [2.5]

    assert transformer_ops.read_back(Flat()) == 2.5
    assert transformer_ops.read_back(None) is None


def test_decode_multiplies_every_weight_its_flop_count_charges_for():
    """decode_step_flops counts q, k, v and o projections and a two-matrix
    MLP. run_decode used q and o only until 2026-09-11: it read 10 h^2
    weights per layer where a decode reads 12, while the implied_tflops
    beside it charged for 12."""
    code = sourcecheck.function_code(llm_inference.run_decode)
    for weight in ("q", "k", "v", "o", "w1", "w2"):
        assert f'params [ "{weight}" ]' in code or f'params["{weight}"]' in code, weight


def test_the_new_tokens_key_and_value_reach_the_output():
    """Computed but unused, k and v would be dead code the compiler drops --
    projections charged for and never run. Both must feed the attention."""
    code = sourcecheck.flat_function_code(llm_inference.run_decode)
    assert "k_new" in code[code.index("fresh ="):code.index("probs =")]
    assert "v_new" in code[code.index("attended ="):code.index("state = state +")]
