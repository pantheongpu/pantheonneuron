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
    """
    import inspect

    module, function, _ = FAMILY_ENTRY_POINTS[name]
    source = inspect.getsource(getattr(module, function))
    assert "transformer_ops.output_check(" in source, name
    assert "read_back" in source, name
    assert '"warning": transformer_ops.verify_output_is_a_number' not in source, (
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
