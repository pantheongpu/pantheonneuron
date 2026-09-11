"""End-to-end orchestrator behaviour."""

import json
import os
import pathlib

import pytest

import pantheon_neuron
import sourcecheck
from kernels import nki_backend, registry
from neuron_device import NeuronDevice


TRN1 = [NeuronDevice(i, "trn1", "v2", 2, 32 * 1024**3, True) for i in range(2)]
INF2 = [NeuronDevice(i, "inf2", "v2", 2, 32 * 1024**3, False) for i in range(2)]


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("PANTHEON_NEURON_MOCK", "1")
    yield
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)


def _workload(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


def test_skipped_workload_reports_reason(mock_env):
    """Training work is genuinely unavailable on Inferentia."""
    row = pantheon_neuron.run_workload(
        _workload("transformer_train_step"), INF2, duration=1, monitor_period=0.01
    )
    assert row["Status"] == "SKIPPED"
    assert "training" in row["Detail"]


def test_collectives_are_not_skipped_on_inferentia(mock_env):
    """inf2 has NeuronLink; all_reduce must actually run there."""
    row = pantheon_neuron.run_workload(
        _workload("all_reduce"), INF2, duration=1, monitor_period=0.01
    )
    assert row["Status"] == "PASS"


def test_workload_runs_in_mock_mode(mock_env):
    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.01
    )
    assert row["Status"] == "PASS"
    assert row["Telemetry"]["samples"] > 0


def test_unimplemented_workload_does_not_silently_pass_on_hardware(monkeypatch):
    """A workload without a kernel must never be reported as a successful run.

    Every registry workload now has one, so this uses a synthetic workload
    instead of scanning for a gap. The invariant is about the next workload
    someone adds, not about today's registry: declaring one and forgetting
    the kernel must raise, not PASS with no Score.

    The test has been rewritten twice for the same reason. It first named
    tensor_virus and stopped testing anything the day tensor_virus got a
    kernel; it then scanned for unimplemented workloads and had nothing left
    to scan. A synthetic workload cannot be overtaken by coverage.
    """
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.setattr(
        nki_backend, "require_toolchain", lambda: {"neuronxcc": "2.x"}
    )

    invented = registry.Workload(
        "workload_with_no_kernel", "core", "Declared but never implemented.",
        frozenset({"compute"}), unit="TFLOPS",
    )
    assert invented.name not in pantheon_neuron.IMPLEMENTED

    with pytest.raises(NotImplementedError):
        pantheon_neuron._execute(invented, TRN1, duration=1)


def test_every_registry_workload_has_a_kernel():
    """Coverage is complete; a new workload must arrive with an implementation.

    This is the other half of the test above. That one says an unknown
    workload raises; this one says no workload in the registry is unknown,
    so the raise is unreachable in practice rather than merely unreached.
    """
    missing = sorted(
        w.name for w in registry.WORKLOADS
        if w.name not in pantheon_neuron.IMPLEMENTED
    )
    assert not missing, f"workloads declared without a kernel: {missing}"


def test_implemented_set_matches_what_execute_dispatches():
    """IMPLEMENTED is a claim about _execute; keep it honest.

    A name listed here but not dispatched would make the test above skip a
    workload that silently raises in production.
    """
    source = pathlib.Path(pantheon_neuron.__file__).read_text(encoding="utf-8")
    body = source.split("def _execute(", 1)[1].split("\ndef ", 1)[0]
    # Matches the name however it is dispatched -- a direct comparison or
    # membership in a tuple, since one kernel can serve several workloads.
    for name in pantheon_neuron.IMPLEMENTED:
        assert f'"{name}"' in body, name


def test_execution_errors_flip_a_pass_to_fail(mock_env, monkeypatch):
    monkeypatch.setattr(
        pantheon_neuron.neuron_monitor.NeuronMonitor,
        "stop",
        lambda self: {"samples": 3, "execution_errors": 2},
    )
    row = pantheon_neuron.run_workload(
        _workload("tensor_virus"), TRN1, duration=1, monitor_period=0.01
    )
    assert row["Status"] == "FAIL"
    assert "execution error" in row["Detail"]


def test_report_is_written_atomically(mock_env, tmp_path, monkeypatch):
    monkeypatch.setattr(pantheon_neuron, "DATABASE_DIR", str(tmp_path))
    snapshot = pantheon_neuron.get_system_snapshot(TRN1)
    path = pantheon_neuron.write_report(snapshot, [{"Test Name": "x"}], "runid")

    assert os.path.exists(path)
    assert not os.path.exists(f"{path}.tmp")
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle)["run_id"] == "runid"


def test_cli_list_exits_clean(capsys):
    assert pantheon_neuron.main(["--list"]) == 0
    assert "tensor_virus" in capsys.readouterr().out


def test_cli_reports_missing_hardware(monkeypatch, capsys):
    monkeypatch.delenv("PANTHEON_NEURON_MOCK", raising=False)
    monkeypatch.setattr(
        pantheon_neuron.neuron_device.shutil, "which", lambda _: None
    )
    assert pantheon_neuron.main(["--test", "baseline_metrics"]) == 2
    assert "No Neuron devices" in capsys.readouterr().err


def test_cli_rejects_unknown_test(mock_env, capsys):
    assert pantheon_neuron.main(["--test", "nope"]) == 2
    assert "Unknown test" in capsys.readouterr().err


def test_full_mock_run_writes_clean_report(mock_env, tmp_path, monkeypatch):
    monkeypatch.setattr(pantheon_neuron, "DATABASE_DIR", str(tmp_path))
    assert pantheon_neuron.main(
        ["--test", "baseline_metrics", "--duration", "1", "--monitor-period", "0.01"]
    ) == 0

    reports = list(tmp_path.glob("*.json"))
    assert len(reports) == 1
    blob = reports[0].read_text().lower()
    for forbidden in ("instance_id", "hostname", "availability_zone"):
        assert forbidden not in blob


# -- --duration is a claim about the window that was measured ----------------

def test_a_full_length_window_says_nothing():
    assert pantheon_neuron.short_window(29.4, 30) is None
    assert pantheon_neuron.short_window(30.0, 30) is None
    assert pantheon_neuron.short_window(60.0, 30) is None


def test_a_short_window_names_the_flag_that_did_not_bound_it():
    """The allocation_fragmentation finding, made general.

    10,000 pinned allocations finish in about four seconds on trn1
    whatever --duration says. Three repeats scattered from cv 0.15 to cv
    0.98 and raising the duration never helped, because the flag was not
    connected to the window.
    """
    message = pantheon_neuron.short_window(4.1, 30)
    assert message is not None
    assert "4.1s" in message and "30s" in message
    assert "--duration" in message


def test_the_boundary_is_the_declared_fraction():
    """Half, not something tighter: warm-up and a final wait are real."""
    requested = 20
    edge = requested * pantheon_neuron.SHORT_WINDOW_FRACTION
    assert pantheon_neuron.short_window(edge, requested) is None
    assert pantheon_neuron.short_window(edge - 0.01, requested) is not None


def test_a_kernel_that_reports_no_window_is_not_accused():
    """Absent is not short. nccom-test runs its own iteration count."""
    assert pantheon_neuron.short_window(None, 30) is None


def test_a_zero_duration_cannot_be_a_fraction_of_itself():
    assert pantheon_neuron.short_window(0.5, 0) is None


def test_a_kernel_that_reports_bounded_by_is_not_told_twice():
    """The general check is a floor, not a second opinion.

    allocation_fragmentation reports ``bounded_by`` itself, in terms
    specific to its allocation count. On trn1.2xlarge 2026-09-10 its row
    came back carrying both sentences: the kernel's, and the
    orchestrator's saying the same thing in different words. The first
    guard compared the two message texts for equality, which was never
    the question being asked.
    """
    code = sourcecheck.function_code(pantheon_neuron._measure_once)
    assert '"bounded_by" not in run_result' in code


# -- an invalid Score must not depend on the kernel explaining itself --------

def _row_for(result, monkeypatch):
    """Run one workload with a canned kernel result."""
    workload = next(w for w in registry.WORKLOADS
                    if w.name == "allocation_fragmentation")
    monkeypatch.setattr(pantheon_neuron, "_execute",
                        lambda *a, **k: 1234.5)
    monkeypatch.setitem(pantheon_neuron._LAST_RUN, workload.name, result)
    devices = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]
    return pantheon_neuron._measure_once(workload, devices, 1, 0.5)


def test_an_invalid_score_with_no_warning_still_fails(monkeypatch):
    """It used to pass, because the check was nested under the warning.

    Invalidating a Score and explaining why are separate decisions, and
    the invalidation depended on the kernel happening to do both. Nothing
    had hit it because every kernel setting score_invalid also set a
    warning -- memory_agg's zero-overlap case computes the two
    independently and would have been the first.
    """
    row = _row_for({"score_invalid": True, "elapsed_s": 1.0}, monkeypatch)
    assert row["Status"] == "FAIL"
    assert row["Score"] is None
    assert "invalid" in row["Detail"]


def test_an_invalid_score_with_a_warning_keeps_the_warning(monkeypatch):
    row = _row_for(
        {"score_invalid": True, "warning": "workers never overlapped",
         "elapsed_s": 1.0}, monkeypatch)
    assert row["Status"] == "FAIL"
    assert row["Score"] is None
    assert "never overlapped" in row["Detail"]


def test_a_warning_without_invalidation_still_passes(monkeypatch):
    """The control: a warning is not a failure, or every row would fail."""
    row = _row_for(
        {"warning": "workers overlapped for only 19% of the span",
         "elapsed_s": 1.0}, monkeypatch)
    assert row["Status"] == "PASS"
    assert row["Score"] == 1234.5
    assert "19%" in row["Detail"]


# -- workers that need cores run before this process holds them --------------

def _names(workloads):
    return [w.name for w in workloads]


def test_aggregates_run_before_any_in_process_workload():
    """--test all ran tensor_virus first, the runtime held both cores, and
    both memory_*_agg workers aborted -6 on trn1.2xlarge 2026-09-10."""
    ordered = _names(pantheon_neuron.run_order(registry.resolve("all")))
    first_in_process = min(
        i for i, name in enumerate(ordered)
        if name not in ("baseline_metrics", "memory_read_agg", "memory_write_agg"))
    assert ordered.index("memory_read_agg") < first_in_process
    assert ordered.index("memory_write_agg") < first_in_process


def test_baseline_telemetry_stays_first():
    assert _names(pantheon_neuron.run_order(registry.resolve("all")))[0] == "baseline_metrics"


def test_ordering_neither_drops_nor_duplicates():
    selected = registry.resolve("all")
    ordered = pantheon_neuron.run_order(selected)
    assert sorted(_names(ordered)) == sorted(_names(selected))
    assert len(ordered) == len(selected) > 0


def test_the_memory_suite_runs_its_aggregates_first():
    """The cheaper reproduction: --test memory put memory_read ahead of
    the aggregates, which is the same hazard with fewer workloads."""
    ordered = _names(pantheon_neuron.run_order(registry.resolve("memory")))
    assert ordered[:2] == ["memory_read_agg", "memory_write_agg"], ordered


def test_a_selection_without_aggregates_keeps_registry_order():
    selected = registry.resolve("core")
    assert _names(pantheon_neuron.run_order(selected)) == _names(selected)


def test_main_runs_the_ordered_selection():
    code = sourcecheck.flat_function_code(pantheon_neuron.main)
    assert code.index("workloads = run_order ( workloads )") < code.index(
        "for position , workload in enumerate ( workloads )")


# -- the profiler's core, reserved once the aggregates are done --------------

def test_the_reservation_waits_for_the_aggregates():
    """--test all used to turn the reservation off for the whole run, so
    memory_read and memory_write always fell back to analytic."""
    ordered = pantheon_neuron.run_order(registry.resolve("all"))
    at = pantheon_neuron.reservation_point(ordered)
    names = [w.name for w in ordered]
    assert names.index("memory_read_agg") < at and names.index("memory_write_agg") < at
    # From there on nothing declares cores: all, so the reservation holds.
    aggregate, _ = pantheon_neuron.reservation_cost(ordered[at:])
    assert aggregate == []
    assert "memory_read" in names[at:] and "memory_write" in names[at:]


def test_without_aggregates_the_reservation_comes_first():
    ordered = pantheon_neuron.run_order(registry.resolve("core"))
    assert pantheon_neuron.reservation_point(ordered) == 0


def test_a_selection_of_only_aggregates_reserves_nothing():
    ordered = [w for w in registry.WORKLOADS if (w.problem or {}).get("cores") == "all"]
    assert ordered and pantheon_neuron.reservation_point(ordered) is None


def test_main_reserves_inside_the_loop_at_that_point():
    code = sourcecheck.flat_function_code(pantheon_neuron.main)
    assert "reserve_at = reservation_point ( workloads )" in code
    assert code.index("for position , workload in enumerate ( workloads )") < code.index(
        "reserve_profiler_core ( devices , workloads [ position : ] )")
