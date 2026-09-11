"""NeuronLink collectives: parsing AWS's benchmark, and the sweep guard.

These are the only workloads that cannot be run at all on hardware this
account can rent -- trn1.32xlarge needs 128 vCPUs against a granted 64 -- so
the parser is written against the documented output format and these tests
are the only thing standing behind it until the quota lands.
"""

import pytest

import pantheon_neuron
from kernels import collectives, registry


SAMPLE = """
nccom-test 2.30.59.0
    size(B)    count    type    time(us)    algbw(GB/s)    busbw(GB/s)
    1048576   262144   fp32       23.50          44.61        44.62
    2097152   524288   fp32       42.10          49.81        49.86
    4194304  1048576   fp32       81.30          51.58        51.57
    8388608  2097152   fp32      160.00          52.43        52.43
Avg bus bandwidth: 49.62 GB/s
"""


def _workload(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


# -- parsing -----------------------------------------------------------------

def test_every_sweep_row_is_parsed():
    rows = collectives.parse_busbw(SAMPLE)
    assert [size for size, _ in rows] == [1048576, 2097152, 4194304, 8388608]
    assert rows[0][1] == 44.62
    assert rows[-1][1] == 52.43


def test_the_summary_line_is_not_mistaken_for_a_row():
    """"Avg bus bandwidth" has no size column and must not become a data point."""
    rows = collectives.parse_busbw(SAMPLE)
    assert all(size >= 1048576 for size, _ in rows)
    assert len(rows) == 4


def test_output_without_rows_parses_to_nothing():
    assert collectives.parse_busbw("nccom-test: no devices found") == []


# -- the sweep guard ---------------------------------------------------------

def test_a_full_sweep_passes():
    rows = collectives.parse_busbw(SAMPLE)
    assert collectives.verify_sweep_covers_both_regimes(
        rows, 1048576, 8388608
    ) is None


def test_a_sweep_missing_small_messages_is_flagged():
    """Small messages are latency-bound; averaging without them misleads."""
    rows = [(4194304, 51.5), (8388608, 52.4)]
    message = collectives.verify_sweep_covers_both_regimes(rows, 1048576, 8388608)
    assert message is not None
    assert "small-message regime" in message


def test_a_sweep_missing_large_messages_is_flagged():
    rows = [(1048576, 44.6), (2097152, 49.8)]
    message = collectives.verify_sweep_covers_both_regimes(rows, 1048576, 8388608)
    assert message is not None
    assert "bandwidth-bound regime" in message


def test_an_empty_sweep_is_flagged():
    assert collectives.verify_sweep_covers_both_regimes([], 1, 2) is not None


# -- how they are declared ---------------------------------------------------

def test_they_are_scored_from_nccom_test():
    """The fourth Score source, and the only external benchmark."""
    for name in ("all_reduce", "p2p_thrasher"):
        assert _workload(name).score_source.source == registry.NCCOM
        assert _workload(name).unit == "GB/s"


def test_they_need_two_devices_and_skip_on_one():
    """Every part this account can rent has one device, so these skip."""
    from neuron_device import NeuronDevice

    single = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]
    for name in ("all_reduce", "p2p_thrasher"):
        workload = _workload(name)
        assert workload.min_devices == 2
        assert workload.skip_reason(single) is not None


def test_a_skipped_collective_still_declares_its_unit():
    """So a comparison shows an explicit gap rather than dropping the row."""
    from neuron_device import NeuronDevice

    single = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]
    row = pantheon_neuron.run_workload(
        _workload("all_reduce"), single, duration=1, monitor_period=0.1
    )
    assert row["Status"] == "SKIPPED"
    assert row["Unit"] == "GB/s"
    assert row["Score"] is None


def test_both_are_dispatched():
    assert "all_reduce" in pantheon_neuron.IMPLEMENTED
    assert "p2p_thrasher" in pantheon_neuron.IMPLEMENTED


def test_missing_binary_raises_rather_than_returning_zero():
    with pytest.raises(collectives.CollectivesUnavailable):
        collectives._run(["--version"]) if not collectives.available() else (
            pytest.skip("nccom-test is installed here")
        )


# -- a parse that drops rows silently biases the average ---------------------

_MIXED = """\
#      size    count   type   time   algbw   busbw
       1024      256   fp32   12.1    1.23    2.46
       2048      512   fp32   13.4       3       6
       4096     1024   fp32   15.9    4.56    9.12
"""


def test_a_row_whose_bandwidth_is_an_integer_is_not_parsed():
    """Documenting the gap rather than guessing at a format.

    _ROW requires the trailing field to be `\\d+.\\d+`. This module cannot
    be run on any part this account's quota can reach, so loosening the
    pattern would be guessing -- but the mismatch itself is detectable
    without guessing.
    """
    rows = collectives.parse_busbw(_MIXED)
    assert [size for size, _ in rows] == [1024, 4096]


def test_the_dropped_row_is_counted():
    assert collectives.unparsed_rows(_MIXED) == 1


def test_a_partial_parse_is_reported():
    message = collectives.verify_sweep_was_fully_parsed(_MIXED)
    assert message is not None
    assert "parsed 2 of 3" in message


def test_a_complete_parse_says_nothing():
    """The control: the common case must stay quiet."""
    clean = """\
#      size    count   type   time   algbw   busbw
       1024      256   fp32   12.1    1.23    2.46
       2048      512   fp32   13.4    2.34    4.68
"""
    assert collectives.unparsed_rows(clean) == 0
    assert collectives.verify_sweep_was_fully_parsed(clean) is None


def test_a_header_only_output_is_not_a_dropped_row():
    """Or every run would warn about its own column headings."""
    header = "#      size    count   type   time   algbw   busbw\n"
    assert collectives.unparsed_rows(header) == 0
    assert collectives.verify_sweep_was_fully_parsed(header) is None
