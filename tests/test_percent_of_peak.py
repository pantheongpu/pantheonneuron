"""A Score against the ceiling its part publishes.

26.1 TFLOPS and 66.3 TFLOPS are two numbers. 12% and 31% of peak are a
finding: the first pair invites a cross-vendor comparison, and the second
says whether that comparison would be about silicon or about kernel
quality.

The denominator is the hard part, and this file is mostly about getting
it right rather than about the division.
"""

import pytest

import pantheon_neuron
import sourcecheck
from kernels import registry
from neuron_device import NeuronDevice

TRN1 = [NeuronDevice(0, "trn1", "v2", 2, 32 * 1024**3, True)]
INF2 = [NeuronDevice(0, "inf2", "v2", 2, 32 * 1024**3, False)]


def _named(name):
    return next(w for w in registry.WORKLOADS if w.name == name)


# -- the published figures ---------------------------------------------------

def test_every_peak_cites_a_source():
    """A number in a table without provenance is a number with a table's
    authority and a comment's evidence. That is what this replaced."""
    assert registry.PART_PEAKS, "no peaks declared -- the sweep is vacuous"
    for arch, peak in registry.PART_PEAKS.items():
        assert peak.get("source"), arch
        assert len(peak["source"]) > 30, f"{arch}: source is not a citation"
        for field in ("hbm_gbps", "bf16_tflops", "neuroncores"):
            assert isinstance(peak.get(field), (int, float)), (arch, field)
            assert peak[field] > 0, (arch, field)


def test_every_peak_is_verified_against_the_architecture_docs():
    """Checked 2026-09-10 against the AWS Neuron architecture pages.

    An earlier version of this test asserted the opposite -- that nothing
    was verified -- and was the signal to go and read the documents. It
    now asserts the citation is present and names the source it was read
    from, so a figure edited without a new citation fails.
    """
    for arch, peak in registry.PART_PEAKS.items():
        assert peak["verified"] is True, arch
        assert "AWS Neuron architecture docs" in peak["source"], arch
        assert "2026-09-10" in peak["source"], arch


def test_the_two_parts_are_the_same_silicon_per_chip():
    """They are, and the suspicion that prompted this table was backwards.

    kernels/memory_read.py carried "the part's ~820 GB/s HBM" in prose. I
    took that for an Inferentia2 figure wrongly applied to Trainium1 and
    replaced it with 613, derived by dividing the instance page's 9.8
    TB/s by 16 chips.

    The architecture docs give both parts, in identical words: two
    NeuronCore-v2, 32GiB HBM at 820 GiB/sec, 190 FP16/BF16/cFP8/TF32
    TFLOPS. The prose was right; the correction was wrong; and dividing
    by the smaller ceiling reported the bandwidth kernels at 83-88% of
    peak when they reach about 60%.

    The repo already knew this from the other direction -- both chips
    report NeuronCore-v2 -- which is what should have made the suspicion
    suspicious.
    """
    trn1, inf2 = registry.PART_PEAKS["trn1"], registry.PART_PEAKS["inf2"]
    for field in ("neuroncores", "hbm_gibps", "hbm_gbps", "bf16_tflops"):
        assert trn1[field] == inf2[field], field


def test_the_bandwidth_ceiling_is_converted_out_of_gibibytes():
    """The doc says 820 GiB/sec; every Score here is bytes / 1e9.

    Treating the two as interchangeable is a 7% error in every bandwidth
    percentage, applied silently and in the flattering direction.
    """
    for arch, peak in registry.PART_PEAKS.items():
        assert peak["hbm_gibps"] == 820.0, arch
        expected = 820.0 * (1 << 30) / 1e9
        assert peak["hbm_gbps"] == pytest.approx(expected, abs=0.1), arch
        assert peak["hbm_gbps"] > peak["hbm_gibps"], (
            f"{arch}: the decimal figure must exceed the binary one")


def test_the_ceiling_reconciles_with_the_instance_pages_for_compute():
    """16 chips x 190 TFLOPS is 3.04 PFLOPS against trn1.32xlarge's
    "up to 3 petaflops"; 12 x 190 is 2.28 against inf2.48xlarge's 2.3.

    Bandwidth does not reconcile, and that is why the architecture page
    is the source rather than the instance page: both instance pages
    claim "9.8 TB/s of total memory bandwidth", which is 12 chips'
    worth. See the note in registry.PART_PEAKS.
    """
    per_chip = registry.PART_PEAKS["trn1"]["bf16_tflops"]
    assert 16 * per_chip / 1000 == pytest.approx(3.04, abs=0.05)
    assert 12 * per_chip / 1000 == pytest.approx(2.28, abs=0.05)


# -- the share, which is the part that is easy to get wrong ------------------

def test_a_single_core_workload_is_measured_against_one_core():
    """memory_read declares cores: 1 on a two-core part.

    Comparing it against the whole chip is the error that makes 256 GB/s
    look like 42% of the part when it is 84% of what it was given.
    """
    share = pantheon_neuron.peak_share(_named("memory_read"), TRN1)
    assert share["cores_used"] == 1
    assert share["cores_available"] == 2
    assert share["peak"] == pytest.approx(
        registry.PART_PEAKS["trn1"]["hbm_gbps"] / 2)


def test_an_all_core_workload_is_measured_against_the_chip():
    share = pantheon_neuron.peak_share(_named("memory_read_agg"), TRN1)
    assert share["cores_used"] == 2
    assert share["peak"] == pytest.approx(
        registry.PART_PEAKS["trn1"]["hbm_gbps"])


def test_the_two_memory_workloads_are_not_flattered_by_their_core_count():
    """memory_read and memory_read_agg measure the same thing on the same
    silicon and differ only in how much of it they are allowed.

    A percentage ignoring that makes the aggregate look better for a
    reason that has nothing to do with memory. Measured on trn1.2xlarge
    2026-09-10: 256.1 and 541.5 GB/s, which land within six points of
    each other once each is read against its own share.
    """
    single = pantheon_neuron.percent_of_peak(
        256.1, pantheon_neuron.peak_share(_named("memory_read"), TRN1))
    aggregate = pantheon_neuron.percent_of_peak(
        541.5, pantheon_neuron.peak_share(_named("memory_read_agg"), TRN1))
    assert abs(single - aggregate) < 10, (single, aggregate)


def test_the_peak_is_looked_up_by_architecture():
    """This asserted the two parts get *different* denominators, which
    was my wrong premise rather than a property of the code.

    They publish identical per-chip figures, so the same workload gets
    the same ceiling on both -- and what is worth testing is that the
    lookup happens at all, which an unknown architecture proves by
    getting nothing.
    """
    trn1 = pantheon_neuron.peak_share(_named("memory_read"), TRN1)
    inf2 = pantheon_neuron.peak_share(_named("memory_read"), INF2)
    assert trn1["peak"] == inf2["peak"], "same silicon, same ceiling"
    assert trn1["peak_source"] != inf2["peak_source"], (
        "each part must cite its own document even when the figures agree")


def test_multiple_devices_scale_the_ceiling():
    two = [NeuronDevice(i, "trn1", "v2", 2, 32 * 1024**3, True)
           for i in range(2)]
    one = pantheon_neuron.peak_share(_named("memory_read_agg"), TRN1)
    both = pantheon_neuron.peak_share(_named("memory_read_agg"), two)
    assert both["peak"] == pytest.approx(one["peak"] * 2)


# -- where there is no ceiling to divide by ----------------------------------

def test_a_unit_with_no_published_peak_gets_no_percentage():
    """graph-steps/s and requests/s have no datasheet figure, and
    inventing one would be worse than an empty column."""
    for name in ("graph_replay", "serving_mix", "allocation_fragmentation"):
        assert pantheon_neuron.peak_share(_named(name), TRN1) is None


def test_an_unknown_architecture_gets_no_percentage():
    alien = [NeuronDevice(0, "trn9", "v9", 4, 32 * 1024**3, True)]
    assert pantheon_neuron.peak_share(_named("memory_read"), alien) is None


def test_no_devices_means_no_share():
    assert pantheon_neuron.peak_share(_named("memory_read"), []) is None


# -- the division ------------------------------------------------------------

def test_the_percentage_is_the_score_over_the_share():
    share = pantheon_neuron.peak_share(_named("memory_read_agg"), TRN1)
    assert pantheon_neuron.percent_of_peak(share["peak"], share) == 100.0
    assert pantheon_neuron.percent_of_peak(
        share["peak"] / 2, share) == pytest.approx(50.0)


def test_an_absent_or_impossible_score_gets_no_percentage():
    share = pantheon_neuron.peak_share(_named("memory_read"), TRN1)
    for bad in (None, 0, -1.0, "n/a"):
        assert pantheon_neuron.percent_of_peak(bad, share) is None
    assert pantheon_neuron.percent_of_peak(100.0, None) is None


def test_a_score_above_peak_is_reported_rather_than_clamped():
    """Over 100% means the Score, the peak, or the core split is wrong,
    and clamping it to 100 would hide exactly that.

    memory_read once reported 14,513 GB/s from an elided DMA. A column
    reading 2367% is the loudest possible way to say so.
    """
    share = pantheon_neuron.peak_share(_named("memory_read"), TRN1)
    assert pantheon_neuron.percent_of_peak(
        share["peak"] * 5, share) == pytest.approx(500.0)


# -- the row -----------------------------------------------------------------

def test_the_row_carries_the_percentage_and_its_provenance(mock_env=None):
    code = sourcecheck.flat_function_code(pantheon_neuron._measure_once)
    assert '"Percent Of Peak"' in code
    assert '"Peak"' in code
    # Both on the skipped row too, or the shapes diverge.
    assert code.count('"Percent Of Peak"') == 2


def test_the_console_prints_the_share_and_flags_an_unverified_peak(capsys):
    """A reader who never opens the JSON is exactly the reader who would
    quote a Score against another vendor's, so the share belongs on the
    console -- and so does the fact that the denominator is unchecked.
    """
    import io
    import contextlib

    row = {
        "Status": "PASS", "Detail": "", "Unit": "GB/s",
        "Percent Of Peak": 88.3,
        "Peak": {"peak": 613.0, "cores_used": 2, "cores_available": 2,
                 "peak_verified": False},
        "Repeats": None,
    }
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        pct, share = row["Percent Of Peak"], row["Peak"]
        caveat = "" if share.get("peak_verified") else ", peak unverified"
        print(f"[PANTHEON-NEURON]    {pct}% of {share['peak']} {row['Unit']} "
              f"across {share['cores_used']} of "
              f"{share['cores_available']} core(s){caveat}")
    printed = buffer.getvalue()
    assert "88.3% of 613.0 GB/s" in printed
    assert "2 of 2 core(s)" in printed
    assert "peak unverified" in printed


def test_the_console_block_exists_in_the_run_loop():
    """The test above exercises the arithmetic; this asserts the loop
    actually does it, which is the half a copy of the code cannot show."""
    source = sourcecheck.flat_function_code(pantheon_neuron.main)
    assert '"Percent Of Peak"' in source
    assert "peak unverified" in source


def test_a_verified_peak_would_drop_the_caveat():
    """So the caveat disappears on its own when someone checks the
    datasheet, rather than needing a second edit to remove."""
    source = sourcecheck.flat_function_code(pantheon_neuron.main)
    assert 'if share . get ( "peak_verified" )' in source


# -- a workload that idles by design -----------------------------------------

def test_a_duty_cycled_workload_is_measured_against_its_duty():
    """pulse_virus idles half its run by design and its Score is averaged
    over the whole run, idle halves included.

    Against the full ceiling it read 7.29% of peak where tensor_virus
    read 13.74% -- exactly half, while running the same kernel at the
    same rate during its loaded halves. The column would have told a
    reader the pulsed kernel is half as efficient, which is the opposite
    of what the two numbers show.
    """
    share = pantheon_neuron.peak_share(_named("pulse_virus"), TRN1)
    full = pantheon_neuron.peak_share(_named("tensor_virus"), TRN1)
    duty = _named("pulse_virus").problem["duty_cycle"]
    assert share["peak"] == pytest.approx(full["peak"] * duty)
    assert share["duty_cycle"] == duty


def test_the_pulsed_and_sustained_kernels_now_agree():
    """The check that the correction is right rather than merely applied.

    Same kernel, same rate while loaded, so once each is read against
    what it could reach they should land together. Measured on
    trn1.2xlarge 2026-09-10: pulse_virus 13.85 TFLOPS, tensor_virus
    26.1.
    """
    pulsed = pantheon_neuron.percent_of_peak(
        13.85, pantheon_neuron.peak_share(_named("pulse_virus"), TRN1))
    sustained = pantheon_neuron.percent_of_peak(
        26.1, pantheon_neuron.peak_share(_named("tensor_virus"), TRN1))
    assert abs(pulsed - sustained) < 2.0, (pulsed, sustained)


def test_a_workload_with_no_duty_cycle_gets_the_full_ceiling():
    share = pantheon_neuron.peak_share(_named("tensor_virus"), TRN1)
    assert share["duty_cycle"] is None
    assert share["peak"] == pytest.approx(
        registry.PART_PEAKS["trn1"]["bf16_tflops"])


# -- units the docs do not give a ceiling for --------------------------------

def test_an_integer_workload_gets_no_percentage_and_that_is_deliberate():
    """int_virus reports TOPS for uint8, and the architecture docs quote
    FP16/BF16/cFP8/TF32 and FP32 only.

    Borrowing the bf16 figure for uint8 would divide by a number the
    vendor never claimed for that dtype. An empty column is the honest
    answer until a uint8 figure is published, and this test is what
    stops someone filling it in by analogy.
    """
    assert "TOPS" not in registry.PEAK_FOR_UNIT
    assert pantheon_neuron.peak_share(_named("int_virus"), TRN1) is None
