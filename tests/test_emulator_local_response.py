import numpy as np

from coral.analysis.emulator_local_response import (
    _error_stats,
    intervention_footprint,
    local_masks,
    selected_member_names,
)


def test_footprint_is_union_of_static_field_edits():
    baseline = np.zeros((9, 7, 7), dtype=float)
    member = baseline.copy()
    member[0, 3, 3] = 1.0
    member[2, 1, 5] = 2.0
    member[8, :, :] = 4.0  # scalar channel is intentionally ignored
    footprint = intervention_footprint(member, baseline)
    assert footprint.sum() == 2
    assert footprint[3, 3]
    assert footprint[1, 5]


def test_local_masks_are_truth_defined_and_distance_limited():
    footprint = np.zeros((9, 9), dtype=bool)
    footprint[4, 4] = True
    land = np.ones_like(footprint)
    truth = np.zeros_like(footprint, dtype=float)
    baseline = np.zeros_like(truth)
    truth[4, 5] = 0.2
    baseline[4, 3] = 0.2
    truth[4, 4] = 0.03
    neighbourhood, wet, active = local_masks(
        footprint, land, truth, baseline, radius_m=4, cell_m=4,
        wet_threshold=0.1, response_threshold=0.01)
    assert neighbourhood.sum() == 5
    assert wet[4, 5] and wet[4, 3]
    assert active[4, 4]
    assert not neighbourhood[4, 6]


def test_error_stats_report_tail_and_maximum():
    stats = _error_stats(np.array([-1.0, 0.0, 1.0]))
    assert np.isclose(stats["bias_m"], 0.0)
    assert np.isclose(stats["rmse_m"], np.sqrt(2 / 3))
    assert np.isclose(stats["max_abs_m"], 1.0)


def test_external_manifest_selection_excludes_baselines():
    entries = [
        {"name": "Base2016_000", "interventions": []},
        {"name": "Base2016_001", "interventions": [{"kind": "floodwall"}]},
        {"name": "Int2050_001", "interventions": [{"kind": "depave"}]},
    ]
    assert selected_member_names(entries) == ["Base2016_001", "Int2050_001"]
