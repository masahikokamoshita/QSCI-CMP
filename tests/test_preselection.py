from qscicmp.preselection import (
    excitation_level, hf_index, hierarchy_number, preselection_e2, preselection_h20, preselection_h25, seniority,
)


def test_counts_match_paper_runs():
    # |T(R)| for the 20-qubit H10 system, as recorded in the paper's runs
    assert len(preselection_e2(20, 10)) == 2126
    assert len(preselection_h20(20, 10)) == 3026
    assert len(preselection_h25(20, 10)) == 6626
    # and for the 24-qubit H2O (10e, 12o) system
    assert len(preselection_e2(24, 10)) == 4236
    assert len(preselection_h20(24, 10)) == 6126
    assert len(preselection_h25(24, 10)) == 16206


def test_hierarchy_thresholds():
    n_qubits, n_electrons = 20, 10
    hf = hf_index(n_electrons)
    for index in preselection_h20(n_qubits, n_electrons):
        assert hierarchy_number(index, hf, 10) <= 2.0
    for index in preselection_h25(n_qubits, n_electrons):
        assert hierarchy_number(index, hf, 10) <= 2.5


def test_hf_reference():
    hf = hf_index(10)
    assert excitation_level(hf, hf) == 0
    assert seniority(hf, 10) == 0
