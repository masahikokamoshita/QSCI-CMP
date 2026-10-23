from qscicmp.cost import enumerated_oracle_t_count, state_preparation_t_count, ucj_rotation_count


def test_ucj_rotation_count_formula():
    for n in (20, 24):
        layers = n // 2
        assert ucj_rotation_count(n, layers) == layers * (5 * n * n // 2 - n // 2)


def test_state_preparation_t_count_matches_paper_runs():
    assert abs(state_preparation_t_count(20, 10) - 393476.133281362) < 1e-6
    assert abs(state_preparation_t_count(24, 12) - 696669.6220556276) < 1e-6


def test_enumerated_oracle_cost():
    # one enumerated determinant, one amplification round: (4n - 6) Toffolis = 4(4n - 6) T gates
    assert enumerated_oracle_t_count(20, 1, 1) == 4 * (4 * 20 - 6)
