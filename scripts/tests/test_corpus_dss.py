import numpy as np

from scripts.corpus_dss import dss, logdet_experimental


def test_experimental_logdet_matches_dense_matrix():
    rng = np.random.default_rng(123)
    D = rng.uniform(0.2, 1.5, size=7)
    u = rng.normal(size=7)
    v = rng.normal(size=7)
    E = np.diag(D) + np.outer(u, u) + np.outer(v, v)

    sign, expected = np.linalg.slogdet(E)

    assert sign > 0
    assert np.isclose(logdet_experimental(D, u, v), expected, rtol=1e-12)


def test_relative_dss_matches_direct_dense_calculation():
    rng = np.random.default_rng(456)
    N = 9
    D = rng.uniform(0.5, 2.0, size=N)
    u = rng.normal(scale=0.2, size=N)
    v = rng.normal(scale=0.2, size=N)
    r = rng.normal(size=N)
    factor = rng.normal(scale=0.15, size=(N, 4))
    S = factor @ factor.T
    E = np.diag(D) + np.outer(u, u) + np.outer(v, v)
    C = E + S

    chi2, logdet_ratio, score = dss(S.astype(np.float32), D, u, v, r)
    expected_chi2 = float(r @ np.linalg.solve(C, r))
    expected_ratio = float(np.linalg.slogdet(C)[1] - np.linalg.slogdet(E)[1])

    assert np.isclose(chi2, expected_chi2, rtol=2e-8)
    assert np.isclose(logdet_ratio, expected_ratio, rtol=2e-8)
    assert np.isclose(score, expected_chi2 + expected_ratio, rtol=2e-8)
