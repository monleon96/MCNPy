"""Opción 3: devolver a a_5/a_6 la varianza que el promediado sí tiene.

Dos defectos encadenados, medidos en la run 101a y documentados en
`docs/chi2-mf4/between_experiment_floor_design.md` §7.4-7.8:

  * por encima de `mc_order_cap` el MC congela cada réplica al nominal, así que
    el término entre-modelos se anula y lo que queda es jitter de c0 — con
    |rho| = 1 EXACTO entre los órdenes congelados;
  * y el regularizador near-zero marca justo los bins que SÍ se muestrearon
    (99,3 % en a_5) y les dona la sigma de los congelados (79,8 % de donantes).

Las puertas: que apagado sea la identidad, que encendido no invente
correlaciones, y que el regularizador sólo pueda encoger.
"""
import numpy as np

from scripts.exfor_utils import regularize_near_zero_relative_covariance

L = 3


def _cov(n_bins, l_max=L, seed=11, rel=None):
    """Reproduce el reparto real: los ordenes CONGELADOS llevan una sigma
    relativa diminuta (SNR alto -> no marcados -> son los donantes) y los
    MUESTREADOS una grande (SNR bajo -> marcados). Ese es exactamente el
    reparto que hace que el trasplante ocurra."""
    if rel is None:
        rel = np.array([1.5, 1.5, 0.05])[:l_max]      # a_3 hace de congelado
    rel = np.asarray(rel, dtype=float)
    if rel.size != n_bins * l_max:
        rel = np.tile(rel, n_bins)
    rng = np.random.default_rng(seed)
    n = n_bins * l_max
    a = rng.normal(size=(n, n))
    c = a @ a.T / n
    d = np.sqrt(np.diag(c))
    return (c / np.outer(d, d)) * np.outer(rel, rel)


def _means(n_bins, l_max=L):
    # medias pequeñas -> SNR bajo -> marcados
    return np.tile(np.array([0.5, 0.05, 0.02])[:l_max], n_bins)


def test_apagado_es_la_identidad_bit_a_bit():
    """La puerta de inercia: sin las dos entradas nuevas, exactamente lo de antes."""
    cov, mu = _cov(8), _means(8)
    ca = cov * np.outer(mu, mu)
    old, _ = regularize_near_zero_relative_covariance(
        cov, mu, ca, L, snr_threshold=1.0, n_neighbors=3)
    new, _ = regularize_near_zero_relative_covariance(
        cov, mu, ca, L, snr_threshold=1.0, n_neighbors=3,
        frozen_mask=None, mixture_abs_std=None)
    assert np.array_equal(old, new)


def test_los_congelados_no_pueden_ser_donantes():
    """El caso real: el MISMO orden esta congelado en unos bins y muestreado en
    otros. El vecindario camina por orden, asi que los congelados son
    literalmente los donantes de los muestreados — y su sigma es jitter de c0."""
    n_bins = 12
    rel = np.empty((n_bins, L))
    rel[:, 0] = 1.5
    rel[:, 1] = 1.5
    rel[0::2, 2] = 0.02     # a_3 CONGELADO en los bins pares (SNR alto, donante)
    rel[1::2, 2] = 1.8      # a_3 muestreado en los impares (SNR bajo, marcado)
    cov = _cov(n_bins, rel=rel.ravel())
    mu = _means(n_bins)
    ca = cov * np.outer(mu, mu)
    fz = np.zeros(cov.shape[0], dtype=bool)
    fz[np.arange(0, n_bins, 2) * L + 2] = True

    sin, d0 = regularize_near_zero_relative_covariance(
        cov, mu, ca, L, snr_threshold=1.0, n_neighbors=3)
    con, d1 = regularize_near_zero_relative_covariance(
        cov, mu, ca, L, snr_threshold=1.0, n_neighbors=3, frozen_mask=fz)

    assert d1['n_donors_excluded'] == n_bins // 2
    assert d0['n_donors_excluded'] == 0

    sampled = np.arange(1, n_bins, 2) * L + 2
    got_sin = np.sqrt(np.diag(sin))[sampled]
    got_con = np.sqrt(np.diag(con))[sampled]
    assert np.allclose(got_sin, 0.02, atol=1e-9), \
        "sin la guarda, el muestreado hereda el 2 % del congelado"
    assert np.all(got_con > got_sin * 10), \
        "con la guarda ya no puede heredarlo"


def test_la_mezcla_manda_sobre_el_vecindario_y_solo_encoge():
    cov, mu = _cov(8), _means(8)
    ca = cov * np.outer(mu, mu)
    rel0 = np.sqrt(np.diag(cov))
    # una sigma de mezcla enorme: el objetivo tiene que quedarse en rel0, no crecer
    huge = np.abs(mu) * 1e6      # objetivo gigante => min(.) se queda en rel0
    out, diag = regularize_near_zero_relative_covariance(
        cov, mu, ca, L, snr_threshold=1.0, n_neighbors=3, mixture_abs_std=huge)
    assert diag['n_from_mixture'] > 0
    assert np.allclose(np.sqrt(np.diag(out)), rel0, atol=1e-12), \
        "min(rel_std, mezcla) => con una mezcla enorme no se toca nada"
    # y una pequeña sí encoge
    tiny = np.abs(mu) * 0.01
    out2, _ = regularize_near_zero_relative_covariance(
        cov, mu, ca, L, snr_threshold=1.0, n_neighbors=3, mixture_abs_std=tiny)
    flagged = np.abs(mu) / np.sqrt(np.diag(ca)) < 1.0
    assert np.all(np.sqrt(np.diag(out2))[flagged] <= rel0[flagged] + 1e-12)


def test_sigue_siendo_una_congruencia():
    """No puede mover ni una correlación ni romper la PSD."""
    cov, mu = _cov(8), _means(8)
    ca = cov * np.outer(mu, mu)
    fz = np.zeros(cov.shape[0], dtype=bool); fz[2::L] = True
    out, _ = regularize_near_zero_relative_covariance(
        cov, mu, ca, L, snr_threshold=1.0, n_neighbors=3,
        frozen_mask=fz, mixture_abs_std=np.abs(mu) * 0.5)

    def corr(m):
        d = np.sqrt(np.diag(m))
        return m / np.outer(d, d)

    assert np.allclose(corr(out), corr(cov), atol=1e-12)
    assert np.linalg.eigvalsh(out).min() >= -1e-12


def test_frozen_mask_de_forma_equivocada_falla_ruidosamente():
    cov, mu = _cov(4), _means(4)
    ca = cov * np.outer(mu, mu)
    try:
        regularize_near_zero_relative_covariance(
            cov, mu, ca, L, frozen_mask=np.zeros(3, dtype=bool))
    except ValueError as e:
        assert 'frozen_mask' in str(e)
    else:
        raise AssertionError("una mascara de la longitud equivocada tiene que abortar")


def test_el_bloque_analitico_es_el_between_del_promediado():
    """`analytic_between_block` == Sigma_d w_d (a_d - abar)(a_d - abar)^T."""
    from scripts.exfor_to_endf_research import analytic_between_block

    class _NR:
        # coeffs en c-space con c0=1 => endf_normalize deja a_l = c_l
        degree_weights = {2: 0.25, 4: 0.75}
        all_degrees_info = {
            2: {'coeffs': np.array([1.0, 0.4, 0.2, 0.0, 0.0])},
            4: {'coeffs': np.array([1.0, 0.4, 0.2, 0.1, 0.05])},
        }

    from scripts.resample_AD import endf_normalize_legendre_coeffs as _n

    an = analytic_between_block(_NR(), 4)
    a2 = _n(_NR.all_degrees_info[2]['coeffs'], include_a0=False)
    a4 = _n(_NR.all_degrees_info[4]['coeffs'], include_a0=False)
    w = np.array([0.25, 0.75])
    bar = w[0] * a2 + w[1] * a4
    exp = w[0] * np.outer(a2 - bar, a2 - bar) + w[1] * np.outer(a4 - bar, a4 - bar)
    assert np.allclose(an, exp, atol=1e-12)
    # y el orden ausente en el candidato bajo NO sale con varianza cero
    assert an[2, 2] > 0 and an[3, 3] > 0


def test_un_solo_grado_no_tiene_between():
    from scripts.exfor_to_endf_research import analytic_between_block

    class _NR:
        degree_weights = {4: 1.0}
        all_degrees_info = {4: {'coeffs': np.array([1.0, 0.4, 0.2, 0.1, 0.05])}}

    assert analytic_between_block(_NR(), 4) is None


# ── La puerta que de verdad importa: inercia ─────────────────────────────────

class _NRmix:
    def __init__(self, e_idx, cap):
        self.energy_index = e_idx
        self.mc_order_cap = cap
        self.degree_weights = {2: 0.3, 4: 0.7}
        self.all_degrees_info = {
            2: {'coeffs': np.array([1.0, 0.4, 0.2, 0.0, 0.0])},
            4: {'coeffs': np.array([1.0, 0.45, 0.18, 0.09, 0.04])},
        }


def _mixture_by_bin(e_idxs, max_degree=4):
    """El MC como es hoy: por encima del cap todos los grados llevan el MISMO
    valor congelado, así que el término entre-modelos se anula ahí."""
    out = {}
    for e in e_idxs:
        by_deg = {}
        for d in (2, 4):
            mean = np.array([0.44, 0.185, 0.063, 0.028])
            by_deg[d] = {'n': 5000, 'mean': mean,
                         'cov': np.diag([1e-4, 4e-5, 1e-9, 1e-9])}
        out[e] = by_deg
    return out


def test_flag_apagado_devuelve_exactamente_lo_de_antes():
    from scripts.exfor_to_endf_research import build_mixture_blocks
    nrs = [_NRmix(e, cap=2) for e in range(5)]
    mbb = _mixture_by_bin(range(5))
    off, doff = build_mixture_blocks(mbb, nrs, 4, restore_frozen_between=False)
    ref, dref = build_mixture_blocks(mbb, nrs, 4)
    for e in off:
        assert np.array_equal(off[e]['cov'], ref[e]['cov'])
        assert np.array_equal(off[e]['mean'], ref[e]['mean'])
    assert all('between_var_sampled' not in d for d in doff.values()), \
        "apagado no puede añadir columnas al CSV de diagnóstico"


def test_flag_encendido_devuelve_la_varianza_a_los_ordenes_congelados():
    from scripts.exfor_to_endf_research import build_mixture_blocks
    nrs = [_NRmix(e, cap=2) for e in range(5)]     # a_3 y a_4 congelados
    mbb = _mixture_by_bin(range(5))
    off, _ = build_mixture_blocks(mbb, nrs, 4, restore_frozen_between=False)
    on, don = build_mixture_blocks(mbb, nrs, 4, restore_frozen_between=True)
    for e in on:
        v_off = np.diag(off[e]['cov'])
        v_on = np.diag(on[e]['cov'])
        assert np.all(v_on[2:] > v_off[2:] * 100), \
            "a_3/a_4 congelados: el between del MC era ~0 y ahora no"
        assert np.array_equal(on[e]['mean'], off[e]['mean']), \
            "el CENTRAL no se toca: MF4 se queda como está"
        assert np.linalg.eigvalsh(on[e]['cov']).min() >= -1e-12
        assert 'between_var_sampled' in don[e]


def test_sin_ordenes_congelados_el_flag_no_hace_nada():
    from scripts.exfor_to_endf_research import build_mixture_blocks
    nrs = [_NRmix(e, cap=4) for e in range(5)]     # cap == max_degree
    mbb = _mixture_by_bin(range(5))
    off, _ = build_mixture_blocks(mbb, nrs, 4, restore_frozen_between=False)
    on, _ = build_mixture_blocks(mbb, nrs, 4, restore_frozen_between=True)
    for e in on:
        assert np.array_equal(on[e]['cov'], off[e]['cov'])


# ---------------------------------------------------------------------------
# La guarda, portada a la rama MULTIGRUPO (2026-08-22)
# ---------------------------------------------------------------------------
class _NRgrp:
    """Lo minimo que `mixture_regularisation_inputs` mira de un nominal_result."""
    def __init__(self, energy_index, cap, degrees, weights, max_degree=6):
        self.energy_index = energy_index
        self.mc_order_cap = cap
        self.all_degrees_info = degrees
        self.degree_weights = weights


def _nr_pair(max_degree=6):
    """Dos bins: el 0 con a_5/a_6 congelados, el 1 sin congelar."""
    import numpy as np
    d = {2: {'coeffs': np.array([1.0, 0.5, 0.2])},
         6: {'coeffs': np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.30, 0.20])}}
    w = {2: 0.5, 6: 0.5}
    return [_NRgrp(0, 4, d, w, max_degree), _NRgrp(1, 6, d, w, max_degree)]


def test_el_layout_agrupado_transporta_congelado_por_ANY_y_sigma_por_MAX():
    """Las dos elecciones de transporte, cada una hacia el lado seguro.

    `frozen` por ANY: sólo puede EXCLUIR donantes, que es la dirección del
    arreglo. `abs_std` por MAX: la función calcula ``min(rel_std, s/|mu|)`` y
    sólo puede encoger, así que un `s` mayor la deja más cerca de la identidad —
    y es el mismo argumento que COLLAPSE=max.
    """
    import numpy as np
    from scripts.exfor_to_endf_research import (
        mixture_regularisation_inputs, mixture_regularisation_inputs_grouped)
    L = 6
    nrs = _nr_pair(L)
    valid = [0, 1]
    fine_frozen, fine_abs = mixture_regularisation_inputs(nrs, valid, L)
    # un solo grupo con los dos bins
    g_frozen, g_abs = mixture_regularisation_inputs_grouped(nrs, valid, [[0, 1]], L)
    assert g_frozen.shape == (L,) and g_abs.shape == (L,)
    for l in range(L):
        idx = np.array([0 * L + l, 1 * L + l])
        assert g_frozen[l] == bool(fine_frozen[idx].any()), f"a_{l+1}: ANY roto"
        v = fine_abs[idx]
        v = v[np.isfinite(v)]
        if v.size:
            assert g_abs[l] == v.max(), f"a_{l+1}: MAX roto"
    # y el bin 0 tiene a_5/a_6 congelados, asi que el grupo tambien
    assert g_frozen[4] and g_frozen[5], "el ANY no propago el congelado"
    assert not g_frozen[0], "a_1 no esta congelado en ningun miembro"


def test_la_guarda_agrupada_quita_el_techo_del_100_por_ciento():
    """⚑ El techo del 100 % del multigrupo NO es un tope elegido.

    `snr_threshold = 1.0` marca todo parámetro con rel_std > 100 % y lo
    sustituye por el de un vecino no marcado, que por construcción está por
    debajo. Con `mixture_abs_std` el objetivo pasa a ser la sigma de la propia
    mezcla y el techo desaparece — sin tocar ningún umbral.
    """
    import numpy as np
    from scripts.exfor_utils import regularize_near_zero_relative_covariance
    L, n = 2, 4
    # tres parametros sanos y uno con rel_std enorme sobre una media diminuta
    mean = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-3])
    rel = np.diag(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 3.0])) ** 2
    cov_abs = rel * np.outer(mean, mean)
    sin, _ = regularize_near_zero_relative_covariance(
        cov_rel=rel, mean_params=mean, cov_abs=cov_abs, max_order=L)
    # la sigma que la mezcla implica para ese parametro: absoluta 3e-3
    mix = np.full(n * L, np.nan)
    mix[7] = 3.0 * 1e-3
    con, _ = regularize_near_zero_relative_covariance(
        cov_rel=rel, mean_params=mean, cov_abs=cov_abs, max_order=L,
        frozen_mask=np.zeros(n * L, bool), mixture_abs_std=mix)
    s_sin = np.sqrt(sin[7, 7])
    s_con = np.sqrt(con[7, 7])
    assert s_sin <= 1.0 + 1e-12, (
        f"sin guarda el parametro tenia que quedar bajo el techo del 100 %, "
        f"dio {s_sin*100:.1f} %")
    assert s_con > 2.9, (
        f"con guarda tenia que conservar la sigma de la mezcla (300 %), "
        f"dio {s_con*100:.1f} %")
