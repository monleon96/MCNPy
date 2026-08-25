"""El suelo por dispersion entre experimentos, version agrupada.

Las puertas que importan son tres, y cada una corresponde a un fallo concreto
que el diseño tuvo que corregir:

  * la congruencia no puede tocar ni una correlacion (lo unico que se declara de
    mas es sigma),
  * la mediana NO es una sigma -- convertir con 0.6745 o el suelo sale 1.48x
    corto, y eso solo lo caza una puerta fuera de muestra,
  * congelar c0 al agregado cuenta la NORMALIZACION de un experimento como
    desacuerdo de FORMA, que es el bug del estimador viejo.
"""
import numpy as np

from scripts.exfor_utils import (
    HALFNORMAL_MEDIAN,
    apply_between_experiment_floor_pooled,
    pool_between_experiment_sigma,
)
from scripts.resample_AD import compute_between_experiment_coeffs

L = 3


class _NR:
    """Lo minimo de NominalFitResult que las dos funciones leen."""
    def __init__(self, e_idx, n_qual, lone=None, per_exp=None, nom=None,
                 energy_mev=None):
        self.energy_index = e_idx
        self.energy_mev = energy_mev
        self.has_data = True
        self.interpolated = False
        self.between_exp_n_qual = n_qual
        self.between_exp_lone_entry = lone
        self.between_exp_per_experiment = per_exp
        self.nominal_coeffs = nom if nom is not None else np.array([1.0, .4, .2, .1])


def _cov(n_bins, l_max=L, seed=7, rel=0.1):
    rng = np.random.default_rng(seed)
    n = n_bins*l_max
    a = rng.normal(size=(n, n))
    c = a @ a.T/n
    d = np.sqrt(np.diag(c))
    return (c/np.outer(d, d))*rel*rel


def test_apply_false_es_la_identidad():
    """Con el flag apagado la covarianza sale intacta, bit a bit."""
    cov = _cov(6)
    nrs = [_NR(i, 1, lone="E1") for i in range(6)]
    out, diag = apply_between_experiment_floor_pooled(
        cov, nrs, list(range(6)), L,
        sigma_pool={("E1", l): 5.0 for l in range(L)},
        sigma_global=np.full(L, 5.0), apply=False)
    assert out is cov or np.array_equal(out, cov)
    assert diag['n_floored'] > 0, "el diagnostico tiene que medir aunque no aplique"


def test_congruencia_conserva_correlaciones_y_psd():
    cov = _cov(6)
    nrs = [_NR(i, 1, lone="E1") for i in range(6)]
    out, _ = apply_between_experiment_floor_pooled(
        cov, nrs, list(range(6)), L,
        sigma_pool={("E1", l): 0.5 for l in range(L)},
        sigma_global=np.full(L, 0.5), apply=True)
    def corr(m):
        d = np.sqrt(np.diag(m))
        return m/np.outer(d, d)
    assert np.allclose(corr(out), corr(cov), atol=1e-12)
    assert np.linalg.eigvalsh(out).min() >= -1e-12
    assert np.all(np.diag(out) >= np.diag(cov) - 1e-15), "el suelo nunca desinfla"


def test_se_apaga_como_uno_sobre_raiz_k():
    """Mismo bin, mismo experimento: mas experimentos => menos inflacion."""
    cov = _cov(1)
    sp = {("E1", l): 0.5 for l in range(L)}
    esc = []
    for k in (1, 4, 16):
        out, _ = apply_between_experiment_floor_pooled(
            cov, [_NR(0, k, lone="E1")], [0], L, sigma_pool=sp,
            sigma_global=np.full(L, 0.5), apply=True, only_blind=False)
        esc.append(np.sqrt(out[0, 0]/cov[0, 0]))
    assert esc[0] > esc[1] > esc[2] or np.isclose(esc[2], 1.0)
    assert np.isclose(esc[0]/esc[1], 2.0, rtol=1e-9) or np.isclose(esc[1], 1.0)


def test_only_blind_no_toca_los_bins_con_compañero():
    cov = _cov(4)
    nrs = [_NR(0, 1, lone="E1"), _NR(1, 3), _NR(2, 2), _NR(3, 1, lone="E1")]
    out, diag = apply_between_experiment_floor_pooled(
        cov, nrs, list(range(4)), L,
        sigma_pool={("E1", l): 0.5 for l in range(L)},
        sigma_global=np.full(L, 0.5), apply=True, only_blind=True)
    for b in (1, 2):
        sl = slice(b*L, (b+1)*L)
        assert np.allclose(out[sl, sl], cov[sl, sl])
    assert diag['n_blind_bins'] == 2


def test_la_mediana_se_convierte_a_sigma():
    """Sobre datos normales, el pool tiene que devolver la sigma, no la mediana."""
    from scripts.resample_AD import endf_normalize_legendre_coeffs
    rng = np.random.default_rng(11)
    # n grande a proposito: el estimador es una MEDIANA, cuyo error tipico es
    # ~1/(2 f(m) sqrt(n)) / 0.6745 = 5.8 % de sigma con n = 400. Con esa dispersion
    # una excursion de 2.7 sigma tumba el test sin que nada este mal. Con n = 4000
    # baja al 1.8 % y la tolerancia del 10 % mide la CONSTANTE, no el ruido.
    sigma_true, n_bins = 0.30, 4000
    nom = np.array([1.0, 0.4, 0.2, 0.1])
    # ⚠ El pool divide por `endf_normalize_legendre_coeffs(nominal_coeffs)`, no
    #   por c_l/c_0: la convencion ENDF lleva su propio factor por orden. Las
    #   desviaciones se generan en ESA escala para que el test mida la conversion
    #   mediana->sigma y no la convencion de normalizacion.
    a_ref = endf_normalize_legendre_coeffs(nom, include_a0=False)[:L]
    nrs = []
    for i in range(n_bins):
        # dos experimentos: d = |a1-a2| tiene sigma sqrt(2)*sigma_true, y el
        # encogimiento de la funcion lo divide por sqrt(2) -> sigma_true
        e1 = rng.normal(0, sigma_true*np.abs(a_ref), L)
        e2 = rng.normal(0, sigma_true*np.abs(a_ref), L)
        nrs.append(_NR(i, 2, per_exp={"A": (e1, 10, 1.0), "B": (e2, 10, 1.0)},
                       nom=nom))
    pool, glob, shape = pool_between_experiment_sigma(nrs, L, min_bins=10)
    assert shape is None, "sin energy_dependent no se construye curva ninguna"
    for l in range(L):
        assert np.isclose(pool[("A", l)], sigma_true, rtol=0.10), (l, pool[("A", l)])
        assert np.isclose(glob[l], sigma_true, rtol=0.10)
        # y el test TIENE que ser sensible a la constante: sin convertir la
        # mediana a sigma el resultado seria 0.6745*sigma y caeria fuera.
        assert not np.isclose(pool[("A", l)]*HALFNORMAL_MEDIAN, sigma_true,
                              rtol=0.10)


def test_median_no_convertida_saldria_corta():
    """La constante importa: sin ella el suelo sale 1.48x corto."""
    assert np.isclose(1.0/HALFNORMAL_MEDIAN, 1.4826, rtol=1e-3)


def _df(mu, y, unc, entry):
    import pandas as pd
    return pd.DataFrame({"mu": mu, "value": y, "unc": unc, "entry": entry})


def test_c0_libre_cancela_una_normalizacion_pura():
    """Dos experimentos con la MISMA forma y distinta escala no discrepan.

    Es el bug del estimador viejo: con c0 congelado al agregado, un factor
    global se cuenta como desacuerdo de forma en TODOS los ordenes.
    """
    import pandas as pd
    mu = np.linspace(-0.95, 0.95, 12)
    shape = 1.0 + 0.4*mu + 0.2*(1.5*mu**2 - 0.5)
    d = pd.concat([_df(mu, shape, 0.01*shape, "A"),
                   _df(mu, 1.30*shape, 0.013*shape, "B")], ignore_index=True)
    libre = compute_between_experiment_coeffs(d, degree=2, fixed_c0=1.15,
                                              freeze_c0=False)
    congelado = compute_between_experiment_coeffs(d, degree=2, fixed_c0=1.15,
                                                  freeze_c0=True)
    assert libre is not None and congelado is not None
    assert np.max(libre['scatter']) < 1e-6, (
        f"c0 libre deberia cancelar la normalizacion, dio {libre['scatter']}")
    assert np.max(congelado['scatter']) > 1e-3, (
        "con c0 congelado la normalizacion TIENE que aparecer como forma; "
        "si no, este test ya no prueba lo que dice")


def test_min_experiments_1_devuelve_el_censo():
    import pandas as pd
    mu = np.linspace(-0.95, 0.95, 12)
    y = 1.0 + 0.4*mu
    d = _df(mu, y, 0.01*y, "A")
    assert compute_between_experiment_coeffs(d, 2, 1.0) is None
    censo = compute_between_experiment_coeffs(d, 2, 1.0, min_experiments=1)
    assert censo is not None
    assert censo['n_experiments'] == 1
    assert censo['scatter'] is None
    assert list(censo['per_experiment']) == ["A"]


# ── El tope por orden: el suelo se para en a_4 ────────────────────────────────
# sigma_e se calibra como |Delta a_l| / |a_l_nom|, asi que el orden mas inflado
# es el que pasa mas cerca de cero, no el peor conocido. a_5 y a_6 aportan ~10 %
# del efecto en la banda DCS y se llevan la varianza de orden alto de 2 % a 27 %.

L6 = 6


def _nrs6(n, lone="E1"):
    return [_NR(i, 1, lone=lone, nom=np.array([1., .4, .2, .1, .05, .02, .01]))
            for i in range(n)]


def test_max_floor_order_deja_a5_y_a6_intactos():
    cov = _cov(5, l_max=L6)
    out, diag = apply_between_experiment_floor_pooled(
        cov, _nrs6(5), list(range(5)), L6,
        sigma_pool={("E1", l): 5.0 for l in range(L6)},
        sigma_global=np.full(L6, 5.0), apply=True, max_floor_order=4)
    d_in, d_out = np.diag(cov), np.diag(out)
    for b in range(5):
        for l in range(4):
            assert d_out[b*L6 + l] > d_in[b*L6 + l], f"l={l+1} tenia que inflarse"
        for l in (4, 5):
            assert d_out[b*L6 + l] == d_in[b*L6 + l], f"l={l+1} NO se toca"
    assert diag['n_withheld'] == 5*2
    assert diag['withheld_per_order'][4][0] == 5
    assert diag['withheld_per_order'][5][0] == 5
    assert diag['per_order'][4][0] == 0 and diag['per_order'][5][0] == 0


def test_max_floor_order_6_restaura_el_comportamiento_viejo():
    """La puerta de reversibilidad: el tope es una eleccion, no un cambio de
    definicion. Con max_floor_order = max_order nada queda retenido."""
    cov = _cov(5, l_max=L6)
    kw = dict(sigma_pool={("E1", l): 5.0 for l in range(L6)},
              sigma_global=np.full(L6, 5.0), apply=True)
    out6, d6 = apply_between_experiment_floor_pooled(
        cov, _nrs6(5), list(range(5)), L6, max_floor_order=6, **kw)
    assert d6['n_withheld'] == 0
    assert d6['n_floored'] == 5*L6
    assert np.all(np.diag(out6) > np.diag(cov))


def test_el_tope_no_rompe_la_congruencia():
    """Sigue siendo S C S: ni una correlacion se mueve, y sigue siendo PSD."""
    cov = _cov(5, l_max=L6)
    out, _ = apply_between_experiment_floor_pooled(
        cov, _nrs6(5), list(range(5)), L6,
        sigma_pool={("E1", l): 0.5 for l in range(L6)},
        sigma_global=np.full(L6, 0.5), apply=True, max_floor_order=4)
    def corr(m):
        d = np.sqrt(np.diag(m))
        return m/np.outer(d, d)
    assert np.allclose(corr(out), corr(cov), atol=1e-12)
    assert np.linalg.eigvalsh(out).min() >= -1e-12


def test_el_diagnostico_de_a5_a6_no_se_pierde():
    """Retener no es callar: la inflacion que NO se aplica sigue medida, para
    que la decision se pueda auditar y revertir con un numero delante."""
    cov = _cov(5, l_max=L6)
    _, diag = apply_between_experiment_floor_pooled(
        cov, _nrs6(5), list(range(5)), L6,
        sigma_pool={("E1", l): 5.0 for l in range(L6)},
        sigma_global=np.full(L6, 5.0), apply=True, max_floor_order=4)
    for l in (4, 5):
        cnt, med = diag['withheld_per_order'][l]
        assert cnt == 5 and med > 1.0
    assert diag['max_floor_order'] == 4


# ── La dependencia en ENERGIA (diseño §8) ────────────────────────────────────
# El estimador de hoy es un escalar sobre todo el rango y fuera de muestra su
# cobertura CAE con la energia (90,6 / 79,3 / 63,1 % por tercios). La forma se
# mide en DCS -- donde no hay denominador -- y entra como factor sobre el
# objetivo. Estas puertas fijan las cuatro cosas que pueden romperse.

from scripts.exfor_utils import (          # noqa: E402
    DCS_MU_GRID,
    _endf_shape_series,
    dcs_shape_disagreement,
    energy_shape_factor,
)


def test_la_serie_reconstruye_la_dcs_normalizada():
    """`_endf_shape_series` tiene que dar y(mu)/c_0, no otra normalizacion.

    Puerta dura: la integral sobre mu es 2 para CUALQUIER a_l, porque solo
    sobrevive a_0 = 1. Si alguien mete el factor (2l+1) en el sitio de mas o de
    menos, esta integral deja de valer 2.
    """
    from numpy.polynomial.legendre import legval
    rng = np.random.default_rng(3)
    for _ in range(5):
        a = rng.normal(scale=0.2, size=6)
        mu = np.linspace(-1, 1, 20001)
        assert np.isclose(np.trapezoid(legval(mu, _endf_shape_series(a)), mu),
                          2.0, rtol=1e-6)
    # y frente a la definicion, coeficiente a coeficiente: a_l = (c_l/c0)/(2l+1)
    from scripts.resample_AD import endf_normalize_legendre_coeffs
    c = np.array([2.5, 1.2, -0.4, 0.3])
    a = endf_normalize_legendre_coeffs(c)
    assert np.allclose(_endf_shape_series(a), c/c[0])


def test_una_normalizacion_pura_no_es_desacuerdo_de_forma():
    """Duplicar c_0 no cambia los a_l, luego el desacuerdo en DCS es 0.

    Es la razon por la que esto puede vivir en MF34: mide FORMA. Si algun dia
    alguien vuelve a `freeze_c0=True` aguas arriba, este test sigue pasando
    pero el de `test_c0_libre_cancela_una_normalizacion_pura` cae -- los dos
    juntos cubren la cadena.
    """
    a = np.array([0.4, 0.2, 0.1])
    assert dcs_shape_disagreement(a, a) == 0.0
    assert dcs_shape_disagreement(a, a*1.0) == 0.0
    # y una diferencia de forma SI se ve
    b = a.copy(); b[1] += 0.05
    assert dcs_shape_disagreement(a, b) > 1e-3


def test_el_factor_es_uno_cuando_no_hay_curva():
    """La puerta de inercia del lector: sin curva, sin energia o sin entrada
    calibrada el factor es exactamente 1.0."""
    assert energy_shape_factor(None, "A", 1.0) == 1.0
    assert energy_shape_factor({}, "A", 1.0) == 1.0
    cur = {"A": (np.array([1.0, 2.0]), np.array([1.0, 3.0]))}
    assert energy_shape_factor(cur, "A", None) == 1.0
    assert energy_shape_factor(cur, "A", float("nan")) == 1.0
    assert energy_shape_factor(cur, "B", 1.5) == 1.0        # sin respaldo
    # con respaldo agrupado, una entrada desconocida lo usa
    cur[None] = (np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    assert energy_shape_factor(cur, "B", 1.5) == 2.0
    # y la extrapolacion es PLANA, no lineal: fuera del rango calibrado la
    # mediana movil no tiene soporte y una recta sobre una cola ruidosa es lo
    # unico que podria hacer el factor no acotado
    assert energy_shape_factor(cur, "A", 0.0) == 1.0
    assert energy_shape_factor(cur, "A", 99.0) == 3.0


def _nrs_energia(n_bins, slope, seed=5, sigma=0.06):
    """Bins con desacuerdo cuya AMPLITUD crece linealmente con la energia.

    ⚠ Los a_l se generan ALREDEDOR del nominal, no como ruido puro alrededor de
    cero, y eso no es cosmetica. `dcs_shape_disagreement` es un cociente
    |Delta| / media(|.|) y SATURA: con a_l ~ N(0, 1) las dos formas dejan de
    parecerse entre si y la mediana se pega al techo (medido: max exactamente
    sqrt(2)), asi que la metrica se vuelve ciega a la amplitud. En el regimen
    real -- a_l ~ 0.1-0.5 y desacuerdos del 10-30 % -- es lineal, que es donde
    se midio y donde tiene que probarse.
    """
    from scripts.resample_AD import endf_normalize_legendre_coeffs
    rng = np.random.default_rng(seed)
    nom = np.array([1.0, 0.4, 0.2, 0.1])
    a_ref = endf_normalize_legendre_coeffs(nom, include_a0=False)[:L]
    E = np.linspace(1.0, 3.0, n_bins)
    out = []
    for i, e in enumerate(E):
        amp = sigma*(1.0 + slope*(e - E[0])/(E[-1] - E[0]))
        e1 = a_ref + rng.normal(0, amp*np.abs(a_ref), L)
        e2 = a_ref + rng.normal(0, amp*np.abs(a_ref), L)
        out.append(_NR(i, 2, per_exp={"A": (e1, 10, 1.0), "B": (e2, 10, 1.0)},
                       nom=nom, energy_mev=float(e)))
    return out


def test_la_curva_sigue_la_pendiente_que_se_le_mete():
    """Puerta de sensibilidad: con desacuerdo que crece 4x, el factor crece.

    ⚠ Se mide SIN recorte a proposito. El recorte pega a 1 toda la mitad por
    debajo de la mediana, asi que comprime el cociente extremo-a-extremo y
    convertiria esta puerta en una medida del recorte y no de la pendiente.
    Que el recorte no desinfla lo fija su propio test.
    """
    nrs = _nrs_energia(600, slope=3.0)
    _, _, shape = pool_between_experiment_sigma(
        nrs, L, min_bins=10, energy_dependent=True, energy_window=40,
        energy_clamp=False)
    assert shape is not None and "A" in shape
    cen, fac = shape["A"]
    assert cen[0] < cen[-1], "los centros tienen que salir ordenados en energia"
    assert fac[0] < fac[-1], "el factor tiene que crecer con la energia"
    assert fac[-1]/fac[0] > 2.5, f"pendiente demasiado plana: {fac[-1]/fac[0]:.2f}"
    # y la puerta tiene que ser SENSIBLE: sin pendiente el cociente es ~1
    _, _, plano = pool_between_experiment_sigma(
        _nrs_energia(600, slope=0.0), L, min_bins=10,
        energy_dependent=True, energy_window=40, energy_clamp=False)
    f0 = plano["A"][1]
    assert f0[-1]/f0[0] < 1.5, f"la puerta no discrimina: {f0[-1]/f0[0]:.2f}"


def test_sin_pendiente_la_curva_es_plana_y_el_recorte_la_deja_en_uno():
    """Puerta de inercia estadistica: si el desacuerdo NO depende de E, el
    estimador con energia tiene que reducirse al constante."""
    nrs = _nrs_energia(600, slope=0.0)
    _, _, shape = pool_between_experiment_sigma(
        nrs, L, min_bins=10, energy_dependent=True, energy_window=40)
    cen, fac = shape["A"]
    assert np.median(fac) < 1.15, f"deberia ser ~1, dio {np.median(fac):.3f}"
    # sin recorte la curva oscila alrededor de 1 por los dos lados
    _, _, libre = pool_between_experiment_sigma(
        nrs, L, min_bins=10, energy_dependent=True, energy_window=40,
        energy_clamp=False)
    assert libre["A"][1].min() < 1.0 < libre["A"][1].max()
    assert np.isclose(np.median(libre["A"][1]), 1.0, atol=0.10)


def test_el_recorte_nunca_declara_menos_que_la_constante():
    """La barra de Juan: sub-declarar es inaceptable, sobre-declarar solo
    desperdicia. Recortado, el factor domina al libre en TODO punto."""
    nrs = _nrs_energia(600, slope=3.0)
    kw = dict(min_bins=10, energy_dependent=True, energy_window=40)
    _, _, rec = pool_between_experiment_sigma(nrs, L, energy_clamp=True, **kw)
    _, _, lib = pool_between_experiment_sigma(nrs, L, energy_clamp=False, **kw)
    assert np.all(rec["A"][1] >= lib["A"][1] - 1e-12)
    assert np.all(rec["A"][1] >= 1.0)


def test_energy_shape_none_es_bit_a_bit_lo_de_hoy():
    """⚑ LA PUERTA DE INERCIA, y es igualdad de arrays, no tolerancia."""
    cov = _cov(6)
    nrs = [_NR(i, 1, lone="E1", energy_mev=1.0 + 0.1*i) for i in range(6)]
    kw = dict(sigma_pool={("E1", l): 0.5 for l in range(L)},
              sigma_global=np.full(L, 0.5), apply=True)
    viejo, dv = apply_between_experiment_floor_pooled(
        cov, nrs, list(range(6)), L, **kw)
    nuevo, dn = apply_between_experiment_floor_pooled(
        cov, nrs, list(range(6)), L, energy_shape=None, **kw)
    assert np.array_equal(viejo, nuevo)
    assert dn['energy_shape'] is False and dn['energy_factor'] == (0, 1.0, 1.0)
    assert dv['n_floored'] == dn['n_floored']


def test_el_factor_multiplica_el_objetivo_y_sigue_siendo_congruencia():
    """Duplicar el factor duplica la sigma declarada donde el suelo actua, y no
    mueve ni una correlacion: sigue siendo S C S."""
    cov = _cov(6)
    nrs = [_NR(i, 1, lone="E1", energy_mev=1.0 + 0.1*i) for i in range(6)]
    kw = dict(sigma_pool={("E1", l): 0.5 for l in range(L)},
              sigma_global=np.full(L, 0.5), apply=True)
    base, _ = apply_between_experiment_floor_pooled(cov, nrs, list(range(6)), L, **kw)
    shape = {"E1": (np.array([0.0, 10.0]), np.array([2.0, 2.0]))}
    doble, dd = apply_between_experiment_floor_pooled(
        cov, nrs, list(range(6)), L, energy_shape=shape, **kw)
    assert np.allclose(np.sqrt(np.diag(doble)), 2.0*np.sqrt(np.diag(base)))
    def corr(m):
        d = np.sqrt(np.diag(m))
        return m/np.outer(d, d)
    assert np.allclose(corr(doble), corr(cov), atol=1e-12)
    assert np.linalg.eigvalsh(doble).min() >= -1e-12
    assert dd['energy_shape'] is True
    assert dd['energy_factor'][0] == 6 and np.isclose(dd['energy_factor'][1], 2.0)


def test_el_factor_no_puede_desinflar():
    """Un factor < 1 no puede llevar la sigma por debajo de la declarada: el
    `max(1, ...)` del objetivo sigue mandando."""
    cov = _cov(6)
    nrs = [_NR(i, 1, lone="E1", energy_mev=1.0 + 0.1*i) for i in range(6)]
    shape = {"E1": (np.array([0.0, 10.0]), np.array([0.01, 0.01]))}
    out, _ = apply_between_experiment_floor_pooled(
        cov, nrs, list(range(6)), L,
        sigma_pool={("E1", l): 0.5 for l in range(L)},
        sigma_global=np.full(L, 0.5), apply=True, energy_shape=shape)
    assert np.all(np.diag(out) >= np.diag(cov) - 1e-15)
