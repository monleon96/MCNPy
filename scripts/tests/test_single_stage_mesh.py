"""La malla en UNA sola etapa: criterio fisico y diagonal conservadora.

Hoy el agrupamiento son DOS etapas -- `find_adaptive_group_boundaries` lleva el
fino a 660 con rho y sigma-ratio LOS DOS leidos en a_1, y el DP por orden
particiona ESOS 660. El DP no puede deshacer lo que la primera etapa fusiono, y
es la primera la que destruye el 42,5 %.

Aqui se fija la ruta de una etapa: la malla se elige por orden DESDE EL FINO con
el criterio fisico (resolucion sigma_E, consistencia de a_l, tope sigma-ratio por
orden), y la diagonal del grupo pasa a ser el MAXIMO, no la media.
"""
import numpy as np

from scripts.per_order_mesh import (
    collapse_relative_per_order,
    folded_variance_ratio,
    per_order_meshes,
    physical_solve_order,
    _group_of,
)

L = 2


def _case(n=60, seed=3):
    """Rejilla fina uniforme; a_1 suave, a_2 con estructura por debajo de sigma_E."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(1.0e6, 2.0e6, n + 1)
    w = np.diff(edges)
    sE = np.full(n, 5.0 * w[0])                       # cabe ~5 bins por resolucion
    a = np.empty(n * L)
    a[0::L] = 0.5 + 0.001 * np.arange(n)              # a_1 casi constante
    a[1::L] = 0.2 + 0.05 * np.sin(np.arange(n))       # a_2 oscila
    rel = np.zeros((n * L, n * L))
    for l in range(L):
        i = np.arange(n) * L + l
        d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        s = (0.05 + 0.02 * rng.random(n)) * (3.0 if l else 1.0)
        rel[np.ix_(i, i)] = (0.99 ** d) * np.outer(s, s)
    return edges, w, sE, a, rel


def test_la_resolucion_acota_la_anchura_del_grupo():
    """Ningun grupo puede ser mas ancho que la sigma_E que cubre."""
    edges, w, sE, a, rel = _case()
    n = len(w)
    i = np.arange(n) * L
    B = rel[np.ix_(i, i)] * np.outer(a[i], a[i])
    cuts = physical_solve_order(B, a[i], w, sE, res_factor=1.0)
    anchos = np.array([w[lo:hi].sum() for lo, hi in zip(cuts[:-1], cuts[1:])])
    peor = np.array([sE[lo:hi].min() for lo, hi in zip(cuts[:-1], cuts[1:])])
    assert np.all(anchos <= peor + 1e-9), "un grupo se paso de la resolucion"
    assert len(cuts) - 1 < n, "con esta sigma_E TIENE que agrupar algo"


def test_a_l_inconsistente_impide_agrupar():
    """⚑ El fichero es RELATIVO: lo homogeneo tiene que ser el DENOMINADOR.

    Mismo caso, misma resolucion; lo unico que cambia es que a_l oscila. Con la
    consistencia activada el numero de grupos tiene que SUBIR.
    """
    edges, w, sE, a, rel = _case()
    n = len(w)
    i = np.arange(n) * L + 1                                   # a_2, el que oscila
    B = rel[np.ix_(i, i)] * np.outer(a[i], a[i])
    sin_k = len(physical_solve_order(B, a[i], w, sE, res_factor=1.0)) - 1
    con_k = len(physical_solve_order(B, a[i], w, sE, res_factor=1.0,
                                     a_consistency_k=0.5)) - 1
    assert con_k > sin_k, (sin_k, con_k)


def test_el_tope_acota_LO_QUE_EL_MAXIMO_SOBRE_DECLARA():
    """⛔ LA PUERTA DEL §J-7: el tope tiene que leerse en la sigma RELATIVA.

    Con ``COLLAPSE = max`` la diagonal del grupo es ``max_j rel_jj``, asi que lo
    que el consumidor ve de mas en el miembro ``i`` es ``max_j rel_jj / rel_ii``
    -- un cociente de varianzas RELATIVAS. ``B`` llega ABSOLUTA, y leer el tope
    ahi acota otra cosa: los dos difieren por el cociente de ``|a_l|`` dentro del
    grupo, que la consistencia de a_l deja libre.

    Medido en la run 99 con c = 3, el maximo sobre-declaraba 536x en a_4 y 283x
    en a_1 con la lectura absoluta. Aqui se construye el caso minimo que lo
    reproduce: sigma ABSOLUTA constante (asi el tope leido en absoluta no rechaza
    NADA) y ``a_l`` que cae un orden de magnitud dentro de una sigma_E.
    """
    n = 12
    edges = np.linspace(1.0e6, 1.1e6, n + 1)
    w = np.diff(edges)
    sE = np.full(n, 1.1 * w.sum())                    # la resolucion permite fusionar todo
    a_g = np.geomspace(1.0, 0.05, n)                  # el denominador barre 20x
    sig_abs = np.full(n, 0.01)                        # ... y la sigma absoluta NO se mueve
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    B = (0.999 ** d) * np.outer(sig_abs, sig_abs)     # absoluta, muy correlacionada

    c = 3.0
    cuts = physical_solve_order(B, a_g, w, sE, res_factor=1.0, sigma_ratio_max=c)
    rel = sig_abs / a_g                               # sigma relativa por bin
    peor = max(
        (rel[lo:hi].max() / rel[lo:hi].min()) if hi - lo > 1 else 1.0
        for lo, hi in zip(cuts[:-1], cuts[1:])
    )
    assert peor <= c * (1 + 1e-9), (
        f"el tope c={c} deja pasar una sobre-declaracion de {peor:.1f}x: "
        f"se esta leyendo en la sigma equivocada")


def test_el_tope_es_inerte_cuando_las_dos_lecturas_coinciden():
    """Con ``|a_l|`` constante, relativa y absoluta son la misma sigma salvo un
    factor global, asi que la malla no puede cambiar. Es el control del test de
    arriba: fija que lo medido alli es el DENOMINADOR y no otra cosa."""
    n = 12
    edges = np.linspace(1.0e6, 1.1e6, n + 1)
    w = np.diff(edges)
    sE = np.full(n, 1.1 * w.sum())
    sig_abs = np.geomspace(0.002, 0.05, n)
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    B = (0.999 ** d) * np.outer(sig_abs, sig_abs)
    for a0 in (1.0, 0.3, 7.0):
        a_g = np.full(n, a0)
        cuts = physical_solve_order(B, a_g, w, sE, res_factor=1.0, sigma_ratio_max=3.0)
        assert np.array_equal(cuts, physical_solve_order(
            B, np.full(n, 1.0), w, sE, res_factor=1.0, sigma_ratio_max=3.0)), a0


def test_monotonia_en_los_tres_criterios():
    """Mas restriccion => nunca menos grupos. Es la puerta que ya cazo un indice
    invertido (a_2 de 948 a 20 grupos)."""
    edges, w, sE, a, rel = _case()
    n = len(w)
    i = np.arange(n) * L
    B = rel[np.ix_(i, i)] * np.outer(a[i], a[i])
    def g(**kw):
        return len(physical_solve_order(B, a[i], w, sE, **kw)) - 1
    base = g(res_factor=2.0)
    assert g(res_factor=1.0) >= base
    assert g(res_factor=2.0, a_consistency_k=1.0) >= base
    assert g(res_factor=2.0, sigma_ratio_max=1.5) >= base
    assert g(res_factor=2.0, sigma_ratio_max=1.5, a_consistency_k=1.0) >= base


def test_el_maximo_no_puede_declarar_menos_que_el_fino():
    """La aritmetica del fichero relativo: R_gg = max_j B_jj => todo miembro
    declara >= lo que el objeto fino le da."""
    edges, w, sE, a, rel = _case()
    n = len(w)
    meshes = {l: edges[::7] if edges[::7][-1] == edges[-1]
              else np.concatenate([edges[::7], [edges[-1]]])
              for l in range(1, L + 1)}
    for variant, esperado in (("mean", False), ("max", True)):
        blocks, grids, W = collapse_relative_per_order(
            edges, rel, a, L, meshes, variant=variant)
        for l in range(1, L + 1):
            rows = np.arange(n) * L + (l - 1)
            gi = _group_of(edges, meshes[l])
            dg = np.diag(blocks[(l, l)])[gi]
            df = np.diag(rel[np.ix_(rows, rows)])
            if esperado:
                assert np.all(dg >= df - 1e-12), f"a_{l}: el maximo se quedo corto"
            else:
                assert np.any(dg < df - 1e-12), (
                    f"a_{l}: la media TENIA que quedarse corta en algun bin; "
                    "si no, este test no prueba nada")


def _mesh_cada(edges, k):
    e = edges[::k]
    return e if e[-1] == edges[-1] else np.concatenate([e, [edges[-1]]])


def test_el_margen_pone_el_minimo_plegado_en_uno_EXACTO():
    """⚑ Y es EXACTO, no iterativo: el plegado es LINEAL en la covarianza, asi
    que escalar por 1/min(cociente) pone el minimo en 1 de una sola vez.

    Se prueba sobre la diagonal 'mean', que es la que sub-declara de verdad
    (min < 1). Sobre 'max' el minimo ya puede estar por encima de 1 y entonces
    el margen NO debe hacer nada -- eso es el test siguiente.
    """
    edges, w, sE, a, rel = _case()
    n = len(w)
    meshes = {l: _mesh_cada(edges, 7) for l in range(1, L + 1)}
    sin, _, _ = collapse_relative_per_order(edges, rel, a, L, meshes, variant="mean")
    con, _, _ = collapse_relative_per_order(
        edges, rel, a, L, meshes, variant="mean", sigma_E=sE, calibrate_margin=True)
    for l in range(1, L + 1):
        rows = np.arange(n) * L + (l - 1)
        gi = _group_of(edges, meshes[l])
        fine = rel[np.ix_(rows, rows)]
        r0 = folded_variance_ratio(fine, sin[(l, l)], gi, a[rows], edges, sE)
        r1 = folded_variance_ratio(fine, con[(l, l)], gi, a[rows], edges, sE)
        assert np.nanmin(r0) < 1.0, (
            f"a_{l}: la media TENIA que sub-declarar aqui; si no, el test no "
            "prueba que el margen lo repare")
        assert np.isclose(np.nanmin(r1), 1.0, atol=1e-9), f"a_{l}: {np.nanmin(r1)}"


def test_el_margen_nunca_desinfla():
    """Si el plegado ya esta por encima de 1, el margen no toca nada: es un
    suelo, no una calibracion."""
    edges, w, sE, a, rel = _case()
    n = len(w)
    meshes = {l: _mesh_cada(edges, 7) for l in range(1, L + 1)}
    sin, _, _ = collapse_relative_per_order(edges, rel, a, L, meshes, variant="max")
    con, _, _ = collapse_relative_per_order(
        edges, rel, a, L, meshes, variant="max", sigma_E=sE, calibrate_margin=True)
    for l in range(1, L + 1):
        rows = np.arange(n) * L + (l - 1)
        gi = _group_of(edges, meshes[l])
        fine = rel[np.ix_(rows, rows)]
        r0 = folded_variance_ratio(fine, sin[(l, l)], gi, a[rows], edges, sE)
        r1 = folded_variance_ratio(fine, con[(l, l)], gi, a[rows], edges, sE)
        assert np.nanmin(r1) >= 1.0 - 1e-9, f"a_{l}: sub-declara"
        assert np.nanmin(r1) >= np.nanmin(r0) - 1e-12, f"a_{l}: el margen desinflo"


def test_singletons_dan_cociente_uno_exacto():
    """La puerta obligatoria del plegado: sin agrupar, el cociente es 1 EXACTO.
    Si no, el plegado o la expansion estan mal."""
    edges, w, sE, a, rel = _case()
    n = len(w)
    meshes = {l: edges for l in range(1, L + 1)}
    blocks, _, _ = collapse_relative_per_order(edges, rel, a, L, meshes, variant="max")
    for l in range(1, L + 1):
        rows = np.arange(n) * L + (l - 1)
        fine = rel[np.ix_(rows, rows)]
        r = folded_variance_ratio(fine, blocks[(l, l)], _group_of(edges, edges),
                                  a[rows], edges, sE)
        assert np.nanmax(np.abs(r - 1.0)) < 1e-10, f"a_{l}: {np.nanmax(np.abs(r-1)):.2e}"


def test_inercia_el_camino_de_siempre_no_se_mueve():
    """Sin `physical` ni `variant`, `per_order_meshes` y el colapso dan lo de
    siempre, bit a bit."""
    edges, w, sE, a, rel = _case()
    m0 = per_order_meshes(edges, rel, a, L)
    m1 = per_order_meshes(edges, rel, a, L, physical=None, sigma_E=sE)
    for l in range(1, L + 1):
        assert np.array_equal(m0[l], m1[l])
    b0, g0, w0 = collapse_relative_per_order(edges, rel, a, L, m0)
    b1, g1, w1 = collapse_relative_per_order(edges, rel, a, L, m0, variant="mean")
    for k in b0:
        assert np.array_equal(b0[k], b1[k])


# ---------------------------------------------------------------------------
# El suelo de no-cancelacion en la diagonal 'max' (2026-08-22)
# ---------------------------------------------------------------------------
def _caso_cancelacion(n=8):
    """Un grupo cuyos dos bins estan EXACTAMENTE anticorrelados.

    Es la situacion que la run 102cp trae a manos llenas: pares adyacentes con
    rho = -1,0000 exacto son 46/236/485 en a_4/a_5/a_6 y CERO en a_1-a_3, que es
    donde la etapa 1 decide sus fusiones.  La media ponderada de un par asi
    tiene varianza exactamente 0.
    """
    edges = np.linspace(1.0e6, 2.0e6, n + 1)
    a = np.empty(n * L)
    a[0::L] = 1.0
    a[1::L] = 1.0
    rel = np.zeros((n * L, n * L))
    for l in range(L):
        i = np.arange(n) * L + l
        rel[np.ix_(i, i)] = np.eye(n) * 0.04
    # el par (0, 1) del orden a_2, con rho = -1 exacto
    i = np.arange(n) * L + 1
    rel[i[0], i[1]] = rel[i[1], i[0]] = -0.04
    meshes = {l: edges[::2] for l in range(1, L + 1)}
    return edges, a, rel, meshes


def test_la_cancelacion_exacta_no_puede_dar_una_congruencia_de_1e149():
    """⛔ El `1e-300` de `dg` no era una guarda, era el mecanismo.

    Con `diag(A B A^T) = 0` exacto, `s = sqrt(target/1e-300) ~ 4e149` multiplica
    la fila y la columna enteras del grupo -- y eso se escribe en una covarianza
    relativa con 7 cifras.  El suelo correcto es la propia restriccion de
    no-cancelacion del DP.
    """
    edges, a, rel, meshes = _caso_cancelacion()
    blocks, _, W = collapse_relative_per_order(
        edges, rel, a, L, meshes, variant="max")
    B22 = blocks[(2, 2)]
    assert np.all(np.isfinite(B22)), "la congruencia se fue a infinito"
    # ⚑ LO QUE ACOTA ES LA FILA ENTERA, no solo la diagonal. Con el `1e-300` el
    # grupo cancelado tenia diag EXACTAMENTE 0 (0 * 1e149 = 0) y eran sus
    # CRUZADOS los que salian a 1e149 -- por eso el cociente plegado daba 1e139.
    assert np.abs(B22).max() <= 2.0 * 0.04 * (1.0 + 1e-9), (
        f"la fila del grupo cancelado llega a {np.abs(B22).max():.3e}: la "
        "congruencia sigue sin acotar")
    # ⛔ Y LO QUE NO SE ARREGLA AQUI, A PROPOSITO: una congruencia NO PUEDE
    # levantar un cero (s * 0 * s = 0 para cualquier s), asi que el grupo
    # cancelado declara 0 pase lo que pase. Eso es sub-declarar, y la barra dice
    # que es inaceptable -- pero el sitio donde se arregla es la MALLA, vetando
    # el grupo (la restriccion de no-cancelacion del DP), no el colapso.
    assert np.diag(B22)[0] == 0.0, (
        "si esto deja de dar 0 es que el colapso ha empezado a inventar "
        "varianza; ese cambio se discute, no se cuela")


def test_el_suelo_de_no_cancelacion_es_INERTE_donde_el_DP_admite():
    """Donde la restriccion se cumple -- todo lo que el DP deja pasar -- el
    cambio tiene que ser BIT A BIT identico al de siempre.

    Sin esta puerta el suelo podria estar moviendo el objeto que ya embarca.
    """
    edges, w, sE, a, rel = _case()
    meshes = {l: _mesh_cada(edges, 7) for l in range(1, L + 1)}
    blocks, _, W = collapse_relative_per_order(
        edges, rel, a, L, meshes, variant="max")
    # se rehace a mano con el 1e-300 de antes
    n = len(w)
    for l in range(1, L + 1):
        rows = np.arange(n) * L + (l - 1)
        own = W[l] @ rel[np.ix_(rows, rows)] @ W[l].T
        gi = _group_of(edges, meshes[l])
        d_fine = np.maximum(np.diag(rel[np.ix_(rows, rows)]), 0.0)
        target = np.zeros(own.shape[0])
        np.maximum.at(target, gi, d_fine)
        indep = (W[l] ** 2) @ d_fine
        assert np.all(np.diag(own) >= indep - 1e-12), (
            f"a_{l}: el caso base ya viola la no-cancelacion; este test no "
            "probaria la inercia")
        s_viejo = np.sqrt(np.maximum(target / np.maximum(np.diag(own), 1e-300), 1.0))
        esperado = s_viejo[:, None] * own * s_viejo[None, :]
        assert np.array_equal(blocks[(l, l)], esperado), \
            f"a_{l}: el suelo movio un objeto que el DP admite"


def test_el_DP_FISICO_veta_el_grupo_que_cancela_de_punta_a_punta():
    """Opción 1 (Juan, 22-ago-2026): que la malla vete el grupo, no que el
    colapso lo parchee.

    ⚑ Y ES LA PUERTA QUE FALTABA. `physical_solve_order` lleva la restricción
    de no-cancelación (`g >= vv`), pero eso sólo protege lo que el DP VE. En la
    ruta de dos etapas el par ya viene fusionado por `find_adaptive_group_
    boundaries`, que decide leyendo a_1 -- y en a_1 no hay ni un par con
    rho = -1 mientras que en a_6 son el 28 %. Aquí se comprueba de punta a
    punta: malla desde el FINO -> colapso -> ni un grupo viola la restricción.
    """
    n = 40
    edges = np.linspace(1.0e6, 2.0e6, n + 1)
    w = np.diff(edges)
    sE = np.full(n, 20.0 * w[0])          # resolución MUY laxa: sin la
    a = np.empty(n * L)                   # restricción, el DP fusionaría todo
    a[0::L] = 1.0
    a[1::L] = 1.0
    rel = np.zeros((n * L, n * L))
    for l in range(L):
        i = np.arange(n) * L + l
        rel[np.ix_(i, i)] = np.eye(n) * 0.04
    # una escalera de pares EXACTAMENTE anticorrelados en a_2, como en 102cp
    i = np.arange(n) * L + 1
    for k in range(0, n - 1, 2):
        rel[i[k], i[k + 1]] = rel[i[k + 1], i[k]] = -0.04

    meshes = per_order_meshes(
        edges, rel, a, L, sigma_E=sE,
        physical=dict(res_factor=1.0, a_consistency_k=10.0, sigma_ratio_max=3.0))
    blocks, _, W = collapse_relative_per_order(
        edges, rel, a, L, meshes, variant="max")

    for l in range(1, L + 1):
        rows = np.arange(n) * L + (l - 1)
        own = W[l] @ rel[np.ix_(rows, rows)] @ W[l].T
        d_fine = np.maximum(np.diag(rel[np.ix_(rows, rows)]), 0.0)
        indep = (W[l] ** 2) @ d_fine
        viola = np.flatnonzero(np.diag(own) < indep - 1e-12)
        assert viola.size == 0, (
            f"a_{l}: {viola.size} grupos violan la no-cancelación; el DP físico "
            "tenía que haberlos vetado")
    # y el test no prueba nada si el DP se limitó a dejar singletons en todo
    assert len(meshes[1]) - 1 < n, "el DP no agrupó nada: el caso no discrimina"


# ---------------------------------------------------------------------------
# La tolerancia de la no-cancelacion: la MISMA en el DP y en el aviso (2026-08-23)
# ---------------------------------------------------------------------------
class _Log:
    def __init__(self):
        self.msgs = []

    def warning(self, m, **kw):
        self.msgs.append(m)

    def info(self, m, **kw):
        pass


def _un_orden(rho, n=2, var=0.04):
    """`n` bins iguales con correlacion `rho` entre vecinos, un solo orden."""
    edges = np.linspace(1.0e6, 2.0e6, n + 1)
    a = np.ones(n)
    C = np.eye(n) * var
    for k in range(n - 1):
        C[k, k + 1] = C[k + 1, k] = rho * var
    return edges, np.diff(edges), a, C


def test_el_DP_FUSIONA_el_par_que_esta_EXACTAMENTE_en_la_frontera():
    """Dos bins INCORRELADOS caen justo encima de la restriccion, y pasan.

    ⚑ ES EL CASO QUE GENERABA LOS 121 AVISOS.  Con anchuras iguales,

        g  = w^2 (B11 + B22 + 2 B12)      vv = w^2 (B11 + B22)

    o sea ``g == vv`` EXACTO cuando ``B12 = 0``.  Fusionar ahi no destruye nada
    -- la varianza fusionada ES la del promediado independiente -- y ahorra un
    grupo, asi que el DP tiene que admitirlo.  Negarse cuesta: sobre la run
    103R4 son 7 820 -> 8 432 parametros.
    """
    edges, w, a, C = _un_orden(0.0)
    cuts = physical_solve_order(C * np.outer(a, a), a, w, None)
    assert len(cuts) - 1 == 1, (
        f"el DP se nego a fusionar un par que no destruye nada: "
        f"{len(cuts)-1} grupos")


def test_el_DP_VETA_el_par_anticorrelado():
    """Y la restriccion sigue mordiendo donde tiene que morder: rho = -1."""
    edges, w, a, C = _un_orden(-1.0)
    cuts = physical_solve_order(C * np.outer(a, a), a, w, None)
    assert len(cuts) - 1 == 2, (
        f"el DP fusiono un par con rho = -1: {len(cuts)-1} grupos")


def test_el_aviso_NO_salta_por_el_ultimo_bit():
    """⛔ 121 avisos en la run 103R4 y los 121 eran redondeo.

    El DP admitia con ``- 1e-12|vv|`` y el aviso marcaba con ``- 1e-300``: los
    segmentos que el DP apoya sobre la restriccion -- que son muchos, porque
    MINIMIZA grupos -- caian del otro lado al recalcularlos por el otro camino
    aritmetico.  Medido: cociente ``indep/own = 1`` a DIEZ cifras en los 121, y
    ninguno por encima de 1,001.  Aqui se reproduce con un par incorrelado, que
    es exactamente la frontera.
    """
    n = 4
    edges = np.linspace(1.0e6, 2.0e6, n + 1)
    a = np.ones(n * L)
    rel = np.zeros((n * L, n * L))
    for l in range(L):
        i = np.arange(n) * L + l
        rel[np.ix_(i, i)] = np.eye(n) * 0.04          # incorrelado: la frontera
    meshes = {l: edges[::2] for l in range(1, L + 1)}
    log = _Log()
    collapse_relative_per_order(edges, rel, a, L, meshes, variant="max",
                                logger=log)
    assert not log.msgs, f"el aviso salto por redondeo: {log.msgs}"


def test_el_aviso_SI_salta_con_una_cancelacion_de_verdad_y_da_la_severidad():
    """Y cuando la cancelacion es real, el aviso tiene que llevar CUANTO.

    Un recuento a secas no distingue un 1 + 1e-16 de un x10, que es lo que hacia
    ilegible el mensaje anterior ("la malla no los vetó" para 121 grupos que la
    malla habia vetado correctamente).
    """
    edges, a, rel, meshes = _caso_cancelacion()
    log = _Log()
    collapse_relative_per_order(edges, rel, a, L, meshes, variant="max",
                                logger=log)
    assert log.msgs, "el colapso no aviso de una cancelacion exacta"
    m = " ".join(log.msgs)
    assert "indep/own" in m and "max" in m, f"el aviso no lleva severidad: {m}"


def test_un_bin_mas_ancho_que_su_sigma_E_no_rinde_el_tramo():
    """⛔ EL FALLO QUE COSTABA 3 636 PARAMETROS DE 10 428.

    Si un bin es mas ancho que su propia `sigma_E`, su SINGLETON deja de ser
    admisible; entonces no existe NINGUNA particion admisible del tramo, `f[n]`
    sale infinito y el DP cae en su rama de emergencia -- todo el tramo a
    singletons. En la run 103R4 son 82 bins por encima de 3,47 MeV, y como a_2
    no cambia de signo nunca su unico tramo son los 1738 bins: el orden entero
    salia sin agrupar (1738 grupos donde el criterio permite 749).

    Aqui: 20 bins que la resolucion permite fusionar de tres en tres, y UNO al
    final que es el doble de ancho que su sigma_E. El tramo tiene que seguir
    agrupando; el bin ancho se queda solo, que es lo correcto.
    """
    n = 21
    edges = np.concatenate([np.arange(n) * 1.0e3, [20.0e3 + 5.0e3]]) + 1.0e6
    w = np.diff(edges)
    sE = np.full(n, 3.0e3)
    sE[-1] = 2.5e3                       # el ultimo bin (5 keV) es MAS ancho
    assert w[-1] > sE[-1], "el caso no reproduce el fallo"
    a = np.ones(n)
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    B = (0.999 ** d) * 0.04

    cuts = physical_solve_order(B, a, w, sE, res_factor=1.0)
    assert len(cuts) - 1 < n, (
        f"el DP se rindio a singletons por un solo bin ancho: {len(cuts)-1} de {n}")
    # y el bin ancho es su propio grupo, no se ha colado en el vecino
    assert cuts[-2] == n - 1, f"el bin ancho quedo fusionado: cortes {cuts}"
    # ningun grupo de dos o mas se pasa de la resolucion
    for lo_, hi_ in zip(cuts[:-1], cuts[1:]):
        if hi_ - lo_ > 1:
            assert w[lo_:hi_].sum() <= sE[lo_:hi_].min() + 1e-9, (lo_, hi_)


def test_la_resolucion_SIGUE_acotando_los_grupos_de_dos_o_mas():
    """El control del test de arriba: la exencion es SOLO del singleton.

    Sin esto, «un bin siempre cabe en si mismo» podria haberse implementado
    aflojando la condicion entera y la resolucion dejaria de acotar nada.
    """
    edges, w, sE, a, rel = _case()
    n = len(w)
    i = np.arange(n) * L
    B = rel[np.ix_(i, i)] * np.outer(a[i], a[i])
    cuts = physical_solve_order(B, a[i], w, sE, res_factor=1.0)
    for lo_, hi_ in zip(cuts[:-1], cuts[1:]):
        if hi_ - lo_ > 1:
            assert w[lo_:hi_].sum() <= sE[lo_:hi_].min() + 1e-9, (lo_, hi_)
    assert len(cuts) - 1 < n, "el caso no discrimina: no agrupo nada"
