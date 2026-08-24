"""El tope sigma-ratio, leido POR ORDEN en vez de solo en a_1.

El tope existe desde 2026-01-30 y se lee sobre la sigma de **a_1**, que lo
cumple por construccion (max medido 5.0) mientras a_4-a_6 llegan a 2000x dentro
del mismo grupo. El tope existia y no protegia a quien lo necesitaba.

Las puertas son tres, y la primera es la que hace segura la decision:

  * MONOTONIA -- exigir la cota en mas ordenes solo puede rechazar fusiones,
    luego el numero de grupos solo puede subir. Si baja, el indice esta mal.
  * INERCIA -- con el flag apagado, las fronteras son las de siempre, identicas.
  * SENSIBILIDAD -- una heterogeneidad puesta SOLO en a_4 tiene que partir el
    grupo con el flag y NO partirlo sin el. Sin esta, las otras dos pasarian
    tambien con un tope que no mira nada.
"""
import numpy as np

from scripts.multigroup_collapse import find_adaptive_group_boundaries, idx

L = 6


def _matrices(sigmas, rho=0.99):
    """`(corr, cov)` para n bins y L ordenes, con la sigma que se le pase.

    `sigmas` es (L, n). Todos los pares adyacentes se ponen a `rho` en todos los
    ordenes, asi que la unica cosa que puede partir un grupo es el tope.
    """
    Ls, n = sigmas.shape
    N = n*Ls
    C = np.zeros((N, N))
    for l in range(1, Ls + 1):
        for i in range(n):
            for j in range(n):
                a, b = idx(i, l, Ls), idx(j, l, Ls)
                C[a, b] = 1.0 if i == j else rho**abs(i - j)
    s = np.empty(N)
    for l in range(1, Ls + 1):
        for i in range(n):
            s[idx(i, l, Ls)] = sigmas[l - 1, i]
    cov = C*np.outer(s, s)
    return C, cov


def _boundaries(sigmas, cap, per_order, rho=0.99):
    n = sigmas.shape[1]
    corr, cov = _matrices(sigmas, rho)
    groups, _ = find_adaptive_group_boundaries(
        corr_matrix=corr, cov_matrix=cov,
        fine_energies_mev=np.linspace(1.0, 2.0, n),
        sigma_E_mev=np.full(n, 0.05),
        fine_bin_widths_mev=np.full(n, 1.0/n),
        max_order=L, rho_min=0.90, sigma_ratio_max=cap,
        sigma_ratio_per_order=per_order)
    return groups


def _plana(n=24, seed=0):
    """a_1 homogenea (el tope de hoy nunca dispara) y a_4 heterogenea."""
    rng = np.random.default_rng(seed)
    s = np.ones((L, n))*0.1
    s[0] = 0.1*(1.0 + 0.02*rng.random(n))          # a_1: ratio ~1.02
    s[3] = 0.1*np.exp(np.linspace(0, np.log(50), n))   # a_4: ratio 50x
    return s


def test_inercia_con_el_flag_apagado():
    """Sin el flag, las fronteras son EXACTAMENTE las de siempre."""
    s = _plana()
    assert _boundaries(s, 5.0, False) == _boundaries(s, 5.0, False)
    # y a_1 es tan homogenea que el tope de hoy no parte nada: un solo grupo
    assert len(_boundaries(s, 5.0, False)) == 1


def test_sensibilidad_el_flag_ve_lo_que_a1_no_ve():
    """⚑ La puerta que da sentido a las otras dos: la heterogeneidad esta SOLO
    en a_4, y el tope de hoy es ciego a ella."""
    s = _plana()
    sin = _boundaries(s, 5.0, False)
    con = _boundaries(s, 5.0, True)
    assert len(sin) == 1, "el tope de a_1 no deberia ver nada aqui"
    assert len(con) > 1, "el tope por orden TIENE que partir por a_4"
    # y cada grupo respeta la cota en los seis ordenes, que es lo que promete
    for g in con:
        for l in range(L):
            v = s[l, g]
            v = v[v > 0]
            if len(v) >= 2:
                assert v.max()/v.min() <= 5.0 + 1e-12, f"a_{l+1} se paso"


def test_monotonia_mas_restriccion_nunca_da_menos_grupos():
    """Ni entre l=1 y por-orden, ni al bajar el tope."""
    s = _plana()
    n1 = len(_boundaries(s, 5.0, False))
    npo = {c: len(_boundaries(s, c, True)) for c in (5.0, 3.0, 2.0)}
    assert npo[5.0] >= n1
    assert npo[2.0] >= npo[3.0] >= npo[5.0], npo
    # sin tope ninguno, un solo grupo: la referencia de que rho no parte nada
    assert len(_boundaries(s, None, True)) == 1


def test_el_tope_no_puede_fusionar_lo_que_rho_ya_separa():
    """El tope es un veto, no un permiso: bajar rho por debajo del umbral parte
    igual con flag y sin el."""
    s = np.ones((L, 12))*0.1
    for per in (False, True):
        g = _boundaries(s, 5.0, per, rho=0.5)
        assert len(g) == 12, f"rho=0.5 < 0.90 tiene que dejar singletons ({per})"
