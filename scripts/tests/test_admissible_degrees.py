"""El filtro de candidatos fisicamente imposibles (27-ago-2026).

a_l = <P_l(mu)> con |P_l(mu)| <= 1 en mu in [-1, 1], luego |a_l| <= 1 para
CUALQUIER distribucion angular admisible.  Un ajuste por minimos cuadrados sin
restriccion puede salirse, y si se cuela en la mezcla AIC contamina el central
que va a MF4 y la varianza entre-modelos que va a MF34.

Origen: run 105T1, bins 1,558 y 1,560 MeV.  between_var(a_1) = 2,61 y 2,78
contra la cota fisica 1 - abar^2 = 0,98, y avg_a_1 volteado a -0,12 con todos
los vecinos en +0,28..+0,50.  De ahi salia sigma(a_1) = 1,667.
Ver docs/chi2-mf4/relative_peaks_absolute_check.md.
"""
import numpy as np
import pytest

from scripts.exfor_to_endf_research import admissible_degrees


def _info(**by_degree):
    """{grado: {'coeffs': [c0, c1, ...]}} en convencion legval."""
    return {int(d): {"coeffs": np.asarray(c, dtype=float)}
            for d, c in by_degree.items()}


def _coeffs_for(a, c0=0.25):
    """c_l a partir de los a_l pedidos:  a_l = c_l / ((2l+1) c0)."""
    return [c0] + [a_l * (2 * l + 1) * c0 for l, a_l in enumerate(a, start=1)]


class TestAdmissibleDegrees:

    def test_todos_admisibles_no_descarta_nada(self):
        info = _info(**{"1": _coeffs_for([0.3]),
                        "2": _coeffs_for([0.3, 0.2]),
                        "3": _coeffs_for([0.3, 0.2, 0.1])})
        kept, bad = admissible_degrees(info, sorted(info))
        assert kept == [1, 2, 3]
        assert bad == {}

    def test_descarta_el_que_se_sale_en_a1(self):
        # el grado 2 afirma <mu> = 1.4, que es imposible
        info = _info(**{"1": _coeffs_for([0.3]),
                        "2": _coeffs_for([1.4, 0.2]),
                        "3": _coeffs_for([0.3, 0.2, 0.1])})
        kept, bad = admissible_degrees(info, sorted(info))
        assert kept == [1, 3]
        assert set(bad) == {2}
        l_bad, a_bad = bad[2]
        assert l_bad == 1
        assert a_bad == pytest.approx(1.4)

    def test_descarta_en_cualquier_orden_no_solo_a1(self):
        info = _info(**{"3": _coeffs_for([0.2, 0.1, -1.9])})
        kept, bad = admissible_degrees(info, sorted(info))
        assert kept == []
        assert bad[3][0] == 3
        assert bad[3][1] == pytest.approx(-1.9)

    def test_el_limite_exacto_1_es_admisible(self):
        # |a_l| = 1 es alcanzable (delta en mu = +1): NO se descarta
        info = _info(**{"1": _coeffs_for([1.0])})
        kept, bad = admissible_degrees(info, sorted(info))
        assert kept == [1] and bad == {}

    def test_la_tolerancia_absorbe_el_ruido_de_coma_flotante(self):
        info = _info(**{"1": _coeffs_for([1.0 + 1e-12])})
        kept, _ = admissible_degrees(info, sorted(info), tol=1e-6)
        assert kept == [1]
        kept2, bad2 = admissible_degrees(info, sorted(info), tol=0.0)
        assert kept2 == [] and 1 in bad2

    def test_c0_no_positivo_es_imposible(self):
        # seccion eficaz integrada <= 0: imposible, y ademas rompe la
        # normalizacion. Se marca con l = 0 para distinguirlo.
        for c0 in (0.0, -0.25):
            info = _info(**{"2": [c0, 0.1, 0.05]})
            kept, bad = admissible_degrees(info, sorted(info))
            assert kept == [], f"c0={c0} deberia ser inadmisible"
            assert bad[2][0] == 0

    def test_coeficientes_no_finitos_son_imposibles(self):
        info = _info(**{"2": [0.25, np.nan, 0.05]})
        kept, bad = admissible_degrees(info, sorted(info))
        assert kept == [] and 2 in bad

    def test_grado_ausente_de_all_degrees_info_se_ignora(self):
        info = _info(**{"1": _coeffs_for([0.3])})
        kept, bad = admissible_degrees(info, [1, 2, 3])
        assert kept == [1]
        assert bad == {}

    def test_reproduce_el_caso_de_la_105T1(self):
        # El sintoma medido: la mezcla declaraba between_var = 2,78 con
        # abar = -0,116, cuando la cota fisica es 1 - abar^2 = 0,987.
        # Eso EXIGE un componente con |a_1| > 1; aqui se comprueba que el
        # filtro lo saca y que la cota se recupera.
        w = np.array([0.5, 0.5])
        a_ok, a_malo = 0.40, 1.55           # el segundo es imposible
        abar = float(w @ [a_ok, a_malo])
        between = float(w @ (np.array([a_ok, a_malo]) - abar) ** 2)
        assert between > 1.0 - abar ** 2    # el defecto, reproducido

        info = _info(**{"2": _coeffs_for([a_ok, 0.1]),
                        "3": _coeffs_for([a_malo, 0.1, 0.05])})
        kept, bad = admissible_degrees(info, sorted(info))
        assert kept == [2] and set(bad) == {3}

        # con un solo superviviente la varianza entre-modelos es 0, que
        # trivialmente cumple la cota
        abar2 = a_ok
        assert 0.0 <= 1.0 - abar2 ** 2
