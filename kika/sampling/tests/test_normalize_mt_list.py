"""Unit tests for ``kika.sampling.utils.normalize_mt_list``.

These tests exercise the helper that backs per-file MT selection in the three
perturbation entry points (``perturb_ACE_files``, ``perturb_ENDF_files``,
``perturb_seprate_ACE_files``). The helper is pure Python and has no heavy
dependencies, so it can be run in any environment.
"""
import pytest

from kika.sampling.utils import normalize_mt_list


class TestNormalizeMtListFlatInput:
    def test_broadcasts_flat_list_to_every_unit(self):
        assert normalize_mt_list([1, 2, 18], 3) == [[1, 2, 18], [1, 2, 18], [1, 2, 18]]

    def test_empty_flat_list_expands_to_empty_per_unit(self):
        assert normalize_mt_list([], 2) == [[], []]

    def test_none_is_treated_as_empty(self):
        assert normalize_mt_list(None, 2) == [[], []]

    def test_returned_inner_lists_are_independent_copies(self):
        mts = [1, 2]
        result = normalize_mt_list(mts, 2)
        result[0].append(99)
        # The original input and the second inner list stay unchanged.
        assert mts == [1, 2]
        assert result[1] == [1, 2]


class TestNormalizeMtListNestedInput:
    def test_accepts_list_of_lists_matching_unit_count(self):
        assert normalize_mt_list([[1, 2], [102], []], 3) == [[1, 2], [102], []]

    def test_rejects_nested_input_with_wrong_length(self):
        with pytest.raises(ValueError, match="2 entries but 3"):
            normalize_mt_list([[1], [2]], 3)

    def test_error_message_uses_provided_unit_label(self):
        with pytest.raises(ValueError, match="ZAIDs"):
            normalize_mt_list([[1], [2]], 3, unit_label="ZAIDs")

    def test_rejects_mixed_flat_and_nested(self):
        with pytest.raises(TypeError, match="flat List\\[int\\] or a List\\[List\\[int\\]\\]"):
            normalize_mt_list([1, [2, 3]], 2)

    def test_rejects_non_integer_inner_values(self):
        with pytest.raises(TypeError, match="mt_list\\[1\\]"):
            normalize_mt_list([[1, 2], [3, "x"]], 2)


class TestPerturbSignaturesAcceptBothShapes:
    """Signature-level guard: the three entry points must declare the new union
    type so that static tools and downstream callers know both shapes are
    supported. We inspect the function annotations without calling them, since
    they require heavy ACE/ENDF fixtures to actually run."""

    def test_perturb_ACE_files_signature(self):
        from kika.sampling.ace_perturbation import perturb_ACE_files
        annotation = perturb_ACE_files.__annotations__["mt_list"]
        assert "List[int]" in str(annotation)
        assert "List[List[int]]" in str(annotation)

    def test_perturb_ENDF_files_signature(self):
        from kika.sampling.endf_perturbation import perturb_ENDF_files
        annotation = perturb_ENDF_files.__annotations__["mt_list"]
        assert "List[int]" in str(annotation)
        assert "List[List[int]]" in str(annotation)

    def test_perturb_seprate_ACE_files_signature(self):
        from kika.sampling.ace_perturbation_separate import perturb_seprate_ACE_files
        annotation = perturb_seprate_ACE_files.__annotations__["mt_list"]
        assert "List[int]" in str(annotation)
        assert "List[List[int]]" in str(annotation)
