"""
tests/test_recognition_threshold_study.py

Couvre l'intervalle de confiance publié avec les taux de mauvais noms.

Lancer : pytest tests/test_recognition_threshold_study.py -v
"""

import pytest

from tools.recognition_threshold_study import wilson


class TestWilson:

    def test_zero_errors_is_not_a_zero_rate(self):
        """0 sur 300 ne prouve pas 0 % : la borne haute reste vers 3/n (règle de trois)."""
        low, high = wilson(0, 300)

        assert low == 0.0
        assert 2.5 / 300 < high < 4 / 300

    def test_interval_contains_the_observed_rate(self):
        low, high = wilson(18, 607)

        assert low < 18 / 607 < high
        assert low == pytest.approx(0.0188, abs=1e-3)
        assert high == pytest.approx(0.0465, abs=1e-3)

    def test_no_sample_gives_no_information(self):
        assert wilson(0, 0) == (0.0, 1.0)
