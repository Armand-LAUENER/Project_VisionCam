"""
tests/test_nested_boxes.py

Couvre la suppression des boîtes emboîtées (roadmap 1.8) : une personne assise
reçoit parfois une seconde boîte, contenue à ~93 % dans la première (haut du
corps dans corps entier), que le NMS laisse passer (IoU ~0,6). Elle fait
naître une seconde piste, et le nom saute de l'une à l'autre.

Lancer : pytest tests/test_nested_boxes.py -v
"""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault('ultralytics', MagicMock())
sys.modules.setdefault('insightface', MagicMock())
sys.modules.setdefault('insightface.app', MagicMock())

from core.face_body_tracker import suppress_nested_boxes  # noqa: E402


def det(x, y, w, h, conf):
    return ([x, y, w, h], conf, 'person', {'nose': None})


class TestSuppressNestedBoxes:

    def test_box_inside_another_is_dropped(self):
        """Mesuré sur CHIRLA : IoU 0,56-0,64, contenue à 93-94 %."""
        body = det(100, 100, 200, 350, 0.9)
        upper = det(110, 100, 180, 210, 0.7)

        assert suppress_nested_boxes([body, upper]) == [body]

    def test_the_most_confident_box_is_kept(self):
        body = det(100, 100, 200, 350, 0.6)
        upper = det(110, 100, 180, 210, 0.8)

        assert suppress_nested_boxes([body, upper]) == [upper]

    def test_two_people_side_by_side_are_kept(self):
        a, b = det(100, 100, 200, 350, 0.9), det(330, 100, 200, 350, 0.8)

        assert suppress_nested_boxes([a, b]) == [a, b]

    def test_small_person_behind_is_kept(self):
        """Contenue mais petite (IoU faible) : une autre personne, plus loin derrière."""
        front = det(100, 100, 300, 500, 0.9)
        behind = det(200, 150, 80, 160, 0.7)

        assert suppress_nested_boxes([front, behind]) == [front, behind]

    def test_disabled(self):
        body, upper = det(100, 100, 200, 350, 0.9), det(110, 100, 180, 210, 0.7)

        assert suppress_nested_boxes([body, upper], min_contained=0) == [body, upper]
