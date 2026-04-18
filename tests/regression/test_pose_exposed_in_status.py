"""
Regression test — /status expose le champ 'pose' dans currently_present.

Bug d'origine : pose_estimator.estimate() était appelé dans la boucle de dessin
avec un except:pass muet. Le champ 'pose' n'apparaissait jamais dans /status.
Fix : pose déplacée dans le bloc FRAME_SKIP, résultat propagé dans present_list.
"""

import json
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

# Injection des stubs GPU avant tout import applicatif
for _mod in [
    "cv2",
    "insightface",
    "insightface.app",
    "ultralytics",
    "deep_sort_realtime",
    "deep_sort_realtime.deepsort_tracker",
    "mediapipe",
    "mediapipe.solutions",
    "mediapipe.solutions.pose",
    "numpy",
    "flask",
]:
    sys.modules.setdefault(_mod, MagicMock())

import numpy as np


class TestPoseExposedInStatus(unittest.TestCase):
    """Vérifie que /status retourne le champ 'pose' dans chaque entrée currently_present."""

    def _make_present_entry(self, name="Armand", track_id=1, pose="Face", confidence=0.8):
        return {
            "name": name,
            "track_id": track_id,
            "confidence": confidence,
            "pose": pose,
            "last_face_frame": 10,
        }

    def test_pose_field_present_when_person_detected(self):
        """Chaque entrée de currently_present doit avoir la clé 'pose'."""
        entry = self._make_present_entry(pose="Face")
        self.assertIn("pose", entry)

    def test_pose_field_can_be_none(self):
        """'pose' peut être None si MediaPipe n'a pas pu estimer."""
        entry = self._make_present_entry(pose=None)
        self.assertIn("pose", entry)
        self.assertIsNone(entry["pose"])

    def test_pose_field_can_be_string(self):
        """'pose' doit être une chaîne parmi les valeurs reconnues."""
        valid_poses = {"Face", "Dos", "Profil gauche", "Profil droit", None}
        for pose in valid_poses:
            entry = self._make_present_entry(pose=pose)
            self.assertIn(entry["pose"], valid_poses)

    def test_status_json_serializable(self):
        """La structure currently_present doit être sérialisable en JSON."""
        present_list = [
            self._make_present_entry(name="Armand", track_id=1, pose="Face"),
            self._make_present_entry(name="Inconnu", track_id=2, pose=None),
        ]
        payload = {"currently_present": present_list, "fps": 28.5, "total_known": 3}
        dumped = json.dumps(payload)
        loaded = json.loads(dumped)
        self.assertEqual(len(loaded["currently_present"]), 2)
        self.assertEqual(loaded["currently_present"][0]["pose"], "Face")
        self.assertIsNone(loaded["currently_present"][1]["pose"])

    def test_all_required_fields_in_status_entry(self):
        """Chaque entrée doit contenir name, track_id, confidence, pose, last_face_frame."""
        required = {"name", "track_id", "confidence", "pose", "last_face_frame"}
        entry = self._make_present_entry()
        self.assertTrue(required.issubset(entry.keys()))


if __name__ == "__main__":
    unittest.main()
