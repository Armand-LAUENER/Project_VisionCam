"""
tests/test_tracker_backends.py

Couvre la bascule entre les deux implémentations de l'association de tracks
(`config.TRACKER_BACKEND`) et l'adaptateur qui branche le crate Rust
`deepsort_rs` sur le pipeline Body-First.

Le vrai `deepsort_rs` est utilisé : c'est lui qu'on veut voir répondre. Seul
l'embedder MobileNetV2 est remplacé — il chargerait torch et le GPU, et le
tracking ne dépend de lui que par les vecteurs qu'il produit.

Aucun stub n'est posé dans `sys.modules` au niveau module : il y resterait pour
tous les fichiers de test collectés ensuite (cf. commit cadc2a7).

Lancer : pytest tests/test_tracker_backends.py -v
"""

import importlib
import sys
import types

import numpy as np
import pytest

import config
from core import tracker_backends
from core.tracker_backends import (
    PythonDeepSortBackend,
    RustDeepSortBackend,
    build_body_tracker,
)

EMBED_DIM = 8


# ─────────────────────────────────────────────────────────────────────────────
# Doublures
# ─────────────────────────────────────────────────────────────────────────────

class FakeEmbedder:
    """Embedding dérivé du contenu du crop : deux personnes de couleurs
    différentes reçoivent des vecteurs différents, la même personne d'une frame
    à l'autre reçoit le même. Suffisant pour que la cascade d'apparence ait un
    signal exploitable, sans charger MobileNetV2."""

    def __init__(self, *_args, **_kwargs):
        self.batches: list[list[np.ndarray]] = []

    def predict(self, crops):
        self.batches.append(crops)
        out = []
        for crop in crops:
            colour = np.asarray(crop, dtype=np.float32).reshape(-1, 3).mean(axis=0)
            out.append(np.resize(colour / 255.0, EMBED_DIM).astype(np.float32))
        return out


class SpyDeepSort:
    """Remplace `DeepSort` dans le module testé. `crop_bb` reproduit le
    découpage de la référence et enregistre ses appels, ce qui permet de
    vérifier que le backend Rust lui délègue bien les crops au lieu d'en
    recalculer de son côté."""

    calls: list[tuple] = []

    @staticmethod
    def crop_bb(frame, raw_dets, instance_masks=None):
        SpyDeepSort.calls.append((frame, list(raw_dets)))
        crops = []
        height, width = frame.shape[:2]
        for det in raw_dets:
            left, top, w, h = (int(v) for v in det[0])
            crops.append(
                frame[max(0, top):min(height, top + h), max(0, left):min(width, left + w)]
            )
        return crops, None


@pytest.fixture
def rust_backend(monkeypatch):
    """`RustDeepSortBackend` complet — vrai tracker Rust, faux embedder."""
    SpyDeepSort.calls = []
    fake_module = types.ModuleType("deep_sort_realtime.embedder.embedder_pytorch")
    fake_module.MobileNetv2_Embedder = FakeEmbedder
    monkeypatch.setitem(
        sys.modules, "deep_sort_realtime.embedder.embedder_pytorch", fake_module
    )
    monkeypatch.setattr(tracker_backends, "DeepSort", SpyDeepSort)
    monkeypatch.setattr(config, "DEEPSORT_EMBEDDER", "mobilenet")
    monkeypatch.setattr(config, "DEEPSORT_MAX_AGE", 5)
    monkeypatch.setattr(config, "DEEPSORT_N_INIT", 3)
    return RustDeepSortBackend()


def make_frame(people: dict[tuple, int] | None = None) -> np.ndarray:
    """Frame 640×480 noire, avec un aplat de gris par personne pour que
    l'embedder factice produise des vecteurs distincts."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for (x, y, w, h), shade in (people or {}).items():
        frame[y:y + h, x:x + w] = shade
    return frame


def detection(ltwh, conf=0.9):
    """4-tuple au format attendu par le pipeline (cf. `_detect_bodies`)."""
    return (list(ltwh), conf, "person", {"nose": None})


# ─────────────────────────────────────────────────────────────────────────────
# Sélection du backend — c'est le chemin de retour arrière
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendSelection:

    def test_python_is_the_default(self, monkeypatch):
        """Le défaut ne doit pas bouger : l'ancien tracker reste en place tant
        que TRACKER_BACKEND n'est pas explicitement passé à 'rust'. `.env` est
        neutralisé le temps du rechargement, sinon le test dépendrait de ce que
        la machine a mis dedans."""
        import dotenv

        monkeypatch.delenv("TRACKER_BACKEND", raising=False)
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
        try:
            assert importlib.reload(config).TRACKER_BACKEND == "python"
        finally:
            monkeypatch.undo()
            importlib.reload(config)

    def test_builds_python_backend(self, monkeypatch):
        monkeypatch.setattr(config, "TRACKER_BACKEND", "python")
        monkeypatch.setattr(PythonDeepSortBackend, "__init__", lambda self: None)
        assert isinstance(build_body_tracker(), PythonDeepSortBackend)

    def test_builds_rust_backend(self, monkeypatch):
        monkeypatch.setattr(config, "TRACKER_BACKEND", "rust")
        monkeypatch.setattr(RustDeepSortBackend, "__init__", lambda self: None)
        assert isinstance(build_body_tracker(), RustDeepSortBackend)

    def test_unknown_backend_fails_loudly(self, monkeypatch):
        """Une faute de frappe dans .env doit arrêter le démarrage, pas
        retomber silencieusement sur un backend au hasard."""
        monkeypatch.setattr(config, "TRACKER_BACKEND", "rustt")
        with pytest.raises(ValueError, match="rustt"):
            build_body_tracker()

    def test_rust_backend_refuses_an_embedder_it_cannot_reproduce(self, monkeypatch):
        monkeypatch.setattr(config, "DEEPSORT_EMBEDDER", "torchreid")
        with pytest.raises(ValueError, match="torchreid"):
            RustDeepSortBackend()


# ─────────────────────────────────────────────────────────────────────────────
# Adaptateur Rust
# ─────────────────────────────────────────────────────────────────────────────

class TestRustBackendUpdate:

    def test_boxes_are_converted_from_ltwh_to_xyxy(self, rust_backend):
        """Le pipeline produit du (left, top, width, height), le crate attend
        du (x1, y1, x2, y2). Une personne immobile doit ressortir sur sa propre
        boîte : si la conversion manquait, le coin bas-droit serait faux."""
        ltwh = (100, 80, 60, 180)
        frame = make_frame({ltwh: 200})

        for _ in range(config.DEEPSORT_N_INIT):
            tracks = rust_backend.update([detection(ltwh)], frame)

        confirmed = [t for t in tracks if t.is_confirmed()]
        assert len(confirmed) == 1
        np.testing.assert_allclose(
            confirmed[0].to_ltrb(), [100, 80, 160, 260], atol=1e-3
        )

    def test_crops_are_delegated_to_the_reference_implementation(self, rust_backend):
        """Les embeddings ne sont comparables entre les deux backends que si
        les crops le sont — d'où la réutilisation de `DeepSort.crop_bb`."""
        ltwh = (10, 20, 40, 90)
        frame = make_frame({ltwh: 150})

        rust_backend.update([detection(ltwh)], frame)

        assert len(SpyDeepSort.calls) == 1
        called_frame, called_dets = SpyDeepSort.calls[0]
        assert called_frame is frame
        assert called_dets == [detection(ltwh)]

    def test_degenerate_boxes_never_reach_the_tracker(self, rust_backend):
        """Une boîte de largeur ou de hauteur nulle ferait planter le resize de
        l'embedder. deep_sort_realtime les écarte ; on fait pareil, et la boîte
        valide ne doit pas être décalée au passage."""
        valid = (300, 100, 50, 120)
        frame = make_frame({valid: 220})
        dets = [
            detection((0, 0, 0, 50)),
            detection(valid),
            detection((500, 50, 30, 0)),
        ]

        for _ in range(config.DEEPSORT_N_INIT):
            tracks = rust_backend.update(dets, frame)

        _, called_dets = SpyDeepSort.calls[0]
        assert called_dets == [detection(valid)]
        confirmed = [t for t in tracks if t.is_confirmed()]
        assert len(confirmed) == 1
        np.testing.assert_allclose(
            confirmed[0].to_ltrb(), [300, 100, 350, 220], atol=1e-3
        )

    def test_frame_without_detection_still_ages_tracks(self, rust_backend):
        """Sans détection il faut quand même appeler le tracker : c'est ce qui
        fait vieillir les pistes et finit par les supprimer à max_age."""
        ltwh = (200, 150, 70, 160)
        frame = make_frame({ltwh: 180})
        empty = make_frame()

        for _ in range(config.DEEPSORT_N_INIT):
            rust_backend.update([detection(ltwh)], frame)

        still_there = rust_backend.update([], empty)
        assert [t.track_id for t in still_there if t.is_confirmed()] == [1]

        for _ in range(config.DEEPSORT_MAX_AGE + 1):
            tracks = rust_backend.update([], empty)
        assert tracks == []

    def test_identity_survives_a_short_occlusion(self, rust_backend):
        """Le point qui justifie DeepSORT plutôt qu'un simple IoU : une personne
        qui disparaît quelques frames doit revenir avec le même track_id."""
        ltwh = (120, 90, 60, 170)
        frame = make_frame({ltwh: 210})

        for _ in range(config.DEEPSORT_N_INIT):
            tracks = rust_backend.update([detection(ltwh)], frame)
        original_id = [t.track_id for t in tracks if t.is_confirmed()][0]

        for _ in range(2):
            rust_backend.update([], make_frame())

        tracks = rust_backend.update([detection(ltwh)], frame)
        assert [t.track_id for t in tracks if t.is_confirmed()] == [original_id]

    def test_two_people_keep_distinct_ids(self, rust_backend):
        left = (50, 100, 60, 180)
        right = (400, 100, 60, 180)
        frame = make_frame({left: 90, right: 230})

        for _ in range(config.DEEPSORT_N_INIT):
            tracks = rust_backend.update([detection(left), detection(right)], frame)

        confirmed = sorted(
            (t for t in tracks if t.is_confirmed()), key=lambda t: t.to_ltrb()[0]
        )
        assert len({t.track_id for t in confirmed}) == 2
        np.testing.assert_allclose(confirmed[0].to_ltrb(), [50, 100, 110, 280], atol=1e-3)
        np.testing.assert_allclose(confirmed[1].to_ltrb(), [400, 100, 460, 280], atol=1e-3)
