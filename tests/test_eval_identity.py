"""
tests/test_eval_identity.py

Couvre la métrique d'identité de bout en bout (tools/eval_identity.py) : pour
chaque personne annotée, part du temps où l'application affiche le bon nom,
un mauvais nom, « Inconnu » ou rien, et délai avant le premier bon nom.
Vérité terrain et journal des pistes synthétiques.

Lancer : pytest tests/test_eval_identity.py -v
"""

import pytest

from tools.eval_identity import evaluate, load_tracks

FPS = 10.0
BOX = (100, 100, 200, 400)          # vérité terrain : x1, y1, x2, y2
TRACK = (100, 100, 100, 300)        # même boîte dans le journal : x, y, l, h
FAR = (900, 100, 100, 300)          # boîte sans recouvrement


def scene(frames, names, gid=2, box=TRACK):
    """GT : `gid` présent sur `frames` ; pistes : nom affiché par image (None = pas de piste)."""
    gt = {f: [(gid, *BOX)] for f in frames}
    tracks = {f: [(1, *box, name)] for f, name in zip(frames, names) if name is not None}
    return gt, tracks


class TestOutcomes:

    def test_correct_name_all_the_time(self):
        gt, tracks = scene(range(1, 11), ["Alice"] * 10)

        report = evaluate(gt, tracks, {2: "Alice"}, FPS)

        assert report.enrolled.as_dict() == {"frames": 10, "correct": 10, "wrong": 0,
                                             "unknown": 0, "untracked": 0}

    def test_each_outcome_is_counted(self):
        gt, tracks = scene(range(1, 9), ["Inconnu", "Inconnu", "Alice", "Alice",
                                         "Bob", None, "Alice", "Alice"])

        report = evaluate(gt, tracks, {2: "Alice"}, FPS)

        assert report.enrolled.as_dict() == {"frames": 8, "correct": 4, "wrong": 1,
                                             "unknown": 2, "untracked": 1}

    def test_track_elsewhere_is_not_a_match(self):
        gt, tracks = scene(range(1, 4), ["Alice"] * 3, box=FAR)

        report = evaluate(gt, tracks, {2: "Alice"}, FPS)

        assert report.enrolled.untracked == 3

    def test_unenrolled_person(self):
        """Non enrôlée : « Inconnu » est la bonne réponse, un nom est une erreur."""
        gt, tracks = scene(range(1, 6), ["Inconnu", "Inconnu", "Alice", "Inconnu", None], gid=14)

        report = evaluate(gt, tracks, {14: None}, FPS)

        assert report.unenrolled.as_dict() == {"frames": 5, "correct": 3, "wrong": 1,
                                               "unknown": 0, "untracked": 1}

    def test_rates(self):
        gt, tracks = scene(range(1, 5), ["Alice", "Alice", "Alice", "Bob"])

        rates = evaluate(gt, tracks, {2: "Alice"}, FPS).enrolled.rates()

        assert rates["correct"] == pytest.approx(0.75)
        assert rates["wrong"] == pytest.approx(0.25)


class TestTimeToName:

    def test_delay_before_the_first_correct_name(self):
        gt, tracks = scene(range(1, 21), [None] * 3 + ["Inconnu"] * 5 + ["Alice"] * 12)

        report = evaluate(gt, tracks, {2: "Alice"}, FPS)

        assert report.time_to_name == [pytest.approx(0.8)]

    def test_each_return_is_a_new_appearance(self):
        """Absente plus d'une seconde : son retour se compte à part."""
        frames = list(range(1, 11)) + list(range(31, 41))
        names = ["Inconnu"] * 2 + ["Alice"] * 8 + ["Inconnu"] * 5 + ["Alice"] * 5
        gt, tracks = scene(frames, names)

        report = evaluate(gt, tracks, {2: "Alice"}, FPS)

        assert report.time_to_name == [pytest.approx(0.2), pytest.approx(0.5)]

    def test_never_named_appearance(self):
        gt, tracks = scene(range(1, 11), ["Inconnu"] * 10)

        report = evaluate(gt, tracks, {2: "Alice"}, FPS)

        assert report.time_to_name == [None]


def test_load_tracks(tmp_path):
    path = tmp_path / "tracks.csv"
    path.write_text("12,3,10,20,40,100,Alice\n12,7,200,30,60,120,Inconnu\n13,3,11,20,40,100,Alice\n")

    tracks = load_tracks(str(path))

    assert tracks[12] == [(3, 10.0, 20.0, 40.0, 100.0, "Alice"),
                          (7, 200.0, 30.0, 60.0, 120.0, "Inconnu")]
    assert len(tracks[13]) == 1
