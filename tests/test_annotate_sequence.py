"""
tests/test_annotate_sequence.py

Couvre la logique de tools/annotate_sequence.py et le format des fichiers de
tools/record_sequence.py, sans fenêtre ni caméra : ce sont eux qui décident de
ce que tools/sweep_deepsort.py et tools/eval_mot.py liront ensuite.

Lancer : pytest tests/test_annotate_sequence.py -v
"""

from tools import eval_mot
from tools.annotate_sequence import (
    box_at,
    delete_track,
    load_gt,
    next_event,
    save_gt,
    swap_ids,
)
from tools.record_sequence import write_seqinfo


def rows_for(*tracks):
    """tracks : (identité, première image, dernière image, gauche)."""
    return [[frame, track_id, left, 10.0, 50.0, 120.0]
            for track_id, first, last, left in tracks
            for frame in range(first, last + 1)]


def ids_at(rows, frame):
    return sorted(track_id for f, track_id, *_ in rows if f == frame)


def track_of(rows, left):
    return {frame: track_id for frame, track_id, box_left, *_ in rows if box_left == left}


class TestSwapIds:

    def test_fixes_an_identity_switch_from_the_given_frame(self):
        # Deux personnes (gauche=0 et gauche=200) échangent leurs numéros à l'image 5.
        rows = [[f, 1 if f < 5 else 2, 0.0, 10.0, 50.0, 120.0] for f in range(1, 9)]
        rows += [[f, 2 if f < 5 else 1, 200.0, 10.0, 50.0, 120.0] for f in range(1, 9)]

        fixed = swap_ids(rows, 2, 1, from_frame=5)

        assert set(track_of(fixed, 0.0).values()) == {1}
        assert set(track_of(fixed, 200.0).values()) == {2}

    def test_renames_to_a_free_number_without_touching_earlier_frames(self):
        rows = rows_for((1, 1, 10, 0.0))

        renamed = swap_ids(rows, 1, 7, from_frame=6)

        assert track_of(renamed, 0.0) == {f: (1 if f < 6 else 7) for f in range(1, 11)}

    def test_never_puts_the_same_id_twice_on_a_frame(self):
        rows = rows_for((1, 1, 10, 0.0), (2, 1, 10, 200.0))

        swapped = swap_ids(rows, 1, 2, from_frame=3)

        for frame in range(1, 11):
            assert ids_at(swapped, frame) == [1, 2]


class TestDeleteTrack:

    def test_from_this_frame_onward(self):
        rows = rows_for((1, 1, 10, 0.0), (2, 1, 10, 200.0))

        kept = delete_track(rows, 2, from_frame=4)

        assert ids_at(kept, 3) == [1, 2]
        assert ids_at(kept, 4) == [1]

    def test_this_frame_only(self):
        rows = rows_for((1, 1, 10, 0.0))

        kept = delete_track(rows, 1, from_frame=4, only_this_frame=True)

        assert ids_at(kept, 4) == []
        assert ids_at(kept, 5) == [1]


class TestNavigation:

    def test_box_at_prefers_the_smallest_box(self):
        rows = [[1, 1, 1.0, 1.0, 300.0, 300.0], [1, 2, 51.0, 51.0, 40.0, 40.0]]

        assert box_at(rows, 1, 60, 60) == 2
        assert box_at(rows, 1, 10, 10) == 1
        assert box_at(rows, 1, 400, 400) is None

    def test_next_event_stops_where_an_identity_appears_or_leaves(self):
        rows = rows_for((1, 1, 20, 0.0), (2, 8, 12, 200.0))

        assert next_event(rows, 1, 20) == 8
        assert next_event(rows, 8, 20) == 13
        assert next_event(rows, 13, 20) == 20


class TestFilesAreReadableByTheEvaluation:

    def test_saved_ground_truth_round_trips_through_eval_mot(self, tmp_path):
        rows = rows_for((3, 1, 4, 11.0), (5, 2, 4, 211.0))
        path = str(tmp_path / "gt" / "gt.txt")

        save_gt(path, rows)

        assert sorted(load_gt(path)) == sorted(rows)
        gt = eval_mot.load_ground_truth(path)
        assert len(gt) == len(rows)
        # motmetrics ramène les boîtes en base 0 au chargement.
        assert gt.loc[(1, 3), "X"] == 10.0

    def test_frame_path_handles_six_and_eight_digit_names(self, tmp_path):
        (tmp_path / "img1").mkdir()
        (tmp_path / "img1" / "000007.jpg").write_bytes(b"")
        (tmp_path / "img1" / "00000009.jpg").write_bytes(b"")

        assert eval_mot.frame_path(str(tmp_path), 7).endswith("img1/000007.jpg")
        assert eval_mot.frame_path(str(tmp_path), 9).endswith("img1/00000009.jpg")

    def test_seqinfo_gives_the_sequence_length(self, tmp_path):
        write_seqinfo(str(tmp_path / "seqinfo.ini"), "essai", 15, 42, 1920, 1080)

        assert eval_mot.sequence_length(str(tmp_path)) == 42
