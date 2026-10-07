"""
tests/test_eval_global_identity.py

Cohérence des noms entre caméras (tools/eval_global_identity.py), sur des vues
construites à la main : {caméra: {image: {identité: nom affiché ou None}}}.
"""

from tools.eval_global_identity import UNTRACKED, evaluate_global, shown_names

NAMES = {1: "Alice", 2: "Bob", 9: None}   # 9 : non enrôlée


def test_named_on_one_camera_counts_as_named_somewhere():
    report = evaluate_global({"a": {1: {1: "Alice"}}, "b": {1: {1: "Inconnu"}}}, NAMES)

    assert report.moments == 1 and report.named_somewhere == 1
    assert report.per_camera_named == {"a": 1, "b": 0}


def test_unknown_on_a_camera_while_named_elsewhere_is_propagatable():
    report = evaluate_global({"a": {1: {1: "Alice"}}, "b": {1: {1: "Inconnu"}}}, NAMES)

    assert report.tracked_unknown == 1 and report.propagatable == 1


def test_untracked_is_not_propagatable():
    report = evaluate_global({"a": {1: {1: "Alice"}}, "b": {1: {1: UNTRACKED}}}, NAMES)

    assert report.tracked_unknown == 0


def test_two_names_for_one_person_are_a_conflict_and_a_wrong_name():
    report = evaluate_global({"a": {1: {1: "Alice"}}, "b": {1: {1: "Bob"}}}, NAMES)

    assert report.conflicts == 1 and report.wrong_somewhere == 1 and report.named_somewhere == 1


def test_one_name_on_two_people_is_a_clone():
    report = evaluate_global({"a": {1: {1: "Alice"}}, "b": {1: {2: "Alice"}}}, NAMES)

    assert report.clones == 1
    assert report.wrong_somewhere == 1   # Bob porte le nom d'Alice


def test_the_same_name_on_the_same_person_twice_is_no_clone():
    report = evaluate_global({"a": {1: {1: "Alice"}}, "b": {1: {1: "Alice"}}}, NAMES)

    assert report.clones == 0 and report.conflicts == 0


def test_unenrolled_people_are_counted_apart():
    report = evaluate_global({"a": {1: {9: "Inconnu"}, 2: {9: "Alice"}}}, NAMES)

    assert report.moments == 0
    assert report.unenrolled_moments == 2 and report.unenrolled_named == 1


def test_shown_names_matches_tracks_to_annotations():
    gt = {1: [(1, 0, 0, 10, 20), (2, 100, 0, 110, 20)]}
    tracks = {1: [(7, 0, 0, 10, 20, "Alice")]}

    assert shown_names(gt, tracks) == {1: {1: "Alice", 2: UNTRACKED}}
