"""Test file names must be unique across tests/ (no __init__.py files, so pytest imports each by its base name):
a duplicate stops collection of the whole folder, as the sim image runs it (`pytest tests`)."""
from collections import Counter
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def test_test_file_names_are_unique():
    names = Counter(p.name for p in TESTS.rglob("test_*.py"))
    assert not [n for n, c in names.items() if c > 1]
