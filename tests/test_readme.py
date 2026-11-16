"""The README's numbers must match the published site data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = json.loads((ROOT / "web" / "data" / "site.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_results_match_the_data():
    t, pc, g = SITE["test"], SITE["per_class"], SITE["gender"]
    assert f"| Both classes | {t['precision']:.3f} | {t['recall']:.3f} | {t['map50']:.3f} | {t['map50_95']:.3f} |" in README
    assert f"| Women | | | {pc['Female']['map50']:.3f} | {pc['Female']['map50_95']:.3f} |" in README
    assert f"| Men | | | {pc['Male']['map50']:.3f} | {pc['Male']['map50_95']:.3f} |" in README
    assert f"found {g['found']} of the {g['people']:,} labelled people and gave {g['right']} of those the right label" in README
    assert f"labelled {g['female_as_male']} women as male and {g['male_as_female']} men as female" in README
    assert f"on the {SITE['clean_photos']} test photos" in README
    assert f"({SITE['female_boxes']} women and {SITE['male_boxes']} men labelled)" in README
    assert f"{SITE['leaked']} of the {SITE['test_photos']} merged test photos" in README


def test_training_findings_match_the_runs():
    runs = SITE["runs"]
    final = next(r for r in runs if r["run"] == SITE["final_run"])
    merged = [r for r in runs if r["dataset"] == final["dataset"] and r["start"] == "continued"]
    continued = [r for r in runs if r["start"] == "continued"]
    stalled = [r for r in continued if r["best_epoch"] <= 2 and r["epochs_run"] >= 100]
    assert f"There were {len(runs)} training runs between {runs[0]['date']} and {runs[-1]['date']}" in README
    assert f"peaked at {max(r['val_map50'] for r in merged if r['date'] < final['date']):.3f}" in README
    assert f"reached {final['val_map50']:.3f} and became the final model" in README
    assert " and ".join(f"{r['val_map50']:.3f}" for r in merged if r["date"] > final["date"]) in README
    assert f"{len(stalled)} of the {len(continued)} runs that started from an earlier model" in README
