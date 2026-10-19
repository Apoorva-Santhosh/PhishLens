"""Keep dataset remediation reproducible without leaking sanity domains into training."""

from pathlib import Path

import pandas as pd
import pytest

from phishlens.models.prepare_url import prepare


def test_preparation_audits_source_and_excludes_sanity_domains(tmp_path: Path) -> None:
    """Discard invalid/conflicting examples while preserving real paths and source labels."""
    source = tmp_path / "download.csv"
    pd.DataFrame(
        [
            {"URL": "ordinary.test/docs/topic", "Label": "good"},
            {"URL": "ordinary.test/docs/topic", "Label": "good"},
            {"URL": "threat.test/login", "Label": "bad"},
            {"URL": "", "Label": "bad"},
            {"URL": "bad host.test", "Label": "bad"},
            {"URL": "conflict.test/path", "Label": "good"},
            {"URL": "conflict.test/path", "Label": "bad"},
            {"URL": "https://docs.python.org/3/", "Label": "good"},
        ]
    ).to_csv(source, index=False)
    original = source.read_bytes()
    output = tmp_path / "urls.csv"
    report = prepare(source, output)
    frame = pd.read_csv(output)
    assert report["original_rows"] == 8
    assert sum(report["invalid_rows_by_reason"].values()) == 2
    assert report["conflicting_rows_removed"] == 2
    assert report["duplicate_rows_removed"] == 1
    assert report["sanity_domain_rows_excluded"] == 1
    assert report["rows"] == 2
    assert frame.set_index("url").label.to_dict() == {
        "ordinary.test/docs/topic": 0,
        "threat.test/login": 1,
    }
    assert source.read_bytes() == original
    assert (tmp_path / "urls.provenance.json").exists()
    assert report["class_structure"]["0"]["deep_path_fraction"] == 1


def test_unknown_columns_and_labels(tmp_path: Path) -> None:
    """Do not silently infer incompatible source label conventions."""
    source = tmp_path / "download.csv"
    source.write_text("url,label\na.test,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        prepare(source, tmp_path / "urls.csv")
    source.write_text("URL,Label\na.test,unknown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="labels"):
        prepare(source, tmp_path / "urls.csv")
