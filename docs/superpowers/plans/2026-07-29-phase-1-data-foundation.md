# Phase 1: Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble canonical, leak-safe UFC fight tables joined to historical closing odds, and produce the coverage audit that gates the rest of the project.

**Architecture:** Four published datasets are downloaded once and cached on disk, parsed into four canonical parquet tables (`fighters`, `fights`, `fight_stats`, `odds`), and joined on an order-independent `(date, {fighter names})` key. Every module separates pure transformation from IO so the logic is testable without network access. The phase terminates in a coverage report that answers one question: is there enough unbiased odds coverage to make a market-relative backtest trustworthy?

**Tech Stack:** Python 3.12 (conda), pandas, pyarrow, pyreadr, httpx, pytest.

---

## Context for the implementer

You are building the data layer for a model that predicts UFC fight outcomes from stylistic matchups and is scored against the betting market. Read
`docs/superpowers/specs/2026-07-29-ufc-stylistic-prediction-design.md` first.

Two things drive nearly every decision in this plan:

**1. Leakage is the enemy.** Later phases compute each fighter's features *as of* a fight date, using only bouts strictly before it. That is only possible if this phase preserves **event-level granularity** — one row per fighter per round, each stamped with its fight date. Any column that is a career total "as of today" is poison, because it silently encodes the future. Several of our source files contain exactly such columns; this plan quarantines them explicitly.

**2. Direct scraping is not available.** ufcstats.com serves a JavaScript proof-of-work bot check on every URL. We do not defeat it. All data comes from published, redistributable datasets instead.

### The four sources

| Key | File | Rows | What we take | What we ignore |
|---|---|---|---|---|
| `round_stats` | `ufc_stats.rda` from [mtoto/ufc.stats](https://github.com/mtoto/ufc.stats), MIT | 27,668 fighter-rounds | Everything. This is the spine. | — |
| `fight_results` | `ufc_fights.csv` (tidytuesday 2026-07-07) | 8,736 fights | `fight_url` (canonical UFCStats hash), method, time format, referee | — |
| `odds` | `ultimate_ufc_dataset.csv` (tidytuesday 2026-07-07) | 7,177 fights | `r_fighter`, `b_fighter`, `date`, `r_odds`, `b_odds` **only** | ~90 pre-computed `*_avg_*` aggregate columns — undocumented as-of semantics, treat as leaked |
| `athletes` | `ufc_athletes.csv` (tidytuesday 2026-07-07) | fighter profiles | Static physicals only: height, reach, stance, date of birth | Career rate columns (`sig_strikes_landed`, `takedown_avg`, …) — current-state, leaked |

Static physicals are safe from any date's perspective — a fighter's reach does not change. Career *rates* in the same file are not, because they reflect today's totals. That distinction is the entire reason the `athletes` source is only partially used.

`round_stats` columns confirmed present (verified by reading the `.rda` directly): `fighter`, `knockdowns`, `significant_strikes_landed`, `significant_strikes_attempted`, `significant_strikes_rate`, `total_strikes_landed`, `total_strikes_attempted`, `takedown_successful`, `takedown_attempted`, `takedown_rate`, `submission_attempt`, `reversals`, `head_landed`, `head_attempted`, `body_landed`, `body_attempted`, `leg_landed`, `leg_attempted`, `distance_landed`, `distance_attempted`, `clinch_landed`, `clinch_attempted`, `ground_landed`, `ground_attempted`, `round`, `result`, `last_round`, `time`, `scheduled_rounds`, `winner`, `weight_class`, `event`, `fight_date`, `location`, `attendance`.

**Known gap:** there is no control-time column. Grappling is represented by takedowns, ground strikes, submission attempts and reversals instead. Do not invent a substitute.

### Definition of done

`data/processed/` contains four parquet tables, the full test suite passes, and `notebooks/01_coverage_audit.ipynb` renders a coverage report. The phase gate is a human decision made on that report — see Task 9.

---

## File structure

```
src/mma/
  config.py               paths, single source of truth for directories
  data/
    sources.py            the source registry (URLs, filenames, licenses)
    download.py           cached fetching
  parse/
    round_stats.py        .rda  → tidy round-level frame
    fight_results.py      ufc_fights.csv → fight results frame
    odds.py               ultimate_ufc_dataset.csv → odds frame (5 columns only)
    athletes.py           ufc_athletes.csv → static physicals only
  canonical/
    names.py              name normalization
    keys.py               order-independent fight keys
    build.py              assemble the four canonical tables
  audit/
    coverage.py           coverage statistics and bias checks
tests/
  test_names.py  test_keys.py  test_parse_round_stats.py
  test_parse_odds.py  test_download.py  test_build.py  test_coverage.py
notebooks/
  01_coverage_audit.ipynb
docs/
  DATA_SOURCES.md         provenance and licensing
```

Each parse module handles exactly one file and knows nothing about the others. `canonical/build.py` is the only module that sees more than one source at a time.

---

## Task 1: Environment and package skeleton

**Files:**
- Create: `environment.yml`, `pyproject.toml`, `src/mma/__init__.py`, `src/mma/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write `environment.yml`**

```yaml
name: mma
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pandas
  - pyarrow
  - httpx
  - matplotlib
  - jupyterlab
  - pytest
  - pip
  - pip:
      - pyreadr
```

- [ ] **Step 2: Create the environment**

```bash
conda env create -f environment.yml
```

Expected: environment `mma` solves and installs. This takes a few minutes.

- [ ] **Step 3: Verify pyreadr imports**

```bash
conda run -n mma python -c "import pyreadr, pandas, pyarrow, httpx; print('ok')"
```

Expected: `ok`

`pyreadr` reads R `.rda` files and is the one dependency with meaningful install risk (it ships compiled wheels). Verifying it now avoids discovering a problem in Task 3. If it fails to install, stop and report rather than working around it.

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[project]
name = "mma"
version = "0.1.0"
requires-python = ">=3.12"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: requires downloaded source data (deselect with '-m \"not integration\"')",
]
```

- [ ] **Step 5: Write the failing test**

Create `tests/test_config.py`:

```python
from mma import config


def test_data_dirs_are_under_repo_root():
    assert config.RAW_DIR.parent == config.DATA_DIR
    assert config.PROCESSED_DIR.parent == config.DATA_DIR
    assert config.DATA_DIR.name == "data"
    assert config.DATA_DIR.parent == config.REPO_ROOT


def test_ensure_dirs_creates_them(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    config.ensure_dirs()
    assert (tmp_path / "raw").is_dir()
    assert (tmp_path / "processed").is_dir()
```

- [ ] **Step 6: Run the test to verify it fails**

```bash
conda run -n mma python -m pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mma'`

- [ ] **Step 7: Write `src/mma/__init__.py` and `src/mma/config.py`**

`src/mma/__init__.py` is empty.

`src/mma/config.py`:

```python
"""Filesystem layout. Every module resolves paths through here."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_dirs() -> None:
    """Create the data directories if they do not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 8: Install the package in editable mode**

```bash
conda run -n mma python -m pip install -e .
```

- [ ] **Step 9: Run the test to verify it passes**

```bash
conda run -n mma python -m pytest tests/test_config.py -v
```

Expected: 2 passed

- [ ] **Step 10: Commit**

```bash
git add environment.yml pyproject.toml src/mma tests/test_config.py
git commit -m "feat: package skeleton and path config"
```

---

## Task 2: Source registry and cached download

**Files:**
- Create: `src/mma/data/__init__.py`, `src/mma/data/sources.py`, `src/mma/data/download.py`, `tests/test_download.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_download.py`:

```python
import pytest

from mma.data import download
from mma.data.sources import SOURCES


def test_every_source_has_url_filename_and_license():
    assert set(SOURCES) == {"round_stats", "fight_results", "odds", "athletes"}
    for key, source in SOURCES.items():
        assert source.url.startswith("https://"), key
        assert source.filename, key
        assert source.license, key


def test_fetch_returns_cached_file_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(download.config, "RAW_DIR", tmp_path)
    cached = tmp_path / SOURCES["odds"].filename
    cached.write_bytes(b"already here")

    def explode(*args, **kwargs):
        raise AssertionError("network was used despite a cached file")

    monkeypatch.setattr(download.httpx, "stream", explode)

    assert download.fetch("odds") == cached


def test_fetch_rejects_unknown_source():
    with pytest.raises(KeyError):
        download.fetch("not_a_source")
```

The second test is the important one. Downloads are slow and rate-limited; a caching bug that silently re-downloads on every run will waste hours and is invisible unless asserted. Monkeypatching `httpx.stream` to raise turns "used the network" into a test failure.

- [ ] **Step 2: Run the test to verify it fails**

```bash
conda run -n mma python -m pytest tests/test_download.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mma.data'`

- [ ] **Step 3: Write `src/mma/data/sources.py`**

`src/mma/data/__init__.py` is empty.

```python
"""Registry of published datasets. Adding a source means adding it here only."""

from dataclasses import dataclass

TIDYTUESDAY = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday"
    "/main/data/2026/2026-07-07"
)


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    license: str
    description: str


SOURCES: dict[str, Source] = {
    "round_stats": Source(
        name="round_stats",
        url="https://github.com/mtoto/ufc.stats/raw/master/data/ufc_stats.rda",
        filename="ufc_stats.rda",
        license="MIT (mtoto/ufc.stats)",
        description=(
            "Fighter-by-round stats including distance/clinch/ground and "
            "head/body/leg strike splits. The spine of the project."
        ),
    ),
    "fight_results": Source(
        name="fight_results",
        url=f"{TIDYTUESDAY}/ufc_fights.csv",
        filename="ufc_fights.csv",
        license="TidyTuesday, educational use; upstream UFCStats",
        description="One row per fight: canonical fight_url, method, time format.",
    ),
    "odds": Source(
        name="odds",
        url=f"{TIDYTUESDAY}/ultimate_ufc_dataset.csv",
        filename="ultimate_ufc_dataset.csv",
        license="TidyTuesday, educational use; upstream Kaggle",
        description=(
            "Closing moneylines. ONLY r_fighter/b_fighter/date/r_odds/b_odds "
            "are used; its pre-computed averages are leakage risks."
        ),
    ),
    "athletes": Source(
        name="athletes",
        url=f"{TIDYTUESDAY}/ufc_athletes.csv",
        filename="ufc_athletes.csv",
        license="TidyTuesday, educational use",
        description=(
            "Static physicals only: height, reach, stance, date of birth. "
            "Career rate columns in this file are current-state and leak."
        ),
    ),
}
```

- [ ] **Step 4: Write `src/mma/data/download.py`**

```python
"""Cached downloads. Fetch once, keep the bytes, never re-download by accident."""

from pathlib import Path

import httpx

from .. import config
from .sources import SOURCES


def fetch(name: str, *, force: bool = False) -> Path:
    """Download a registered source if not already cached; return its path.

    Raises KeyError for an unknown source name.
    """
    source = SOURCES[name]
    destination = config.RAW_DIR / source.filename

    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream(
        "GET", source.url, follow_redirects=True, timeout=120.0
    ) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    tmp.replace(destination)
    return destination


def fetch_all(*, force: bool = False) -> dict[str, Path]:
    """Fetch every registered source."""
    return {name: fetch(name, force=force) for name in SOURCES}
```

Writing to a `.part` file and renaming on completion means an interrupted download can never leave a truncated file that looks cached.

- [ ] **Step 5: Run the test to verify it passes**

```bash
conda run -n mma python -m pytest tests/test_download.py -v
```

Expected: 3 passed

- [ ] **Step 6: Download the real data**

```bash
conda run -n mma python -c "from mma.data.download import fetch_all; print(fetch_all())"
```

Expected: four paths under `data/raw/`. Verify sizes are non-trivial:

```bash
ls -la data/raw/
```

Expected: `ufc_stats.rda` ≈ 940 KB, `ultimate_ufc_dataset.csv` ≈ 3.3 MB, `ufc_fights.csv` ≈ 2.3 MB, `ufc_athletes.csv` present.

- [ ] **Step 7: Commit**

```bash
git add src/mma/data tests/test_download.py
git commit -m "feat: source registry and cached downloader"
```

`data/` is gitignored — the downloaded files must not appear in `git status`. If they do, stop and fix `.gitignore`.

---

## Task 3: Parse round-level stats

**Files:**
- Create: `src/mma/parse/__init__.py`, `src/mma/parse/round_stats.py`, `tests/test_parse_round_stats.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_parse_round_stats.py`:

```python
import pandas as pd
import pytest

from mma.parse.round_stats import (
    REQUIRED_ROUND_COLUMNS,
    load_round_stats,
    validate_round_stats,
)


def _minimal_frame() -> pd.DataFrame:
    return pd.DataFrame({column: [1] for column in REQUIRED_ROUND_COLUMNS})


def test_validate_accepts_a_complete_frame():
    validate_round_stats(_minimal_frame())


def test_validate_names_the_missing_columns():
    frame = _minimal_frame().drop(columns=["distance_landed", "clinch_landed"])
    with pytest.raises(ValueError) as excinfo:
        validate_round_stats(frame)
    message = str(excinfo.value)
    assert "clinch_landed" in message
    assert "distance_landed" in message


def test_required_columns_include_the_style_splits():
    # These are the columns the stylistic thesis depends on. If an upstream
    # release drops them, the project needs a different data source, so the
    # requirement is asserted rather than assumed.
    for column in [
        "distance_landed", "distance_attempted",
        "clinch_landed", "clinch_attempted",
        "ground_landed", "ground_attempted",
        "head_landed", "body_landed", "leg_landed",
    ]:
        assert column in REQUIRED_ROUND_COLUMNS


@pytest.mark.integration
def test_load_real_file_has_expected_shape():
    frame = load_round_stats()
    assert len(frame) > 27_000
    assert frame["fight_date"].dtype.kind == "M"
    assert frame["round"].min() == 1
```

Validation is a pure function taking a DataFrame, so the fast tests need no file at all. Only the integration test touches real data. Keep that separation for every parser in this plan.

- [ ] **Step 2: Run the test to verify it fails**

```bash
conda run -n mma python -m pytest tests/test_parse_round_stats.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mma.parse'`

- [ ] **Step 3: Write `src/mma/parse/round_stats.py`**

`src/mma/parse/__init__.py` is empty.

```python
"""Read the round-level stats .rda into a tidy DataFrame."""

from pathlib import Path

import pandas as pd
import pyreadr

from ..data.download import fetch

REQUIRED_ROUND_COLUMNS = frozenset({
    "fighter", "knockdowns",
    "significant_strikes_landed", "significant_strikes_attempted",
    "total_strikes_landed", "total_strikes_attempted",
    "takedown_successful", "takedown_attempted",
    "submission_attempt", "reversals",
    "head_landed", "head_attempted",
    "body_landed", "body_attempted",
    "leg_landed", "leg_attempted",
    "distance_landed", "distance_attempted",
    "clinch_landed", "clinch_attempted",
    "ground_landed", "ground_attempted",
    "round", "result", "last_round", "scheduled_rounds",
    "winner", "weight_class", "event", "fight_date",
})


def validate_round_stats(frame: pd.DataFrame) -> None:
    """Raise ValueError if any required column is absent."""
    missing = REQUIRED_ROUND_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"round stats missing columns: {sorted(missing)}")


def load_round_stats(path: Path | None = None) -> pd.DataFrame:
    """Load the round-level stats, validated and with parsed dates."""
    path = path or fetch("round_stats")
    frame = pyreadr.read_r(str(path))["ufc_stats"]
    validate_round_stats(frame)
    frame = frame.copy()
    frame["fight_date"] = pd.to_datetime(frame["fight_date"])
    frame["round"] = frame["round"].astype(int)
    return frame
```

- [ ] **Step 4: Run the fast tests to verify they pass**

```bash
conda run -n mma python -m pytest tests/test_parse_round_stats.py -v -m "not integration"
```

Expected: 3 passed, 1 deselected

- [ ] **Step 5: Run the integration test**

```bash
conda run -n mma python -m pytest tests/test_parse_round_stats.py -v -m integration
```

Expected: 1 passed. If `pyreadr` reports a different object name than `ufc_stats`, print `pyreadr.read_r(path).keys()` and use the actual key — do not guess.

- [ ] **Step 6: Commit**

```bash
git add src/mma/parse tests/test_parse_round_stats.py
git commit -m "feat: parse round-level fight stats from .rda"
```

---

## Task 4: Name normalization

**Files:**
- Create: `src/mma/canonical/__init__.py`, `src/mma/canonical/names.py`, `tests/test_names.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_names.py`:

```python
import pytest

from mma.canonical.names import normalize_name


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("José Aldo", "jose aldo"),
        ("JOSE ALDO", "jose aldo"),
        ("  Jose   Aldo  ", "jose aldo"),
        ("Antônio Rodrigo Nogueira", "antonio rodrigo nogueira"),
        ("Marcos Rogério de Lima", "marcos rogerio de lima"),
        ("Ronaldo 'Jacaré' Souza", "ronaldo jacare souza"),
        ("Rafael Dos Anjos", "rafael dos anjos"),
        ("Alex Perez", "alex perez"),
    ],
)
def test_normalizes_case_accents_punctuation_and_spacing(raw, expected):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Antonio Carlos Junior", "antonio carlos junior"),
        ("Junior dos Santos", "junior dos santos"),
    ],
)
def test_does_not_strip_name_parts_that_look_like_suffixes(raw, expected):
    # Both fighters really are listed this way. A generic "strip trailing
    # Jr/Junior" rule would corrupt them into different people, so this
    # normalizer deliberately does no suffix handling. Residual mismatches
    # are resolved by the alias table instead.
    assert normalize_name(raw) == expected


def test_rejects_missing_names():
    with pytest.raises(ValueError):
        normalize_name(None)
    with pytest.raises(ValueError):
        normalize_name("   ")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
conda run -n mma python -m pytest tests/test_names.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mma.canonical'`

- [ ] **Step 3: Write `src/mma/canonical/names.py`**

`src/mma/canonical/__init__.py` is empty.

```python
"""Fighter name normalization.

Deliberately conservative: case, accents, punctuation and whitespace only.
No suffix stripping — "Antonio Carlos Junior" and "Junior dos Santos" are real
fighter names in which the suffix-looking token is part of the name. All four
sources derive from UFCStats, so spellings agree closely; the few that do not
are handled by an explicit alias table rather than by clever rules.
"""

import re
import unicodedata

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9\s]")

# Populated during Task 7 from the unmatched-name report. Keys and values are
# both already-normalized names.
ALIASES: dict[str, str] = {}


def normalize_name(name: str | None) -> str:
    """Return a comparable form of a fighter name.

    Raises ValueError if the name is missing or blank.
    """
    if name is None or not str(name).strip():
        raise ValueError(f"fighter name must be a non-empty string, got {name!r}")

    decomposed = unicodedata.normalize("NFKD", str(name))
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    cleaned = _NON_ALPHANUMERIC.sub(" ", without_accents.lower())
    collapsed = " ".join(cleaned.split())
    return ALIASES.get(collapsed, collapsed)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
conda run -n mma python -m pytest tests/test_names.py -v
```

Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/mma/canonical tests/test_names.py
git commit -m "feat: conservative fighter name normalization"
```

---

## Task 5: Order-independent fight keys

**Files:**
- Create: `src/mma/canonical/keys.py`, `tests/test_keys.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_keys.py`:

```python
import pandas as pd
import pytest

from mma.canonical.keys import fight_key


def test_key_is_independent_of_fighter_order():
    # Sources disagree about which fighter is listed first: the odds file uses
    # red/blue corners, the stats file uses its own ordering. A key that depends
    # on order would fail to match roughly half of all fights.
    left = fight_key("2021-10-30", "Israel Adesanya", "Robert Whittaker")
    right = fight_key("2021-10-30", "Robert Whittaker", "Israel Adesanya")
    assert left == right


def test_key_normalizes_names_and_dates():
    assert fight_key("2021-10-30", "José Aldo", "Rob Font") == fight_key(
        pd.Timestamp("2021-10-30 00:00:00"), "jose aldo", "ROB FONT"
    )


def test_key_distinguishes_different_dates():
    assert fight_key("2021-10-30", "A Fighter", "B Fighter") != fight_key(
        "2021-10-31", "A Fighter", "B Fighter"
    )


def test_key_rejects_a_fighter_facing_themselves():
    with pytest.raises(ValueError):
        fight_key("2021-10-30", "José Aldo", "Jose Aldo")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
conda run -n mma python -m pytest tests/test_keys.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mma.canonical.keys'`

- [ ] **Step 3: Write `src/mma/canonical/keys.py`**

```python
"""Join keys shared across sources."""

import pandas as pd

from .names import normalize_name


def fight_key(date, fighter_a: str, fighter_b: str) -> str:
    """Return a stable, order-independent key for one fight.

    Format: "YYYY-MM-DD|<first name>|<second name>", the two normalized names
    sorted alphabetically so corner assignment cannot affect the result.

    Raises ValueError if both names normalize to the same value.
    """
    first, second = sorted([normalize_name(fighter_a), normalize_name(fighter_b)])
    if first == second:
        raise ValueError(f"a fighter cannot face themselves: {fighter_a!r}")
    day = pd.Timestamp(date).strftime("%Y-%m-%d")
    return f"{day}|{first}|{second}"


def add_fight_key(
    frame: pd.DataFrame, date_column: str, name_a_column: str, name_b_column: str
) -> pd.DataFrame:
    """Return a copy of `frame` with a `fight_key` column added."""
    result = frame.copy()
    result["fight_key"] = [
        fight_key(date, name_a, name_b)
        for date, name_a, name_b in zip(
            result[date_column], result[name_a_column], result[name_b_column]
        )
    ]
    return result
```

A sorted string rather than a `frozenset` because it must survive a round trip through parquet, which has no set type.

- [ ] **Step 4: Run the test to verify it passes**

```bash
conda run -n mma python -m pytest tests/test_keys.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/mma/canonical/keys.py tests/test_keys.py
git commit -m "feat: order-independent fight join key"
```

---

## Task 6: Parse odds and physicals

**Files:**
- Create: `src/mma/parse/odds.py`, `src/mma/parse/fight_results.py`, `src/mma/parse/athletes.py`, `tests/test_parse_odds.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_parse_odds.py`:

```python
import pandas as pd
import pytest

from mma.parse.odds import ODDS_COLUMNS, tidy_odds


def _raw_odds_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "r_fighter": ["Israel Adesanya"],
        "b_fighter": ["Robert Whittaker"],
        "date": ["2021-10-30"],
        "r_odds": [-260],
        "b_odds": [210],
        # A leaked aggregate that must not survive parsing:
        "b_avg_sig_str_landed": [4.2],
    })


def test_keeps_only_identity_and_odds_columns():
    result = tidy_odds(_raw_odds_frame())
    assert set(result.columns) == set(ODDS_COLUMNS) | {"fight_key"}
    assert "b_avg_sig_str_landed" not in result.columns


def test_adds_an_order_independent_fight_key():
    result = tidy_odds(_raw_odds_frame())
    assert result.loc[0, "fight_key"] == (
        "2021-10-30|israel adesanya|robert whittaker"
    )


def test_drops_rows_with_missing_odds():
    frame = _raw_odds_frame()
    frame.loc[0, "r_odds"] = None
    assert len(tidy_odds(frame)) == 0


def test_rejects_a_frame_missing_required_columns():
    with pytest.raises(ValueError):
        tidy_odds(_raw_odds_frame().drop(columns=["r_odds"]))
```

The first test is a guard rail, not a formality. The odds file's ~90 aggregate columns are the single most likely way leakage enters this project — a later phase joining on `fight_key` would happily pick them up. Parsing drops them at the boundary so they are never available to be misused.

- [ ] **Step 2: Run the test to verify it fails**

```bash
conda run -n mma python -m pytest tests/test_parse_odds.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mma.parse.odds'`

- [ ] **Step 3: Write `src/mma/parse/odds.py`**

```python
"""Closing moneylines. Five columns in, five columns out, nothing else."""

from pathlib import Path

import pandas as pd

from ..canonical.keys import add_fight_key
from ..data.download import fetch

ODDS_COLUMNS = ["date", "r_fighter", "b_fighter", "r_odds", "b_odds"]


def tidy_odds(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce the raw odds file to identity plus closing lines.

    Every other column is discarded: the source ships pre-computed career
    aggregates with undocumented as-of semantics, which would leak.
    """
    missing = set(ODDS_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"odds source missing columns: {sorted(missing)}")

    result = frame.loc[:, ODDS_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.dropna(subset=["r_odds", "b_odds"])
    result = add_fight_key(result, "date", "r_fighter", "b_fighter")
    return result.reset_index(drop=True)


def load_odds(path: Path | None = None) -> pd.DataFrame:
    """Load and tidy the closing-odds source."""
    path = path or fetch("odds")
    return tidy_odds(pd.read_csv(path))
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
conda run -n mma python -m pytest tests/test_parse_odds.py -v
```

Expected: 4 passed

- [ ] **Step 5: Write `src/mma/parse/fight_results.py`**

```python
"""One row per fight: canonical UFCStats id, method, format, officials."""

from pathlib import Path

import pandas as pd

from ..canonical.keys import add_fight_key
from ..data.download import fetch

RESULT_COLUMNS = [
    "fight_url", "event_name", "date", "location",
    "f1_name", "f2_name", "weight_class",
    "method", "round", "time", "time_format", "referee",
]


def tidy_fight_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce the fight-results file and attach ids and keys."""
    missing = set(RESULT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"fight results missing columns: {sorted(missing)}")

    result = frame.loc[:, RESULT_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"])
    # The trailing path segment of fight_url is the canonical UFCStats id.
    result["fight_id"] = result["fight_url"].str.rstrip("/").str.split("/").str[-1]
    result = add_fight_key(result, "date", "f1_name", "f2_name")
    return result.reset_index(drop=True)


def load_fight_results(path: Path | None = None) -> pd.DataFrame:
    """Load and tidy the fight-results source."""
    path = path or fetch("fight_results")
    return tidy_fight_results(pd.read_csv(path))
```

- [ ] **Step 6: Write `src/mma/parse/athletes.py`**

```python
"""Static physicals only.

Height, reach, stance and date of birth do not change, so they are safe to use
for a fight at any date. The career rate columns in the same file (strikes
landed per minute, takedown averages, and similar) describe the fighter as of
today and would leak future information into past fights. They are dropped here
so no later phase can reach them.
"""

from pathlib import Path

import pandas as pd

from ..canonical.names import normalize_name
from ..data.download import fetch

PHYSICAL_COLUMNS = ["name", "height", "weight", "reach", "stance", "dob"]


def tidy_athletes(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce an athlete profile file to static physical attributes."""
    available = [column for column in PHYSICAL_COLUMNS if column in frame.columns]
    if "name" not in available:
        raise ValueError("athlete source has no name column")

    result = frame.loc[:, available].copy()
    result["fighter_key"] = result["name"].map(normalize_name)
    if "dob" in result.columns:
        result["dob"] = pd.to_datetime(result["dob"], errors="coerce")
    return result.drop_duplicates(subset=["fighter_key"]).reset_index(drop=True)


def load_athletes(path: Path | None = None) -> pd.DataFrame:
    """Load and tidy static physical attributes."""
    path = path or fetch("athletes")
    return tidy_athletes(pd.read_csv(path))
```

`ufc_athletes.csv` and `ufcstats_data.csv` name their physical columns slightly differently, which is why the column selection is tolerant here while the odds parser is strict. Confirm which file actually carries `reach` and `stance` when you run it; if `ufc_athletes.csv` lacks them, switch the `athletes` source URL to `{TIDYTUESDAY}/ufcstats_data.csv` and rerun.

- [ ] **Step 7: Run the whole suite**

```bash
conda run -n mma python -m pytest -v -m "not integration"
```

Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add src/mma/parse tests/test_parse_odds.py
git commit -m "feat: parse odds, fight results and static physicals"
```

---

## Task 7: Build canonical tables

**Files:**
- Create: `src/mma/canonical/build.py`, `tests/test_build.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_build.py`:

```python
import pandas as pd

from mma.canonical.build import build_fight_stats, build_fights


def _round_stats() -> pd.DataFrame:
    """Two fighters, two rounds each, one fight."""
    rows = []
    for fighter in ["Israel Adesanya", "Robert Whittaker"]:
        for round_number in [1, 2]:
            rows.append({
                "fighter": fighter,
                "fight_date": pd.Timestamp("2021-10-30"),
                "event": "UFC 271",
                "round": round_number,
                "significant_strikes_landed": 10,
                "distance_landed": 8,
                "clinch_landed": 1,
                "ground_landed": 1,
                "winner": "Israel Adesanya",
                "weight_class": "Middleweight",
                "scheduled_rounds": 5,
            })
    return pd.DataFrame(rows)


def test_build_fights_produces_one_row_per_fight():
    fights = build_fights(_round_stats())
    assert len(fights) == 1
    row = fights.iloc[0]
    assert row["fight_key"] == "2021-10-30|israel adesanya|robert whittaker"
    assert row["winner_key"] == "israel adesanya"


def test_build_fights_ignores_round_count():
    # Four input rows (2 fighters x 2 rounds) still describe one fight.
    assert len(build_fights(_round_stats())) == 1


def test_build_fight_stats_keeps_round_granularity():
    stats = build_fight_stats(_round_stats())
    assert len(stats) == 4
    assert set(stats["round"]) == {1, 2}
    assert "fight_key" in stats.columns
    assert "fighter_key" in stats.columns


def test_build_fights_drops_fights_without_exactly_two_fighters():
    corrupt = _round_stats()
    corrupt.loc[len(corrupt)] = corrupt.iloc[0].copy()
    corrupt.loc[len(corrupt) - 1, "fighter"] = "Third Person"
    assert len(build_fights(corrupt)) == 0
```

The last test matters: the round-level source is keyed by name and date, not by a fight id, so a data error or an unlucky name collision could group three fighters into one "fight." Dropping those rather than silently taking the first two keeps corrupt rows out of the model.

- [ ] **Step 2: Run the test to verify it fails**

```bash
conda run -n mma python -m pytest tests/test_build.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mma.canonical.build'`

- [ ] **Step 3: Write `src/mma/canonical/build.py`**

```python
"""Assemble canonical tables from the parsed sources."""

import pandas as pd

from .. import config
from ..parse.athletes import load_athletes
from ..parse.fight_results import load_fight_results
from ..parse.odds import load_odds
from ..parse.round_stats import load_round_stats
from .keys import fight_key
from .names import normalize_name


def _with_keys(round_stats: pd.DataFrame) -> pd.DataFrame:
    result = round_stats.copy()
    result["fighter_key"] = result["fighter"].map(normalize_name)
    return result


def build_fights(round_stats: pd.DataFrame) -> pd.DataFrame:
    """One row per fight, derived from the round-level source.

    Groups of other than exactly two distinct fighters are dropped.
    """
    keyed = _with_keys(round_stats)
    fights = []

    for (date, event), group in keyed.groupby(["fight_date", "event"], sort=True):
        for _, bout in group.groupby(_pair_id(group)):
            fighters = sorted(bout["fighter_key"].unique())
            if len(fighters) != 2:
                continue
            fights.append({
                "fight_key": fight_key(date, fighters[0], fighters[1]),
                "date": date,
                "event": event,
                "fighter_a_key": fighters[0],
                "fighter_b_key": fighters[1],
                "winner_key": normalize_name(bout["winner"].iloc[0])
                if pd.notna(bout["winner"].iloc[0])
                else None,
                "weight_class": bout["weight_class"].iloc[0],
                "scheduled_rounds": bout["scheduled_rounds"].iloc[0],
            })

    return pd.DataFrame(fights)


def _pair_id(group: pd.DataFrame) -> pd.Series:
    """Assign each row to a bout within an event.

    The round-level source lists both fighters of a bout consecutively; pairing
    is therefore by (winner, weight_class, scheduled_rounds) within the event,
    which is unique per bout on a card.
    """
    return (
        group["winner"].astype(str)
        + "|"
        + group["weight_class"].astype(str)
        + "|"
        + group["scheduled_rounds"].astype(str)
    )


def build_fight_stats(round_stats: pd.DataFrame) -> pd.DataFrame:
    """Round-level stats with join keys attached. Granularity is preserved."""
    keyed = _with_keys(round_stats)
    fights = build_fights(round_stats)

    fighter_to_fight = {}
    for _, row in fights.iterrows():
        for fighter in [row["fighter_a_key"], row["fighter_b_key"]]:
            fighter_to_fight[(row["date"], fighter)] = row["fight_key"]

    keyed["fight_key"] = [
        fighter_to_fight.get((date, fighter))
        for date, fighter in zip(keyed["fight_date"], keyed["fighter_key"])
    ]
    return keyed.dropna(subset=["fight_key"]).reset_index(drop=True)


def build_all() -> dict[str, pd.DataFrame]:
    """Build every canonical table and write it to data/processed/."""
    config.ensure_dirs()
    round_stats = load_round_stats()

    tables = {
        "fights": build_fights(round_stats),
        "fight_stats": build_fight_stats(round_stats),
        "fighters": load_athletes(),
        "odds": load_odds(),
        "fight_results": load_fight_results(),
    }

    for name, table in tables.items():
        table.to_parquet(config.PROCESSED_DIR / f"{name}.parquet", index=False)
    return tables
```

Note the `_pair_id` heuristic. Because the round-level source has no fight id, bouts on a card are identified by (winner, weight class, scheduled rounds). This is *almost* always unique. Step 5 measures how often it is not.

- [ ] **Step 4: Run the test to verify it passes**

```bash
conda run -n mma python -m pytest tests/test_build.py -v
```

Expected: 4 passed

- [ ] **Step 5: Build the real tables and measure pairing loss**

```bash
conda run -n mma python -c "
from mma.canonical.build import build_all
from mma.parse.round_stats import load_round_stats
raw = load_round_stats()
tables = build_all()
events = raw.groupby(['fight_date','event']).size().shape[0]
print('events:', events)
print('fights built:', len(tables['fights']))
print('fight_stats rows:', len(tables['fight_stats']), 'of', len(raw))
print('dropped rounds:', len(raw) - len(tables['fight_stats']))
"
```

Expected: roughly 8,000+ fights and under 5% dropped rounds. **If more than 5% of rounds are dropped, stop.** The `_pair_id` heuristic is failing and needs replacing — most likely by joining to `fight_results` on `fight_key` to recover the canonical `fight_id` instead of inferring pairs. Report the number rather than proceeding with a lossy table.

- [ ] **Step 6: Commit**

```bash
git add src/mma/canonical/build.py tests/test_build.py
git commit -m "feat: build canonical fights and fight_stats tables"
```

---

## Task 8: Coverage audit

**Files:**
- Create: `src/mma/audit/__init__.py`, `src/mma/audit/coverage.py`, `tests/test_coverage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_coverage.py`:

```python
import pandas as pd

from mma.audit.coverage import coverage_by_year, coverage_summary, unmatched_fights


def _fights() -> pd.DataFrame:
    return pd.DataFrame({
        "fight_key": ["k1", "k2", "k3", "k4"],
        "date": pd.to_datetime(
            ["2015-01-01", "2015-06-01", "2020-01-01", "2020-06-01"]
        ),
        "weight_class": ["Lightweight"] * 4,
    })


def _odds() -> pd.DataFrame:
    return pd.DataFrame({"fight_key": ["k1", "k3"], "r_odds": [-150, 120]})


def test_summary_reports_overall_match_rate():
    summary = coverage_summary(_fights(), _odds())
    assert summary["total_fights"] == 4
    assert summary["matched"] == 2
    assert summary["match_rate"] == 0.5


def test_coverage_by_year_splits_correctly():
    by_year = coverage_by_year(_fights(), _odds()).set_index("year")
    assert by_year.loc[2015, "matched"] == 1
    assert by_year.loc[2015, "total"] == 2
    assert by_year.loc[2020, "match_rate"] == 0.5


def test_unmatched_returns_the_missing_fights():
    missing = unmatched_fights(_fights(), _odds())
    assert set(missing["fight_key"]) == {"k2", "k4"}


def test_summary_respects_a_date_floor():
    summary = coverage_summary(_fights(), _odds(), since="2020-01-01")
    assert summary["total_fights"] == 2
    assert summary["matched"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
conda run -n mma python -m pytest tests/test_coverage.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mma.audit'`

- [ ] **Step 3: Write `src/mma/audit/coverage.py`**

`src/mma/audit/__init__.py` is empty.

```python
"""How much of the fight history has closing odds, and is what's missing biased?"""

import pandas as pd


def _restrict(fights: pd.DataFrame, since: str | None) -> pd.DataFrame:
    if since is None:
        return fights
    return fights[fights["date"] >= pd.Timestamp(since)]


def coverage_summary(
    fights: pd.DataFrame, odds: pd.DataFrame, since: str | None = None
) -> dict:
    """Overall match rate, optionally restricted to fights on or after `since`."""
    scope = _restrict(fights, since)
    matched = scope["fight_key"].isin(set(odds["fight_key"])).sum()
    total = len(scope)
    return {
        "total_fights": int(total),
        "matched": int(matched),
        "match_rate": float(matched / total) if total else 0.0,
    }


def coverage_by_year(
    fights: pd.DataFrame, odds: pd.DataFrame, since: str | None = None
) -> pd.DataFrame:
    """Match rate per calendar year — the primary bias check."""
    scope = _restrict(fights, since).copy()
    scope["year"] = scope["date"].dt.year
    scope["matched"] = scope["fight_key"].isin(set(odds["fight_key"]))
    grouped = (
        scope.groupby("year")["matched"].agg(["sum", "count"]).reset_index()
    )
    grouped.columns = ["year", "matched", "total"]
    grouped["match_rate"] = grouped["matched"] / grouped["total"]
    return grouped


def coverage_by_weight_class(
    fights: pd.DataFrame, odds: pd.DataFrame, since: str | None = None
) -> pd.DataFrame:
    """Match rate per weight class — the secondary bias check."""
    scope = _restrict(fights, since).copy()
    scope["matched"] = scope["fight_key"].isin(set(odds["fight_key"]))
    grouped = (
        scope.groupby("weight_class")["matched"].agg(["sum", "count"]).reset_index()
    )
    grouped.columns = ["weight_class", "matched", "total"]
    grouped["match_rate"] = grouped["matched"] / grouped["total"]
    return grouped.sort_values("match_rate")


def unmatched_fights(
    fights: pd.DataFrame, odds: pd.DataFrame, since: str | None = None
) -> pd.DataFrame:
    """The fights with no closing odds, for eyeballing name-mismatch patterns."""
    scope = _restrict(fights, since)
    return scope[~scope["fight_key"].isin(set(odds["fight_key"]))]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
conda run -n mma python -m pytest tests/test_coverage.py -v
```

Expected: 4 passed

- [ ] **Step 5: Run the full suite**

```bash
conda run -n mma python -m pytest -v
```

Expected: all pass, including integration tests

- [ ] **Step 6: Commit**

```bash
git add src/mma/audit tests/test_coverage.py
git commit -m "feat: odds coverage audit with year and weight-class bias checks"
```

---

## Task 9: The coverage report and the phase gate

**Files:**
- Create: `notebooks/01_coverage_audit.ipynb`, `docs/DATA_SOURCES.md`

- [ ] **Step 1: Produce the numbers**

```bash
conda run -n mma python -c "
import pandas as pd
from mma import config
from mma.audit import coverage

fights = pd.read_parquet(config.PROCESSED_DIR / 'fights.parquet')
odds = pd.read_parquet(config.PROCESSED_DIR / 'odds.parquet')

print('ALL TIME  ', coverage.coverage_summary(fights, odds))
print('2010+     ', coverage.coverage_summary(fights, odds, since='2010-01-01'))
print()
print(coverage.coverage_by_year(fights, odds, since='2010-01-01').to_string(index=False))
print()
print(coverage.coverage_by_weight_class(fights, odds, since='2010-01-01').head(10).to_string(index=False))
print()
print('sample unmatched fights:')
print(coverage.unmatched_fights(fights, odds, since='2010-01-01').head(20).to_string(index=False))
"
```

Record the output. It is the input to the gate decision.

- [ ] **Step 2: Repair name mismatches**

Read the sample of unmatched fights. Where the same bout appears in both sources under differently-spelled names, add entries to `ALIASES` in `src/mma/canonical/names.py`, mapping the normalized odds-side spelling to the normalized stats-side spelling:

```python
# Left side: the spelling as it appears in the odds source, normalized.
# Right side: the spelling as it appears in the round-stats source, normalized.
# These entries are examples of the *shape* — replace them with the real
# mismatches your unmatched report turns up.
ALIASES: dict[str, str] = {
    "jacare souza": "ronaldo souza",
    "shogun rua": "mauricio rua",
}
```

Only add a pair once you have confirmed both spellings refer to the same fighter on the same card. An incorrect alias silently merges two people's fight histories, which is far more damaging than an unmatched row.

Re-run Step 1 after each batch. Stop when additional aliases stop moving the match rate materially — chasing the last 1% by hand is not worth it.

- [ ] **Step 3: Write the notebook**

Create `notebooks/01_coverage_audit.ipynb`. The notebook imports from `mma` and defines no analysis logic of its own — every computation already exists in `mma.audit.coverage`.

Cell 1 (code):

```python
import pandas as pd
import matplotlib.pyplot as plt

from mma import config
from mma.audit import coverage

fights = pd.read_parquet(config.PROCESSED_DIR / "fights.parquet")
odds = pd.read_parquet(config.PROCESSED_DIR / "odds.parquet")
print("all time:", coverage.coverage_summary(fights, odds))
print("2010+   :", coverage.coverage_summary(fights, odds, since="2010-01-01"))
```

Cell 2 (code) — the primary bias check:

```python
by_year = coverage.coverage_by_year(fights, odds, since="2010-01-01")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(by_year["year"], by_year["match_rate"])
ax.axhline(0.8, linestyle="--", label="0.80 gate")
ax.set_ylabel("odds match rate")
ax.set_ylim(0, 1)
ax.legend()
plt.show()
by_year
```

Cell 3 (code) — the secondary bias check:

```python
coverage.coverage_by_weight_class(fights, odds, since="2010-01-01")
```

Cell 4 (code) — what is missing, for eyeballing name-mismatch patterns:

```python
coverage.unmatched_fights(fights, odds, since="2010-01-01").head(20)
```

Cell 5 (markdown) — state the gate decision and the numbers it rests on:

```markdown
## Gate decision

- 2010+ coverage: __%
- Lowest yearly match rate: __% (____)
- Lowest weight-class match rate: __% (____)

**Decision:** [pass / pass with caveats / stop] — because ____.
```

- [ ] **Step 4: Write `docs/DATA_SOURCES.md`**

Document, for each of the four sources: URL, license, what we use it for, and — for `odds` and `athletes` — which columns are deliberately discarded and why. Note that ufcstats.com was not scraped because it serves a JavaScript bot challenge. This matters for a public repo: readers should be able to see the provenance and reproduce the tables.

- [ ] **Step 5: Evaluate the gate**

Per the spec, Phase 1 passes when:
- **Odds coverage ≥80% for fights from 2010 onward**, and
- **unmatched fights show no systematic bias** — no year below roughly 70%, no weight class dramatically worse than the rest.

Three outcomes:

| Result | Action |
|---|---|
| Both conditions met | Proceed to Phase 2. |
| Coverage below 80% but unbiased | Report and ask. A market-relative backtest on a smaller sample may still be viable, but the confidence intervals widen — that is the user's call, not the implementer's. |
| Coverage biased | Stop. Report which years or weight classes are missing. A backtest on a biased subset produces a confident wrong answer, which is worse than no answer. |

Do not tune the alias table to hit 80%. The number is a measurement, not a target.

- [ ] **Step 6: Commit and push**

```bash
git add notebooks/01_coverage_audit.ipynb docs/DATA_SOURCES.md src/mma/canonical/names.py
git commit -m "docs: coverage audit notebook and data provenance"
git push origin main
```

---

## Phase 1 exit criteria

- [ ] Four parquet tables in `data/processed/`: `fights`, `fight_stats`, `fighters`, `odds`
- [ ] `fight_stats` retains one row per fighter per round, with the distance/clinch/ground and head/body/leg columns intact
- [ ] No career-aggregate column from `ultimate_ufc_dataset.csv` or `ufc_athletes.csv` appears in any processed table
- [ ] Full test suite passes
- [ ] Coverage report produced and the gate decision recorded in the notebook
- [ ] `docs/DATA_SOURCES.md` documents provenance and licensing
