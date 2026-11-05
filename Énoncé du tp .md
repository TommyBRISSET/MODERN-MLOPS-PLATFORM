# SUPINFO - 5LAKE: Build a Production MLOps Pipeline with OSS

**Project:** predict the sale price of a house, and run that model the way real teams do:
validated data, tracked experiments, a model registry with promotion rules, a secured API,
monitoring, alerts and dashboards.

**Tools:** PySpark, Feast, MLflow, Dagster, FastAPI, Postgres, Evidently, Prometheus, Grafana,
Docker Compose and `uv`.


------------------------------------------------------------------------

## How to use this guide

The guide is a sequence of **steps**. Each step:

1. explains **what** you are building and **why** it matters in a real system,
2. gives you the **files** to create, each with a short explanation of its key parts,
3. ends with a **Check** that proves the step works before you move on.

Type the files yourself or copy them, but **read the explanation first**. The goal is not only
to have a working project, but to be able to explain every part of it.

> **Tip:** do not skip the checks. A problem found at step 4 takes minutes to fix; the same
> problem found at step 19 can take hours to trace back.

------------------------------------------------------------------------

## What you will build

### The big picture

```text
                      ┌───────────────── Dagster (webserver + daemon) ─────────────────┐
                      │                                                                │
  OpenML ──► raw CSV ─┼─► PySpark validate + prepare ─► Feast ─► train + gate ─────────┼─► MLflow
                      │                                                                │   registry
                      │   daily: drift check + accuracy check ◄─────────┐              │      │
                      └─────────────────────────────────────────────────┼──────────────┘      │
                                                                        │                     │
                                                       ┌────────────────┴───┐    champion     │
  client ──(X-API-Key)──► FastAPI  /predict /feedback ─┤ Postgres           │◄────────────────┘
                              │                        │ predictions        │   loaded at startup
                              │ /metrics               │ feedback           │
                              ▼                        │ monitoring_results │
                         Prometheus ──► Grafana ◄──────┴────────────────────┘
                        (alert rules)
```

Read it from left to right:

- **Training side (top):** Dagster runs a pipeline that downloads the data, checks it with
  PySpark, builds features with Feast, trains a model and registers it in MLflow.
- **Serving side (bottom):** a FastAPI service loads the approved model ("champion") and
  answers price requests. Every prediction is stored in Postgres.
- **Monitoring (bottom right):** Prometheus collects the API's metrics and fires alerts; Grafana
  shows dashboards; Dagster runs daily checks on the stored predictions.

### The life of a model

```text
 training_job ──► registered version N ──► promotion gate ──┬─ fails ─► stays a plain version (tags explain why)
                                                            └─ passes ─► alias "challenger"
                                                                              │
                                     mlops-model promote (a person decides)  ▼
                                                                         alias "champion" ──► API restart serves it
                                     mlops-model rollback  ──────────────► previous champion
```

This is the most important idea in the project: **a new model never goes live by itself.** It
must pass automatic checks (the *gate*), and then a person promotes it. If something goes wrong,
one command brings the previous model back.

### What "production-ready" means here

| Property | What it means in this project |
|---|---|
| **Traceable** | For every model you can answer: which data, which code, which settings, which results? |
| **Gated** | A model must pass quality and sanity checks before it can be promoted. |
| **Reversible** | One command rolls back to the previous model. |
| **Validated** | Bad data stops the pipeline before it reaches training. |
| **Observable** | Metrics, alerts, structured logs, dashboards, and daily drift and accuracy checks. |
| **Secured** | API keys, passwords outside the code, separate database users, services only on localhost. |
| **Tested** | Unit tests, a smoke test, a load test and automated CI. |
| **Reproducible** | Locked dependencies and pinned images: the same project builds the same way everywhere. |
| **Operable** | A runbook tells anyone how to release, roll back and react to alerts. |

------------------------------------------------------------------------

## Step 1: Install the tools

**Goal:** a working development environment.

Install:

-   **Git**: version control.
-   **Python 3.11**: the project's language.
-   **`uv`**: installs Python packages and locks their versions
    (<https://docs.astral.sh/uv/getting-started/installation/>).
-   **Java 17**: PySpark runs on the Java Virtual Machine.
-   **Docker Desktop** (Windows/macOS) or **Docker Engine with Compose v2** (Linux), which runs every
    service in containers.
-   **Visual Studio Code** with the **Python**, **Pylance** and **Docker** extensions.

**Docker memory:** the whole stack uses about 1.8 GB when idle and about 1–1.5 GB more while a
model trains. In Docker Desktop's settings, give Docker at least 4 GB (6 GB is comfortable).

> **Windows:** the data pipeline runs inside Linux containers, so you do not need any extra
> Hadoop tools. WSL2 also works well.

### Check

``` bash
git --version
python --version        # must show 3.11.x (or try: python3.11 --version)
uv --version
java -version           # must show 17
docker --version
docker compose version
docker run --rm hello-world
```

Every command must print a version (and the last one a "Hello from Docker!" message).

------------------------------------------------------------------------

## Step 2: Create the project

**Goal:** an empty project with the right folder structure and the basic configuration files.

``` bash
mkdir modern-mlops
cd modern-mlops
git init
uv init --python 3.11
```

`uv init` creates `README.md`, `.python-version` and a sample `main.py`. **Delete `main.py`**;
keep the other two.

> **Tip:** create the project in a normal folder such as `Documents` or `projects`, **not**
> inside OneDrive, Dropbox or iCloud. Those tools try to sync the thousands of files in the
> virtual environment and slow everything down.

### The folder structure

``` text
modern-mlops/
├── pyproject.toml            project definition and dependencies
├── uv.lock                   exact versions of every package (generated)
├── .env.example / .env       configuration and secrets (.env is never committed)
├── compose.yaml              the whole stack: which containers run and how they connect
├── docker/                   image definitions and service configuration
├── feature_repo/             Feast configuration
├── monitoring/               Prometheus and Grafana configuration
├── src/mlops/                the Python package: all the project's code
├── scripts/                  helper scripts (smoke test, traffic replay)
├── loadtest/                 load test
├── tests/                    unit tests
├── docs/RUNBOOK.md           operating instructions
├── data/  artifacts/         created at runtime, never committed
└── .github/                  CI/CD workflows
```

Create the folders.

macOS/Linux:

``` bash
mkdir -p docker feature_repo src/mlops scripts loadtest tests docs data artifacts \
  monitoring/grafana/dashboards \
  monitoring/grafana/provisioning/dashboards \
  monitoring/grafana/provisioning/datasources \
  .github/workflows
touch src/mlops/__init__.py
```

PowerShell (Windows):

``` powershell
"docker","feature_repo","src/mlops","scripts","loadtest","tests","docs","data","artifacts",
"monitoring/grafana/dashboards","monitoring/grafana/provisioning/dashboards",
"monitoring/grafana/provisioning/datasources",".github/workflows" |
  ForEach-Object { New-Item -ItemType Directory -Force $_ | Out-Null }
New-Item -ItemType File -Force src/mlops/__init__.py | Out-Null
```

> Create `data/` and `artifacts/` yourself, **before** you ever start Docker. If Docker creates
> them, they can end up owned by `root` on Linux, and the containers cannot write to them.

### File: `.gitignore`

Lists what Git must never track. Two kinds of things belong here: **generated files** (the
virtual environment, caches, data, models), which are big and can always be recreated, and
**secrets** (`.env`), which must never end up in a repository, even a private one.

``` gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.env
data/
artifacts/
feature_repo/data/
requirements-audit.txt
.DS_Store
.vscode/
```

### File: `.gitattributes`

Windows and Linux end lines differently (CRLF vs LF). Shell scripts that run inside Linux
containers break with Windows line endings, so this file forces LF for every `.sh` file, whatever
your editor does.

``` gitattributes
*.sh text eol=lf
```

### File: `.dockerignore`

When Docker builds an image it first copies the project folder (the "build context"). This file
keeps out what the image must not contain: your local virtual environment, your data, and above
all your `.env` secrets. It also makes builds faster.

``` text
.git
.venv
.env
data
artifacts
feature_repo/data
**/__pycache__
.pytest_cache
.ruff_cache
```

### File: `.env.example`

All configuration that differs between machines, and all secrets, live in environment
variables, not in the code. This file is the **template**: it lists every variable with a
placeholder value, and it *is* committed. Your real values go into `.env`, which is not.

Why separate passwords for each service? If one service is compromised, the attacker only gets
that service's database, not all of them. This is called **least privilege**.

``` dotenv
# Copy to .env and replace every value. Never commit .env.
# Generate strong values with:
#   uv run python -c "import secrets; print(secrets.token_urlsafe(24))"
# (token_urlsafe output is safe to use inside database URLs.)

POSTGRES_PASSWORD=change-me-postgres
MLFLOW_DB_PASSWORD=change-me-mlflow
DAGSTER_DB_PASSWORD=change-me-dagster
APP_DB_PASSWORD=change-me-app
GRAFANA_DB_PASSWORD=change-me-grafana-reader
GRAFANA_ADMIN_PASSWORD=change-me-grafana-admin

# Comma-separated list of accepted API keys.
API_KEYS=change-me-api-key

# Optional training parameter (used by the promotion-gate exercise in Section 16).
RIDGE_ALPHA=1.0

# Host port for Postgres (containers always use 5432 internally). Change it if 5432 is
# already used or blocked on your machine, and update DATABASE_URL below to match.
POSTGRES_HOST_PORT=5432

# Only used when you run Python natively with: uv run --env-file .env ...
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
DATABASE_URL=postgresql://app:change-me-app@127.0.0.1:5432/monitoring
```

Now create your real `.env`:

``` bash
cp .env.example .env        # PowerShell: Copy-Item .env.example .env
```

Generate a strong random value for every `change-me-*` entry:

``` bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

Run it once per value and paste the results into `.env`. Use the same value for
`APP_DB_PASSWORD` and the password inside `DATABASE_URL`.

> Postgres reads these passwords only the **first** time it starts. Choose them now; changing
> them later means resetting the database.

### Check

``` bash
git status
```

`.env` must **not** appear in the list (it is ignored). `.env.example` must appear.

------------------------------------------------------------------------

## Step 3: Dependencies

**Goal:** a locked, reproducible set of Python packages.

### File: `pyproject.toml`

This file defines the project: its name, its Python version and its dependencies. The
dependencies are split into **three groups**, so that each container installs only what it
needs:

- **main dependencies**: what the prediction API needs. The API image installs only these,
  which keeps it small (no Java, no Spark).
- **`pipeline` group**: PySpark, Feast, Evidently, Dagster: the heavy tools used for data
  processing, training and monitoring.
- **`dev` group**: tests, linting and load testing. Only needed on your machine and in CI.

Version ranges such as `>=2.12,<3` allow bug-fix updates but block major versions that could
break the code. The comments in the file explain the few places where a tighter limit is
needed. `[project.scripts]` creates the `mlops-model` command used in Step 9, and the
`[tool.ruff]` section configures the linter.

``` toml
[project]
name = "modern-mlops"
version = "0.2.0"
description = "Local-first, production-style MLOps student project"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
  "fastapi>=0.110,<1",
  "uvicorn[standard]>=0.27,<1",
  "pydantic>=2.6,<3",
  "pandas>=2.1,<3",
  "numpy>=1.26,<3",
  "scikit-learn>=1.4,<1.7",
  "mlflow>=2.12,<3",
  # MLflow 2.x's server imports FallbackAsyncAdaptedQueuePool, which SQLAlchemy 2.1 removed.
  # MLflow only declares sqlalchemy<3, so the cap has to be set here.
  "sqlalchemy>=2.0,<2.1",
  "psycopg[binary,pool]>=3.1,<4",
  "prometheus-client>=0.20,<1",
]

[project.scripts]
mlops-model = "mlops.registry:main"

[dependency-groups]
pipeline = [
  "pyspark>=3.5.1,<3.6",
  "pyarrow>=14,<20",
  "feast>=0.40,<0.50",
  "evidently>=0.7,<0.8",
  "dagster>=1.8,<2",
  "dagster-webserver>=1.8,<2",
  "dagster-postgres>=0.24,<1",
  # Driver for the MLflow server's Postgres backend. MLflow is tested with psycopg2; with
  # psycopg 3 some of its queries fail ("operator does not exist: integer = character varying").
  "psycopg2-binary>=2.9,<3",
]
dev = [
  "pytest>=8,<9",
  "httpx>=0.27,<1",
  "ruff>=0.6,<1",
  "locust>=2.20,<3",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mlops"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.ruff.lint.isort]
known-first-party = ["mlops"]
```

Install everything:

``` bash
uv lock                  # resolve all versions and write uv.lock
uv sync --all-groups     # create .venv and install every group
```

`uv.lock` records the **exact** version of every package (more than 200). Commit it: it is what
makes the project build identically on every machine.

In VS Code, open the Command Palette → **Python: Select Interpreter** → choose the interpreter
in `.venv`.

### Check

``` bash
uv run python -c "import pyspark, mlflow, fastapi, feast, evidently, dagster; print('imports OK')"
```

------------------------------------------------------------------------

## Step 4: Settings and logging

**Goal:** one place for all configuration, and logs a machine can read.

### File: `src/mlops/settings.py`

Every other module imports its configuration from here, so a path, a column name or a threshold
is defined **once**. Each value has a sensible default and can be overridden with an
environment variable. That is how the same code runs on your laptop and inside a container: only
the environment changes.

Key parts:

- **`ROOT`**: the project folder. Inside containers it comes from `PROJECT_ROOT`.
- **Data contract**: the column names the whole project relies on: the ID, the event time, the
  five features and the target (`SalePrice`).
- **Thresholds**: the numbers behind every automatic decision: the minimum R² for a model, the
  drift and accuracy limits for monitoring. Seeing them side by side makes the system's rules
  easy to review.
- **`api_keys()`**: a function rather than a constant, so the value is read when it is
  needed. Tests can then set keys without restarting Python.

``` python
"""Project configuration. Every value can be overridden with an environment variable."""

import os
from pathlib import Path

# In containers the package is installed into site-packages, so the project root is passed in.
ROOT = Path(os.getenv("PROJECT_ROOT") or Path(__file__).resolve().parents[2])

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "housing.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "housing.parquet"
FEATURES_PATH = DATA_DIR / "features" / "housing_features.parquet"
REPORTS_DIR = ROOT / "artifacts" / "reports"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation.json"
DATA_SUMMARY_PATH = REPORTS_DIR / "data_summary.json"
FEATURE_REPO = ROOT / "feature_repo"

# Data contract
ID_COLUMN = "Id"
ENTITY_KEY = "house_id"
TIMESTAMP = "event_timestamp"
FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]
TARGET = "SalePrice"

# Data validation
MIN_ROWS = int(os.getenv("MIN_ROWS", "1000"))
MAX_NULL_RATE = float(os.getenv("MAX_NULL_RATE", "0.05"))
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "2g")

# Training and promotion gate
RIDGE_ALPHA = float(os.getenv("RIDGE_ALPHA", "1.0"))
TEST_FRACTION = 0.2
MIN_R2 = float(os.getenv("MIN_R2", "0.70"))

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "house-price-regression")
MODEL_NAME = os.getenv("MODEL_NAME", "house-price-model")
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

# Serving
DATABASE_URL = os.getenv("DATABASE_URL", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Monitoring
DRIFT_WINDOW_DAYS = 7
DRIFT_MIN_ROWS = 50
DRIFT_SHARE_THRESHOLD = 0.5  # fail if more than half of the features drifted
QUALITY_WINDOW_DAYS = 30
QUALITY_MIN_ROWS = 30
QUALITY_RMSE_RATIO = 1.5  # fail if live RMSE is 50% worse than the holdout RMSE


def api_keys() -> set[str]:
    """Read on each call, so tests can set keys and a restart picks up rotated keys."""
    return {key.strip() for key in os.getenv("API_KEYS", "").split(",") if key.strip()}


def require_database_url() -> str:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set.")
    return DATABASE_URL
```

### File: `src/mlops/logging_config.py`

Plain-text logs are fine for a person, but hard for tools to search. Here every log line is a
**JSON object** with fixed fields (`time`, `level`, `message`, ...).

The most useful field is **`request_id`**: the API gives every incoming request an ID and adds
it to every log line written while handling that request. When a user reports a problem, you
can find every line about their request by searching for one ID.

- **`request_id_var`**: a *context variable*: it holds a different value for each request,
  even when many requests are handled at the same time.
- **`JsonFormatter`**: turns a log record into one JSON line, including any extra fields
  passed with `logger.info(..., extra={...})`.
- **`configure_logging()`**: installs the formatter, including for the web server's own logs.

``` python
"""JSON logging with a request ID on every line."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        # Anything passed with logger.info(..., extra={...}) is added as a field.
        payload.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Route uvicorn's own loggers through the same JSON handler.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
```

------------------------------------------------------------------------

## Step 5: Data: validate and prepare with PySpark

**Goal:** turn the raw CSV into clean training data, and **refuse** to continue when the data is
bad.

Why so much care about data? A model trained on broken data is broken in ways no metric reveals.
A column that was suddenly exported in different units, or half the rows missing, can produce a
model that looks fine but is wrong. Production systems therefore check data **before** using it,
the same way they test code before shipping it.

### File: `src/mlops/validation.py`

This file is the **data contract**: the rules the data must respect.

- **`RANGES`**: the allowed minimum and maximum for each numeric column. A house with an
  `OverallQual` of 42 (on a 1–10 scale) is an error, not a very good house.
- **`ValidationReport`**: the result: row count, missing-value rates, out-of-range counts,
  duplicate IDs and a list of human-readable **failures**. An empty list means "passed".
- **`validate()`**: runs four kinds of checks: required columns are present, there are enough
  rows, values are within range and not too often missing, and IDs are unique. It never raises;
  it returns a report and lets the caller decide.

``` python
"""Data-quality rules that must pass before any model is trained."""

from dataclasses import asdict, dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from mlops import settings

# Allowed (minimum, maximum) for each numeric column. Keep in sync with the API limits.
RANGES: dict[str, tuple[float, float]] = {
    "OverallQual": (1, 10),
    "GrLivArea": (1, 10_000),
    "GarageCars": (0, 10),
    "TotalBsmtSF": (0, 10_000),
    "YearBuilt": (1800, 2100),
    "SalePrice": (1, 10_000_000),
    "YrSold": (1900, 2100),
    "MoSold": (1, 12),
}


class ValidationError(Exception):
    """Raised when the dataset breaks one or more data-quality rules."""


@dataclass
class ValidationReport:
    row_count: int
    null_rates: dict[str, float] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    duplicate_ids: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {**asdict(self), "passed": self.passed}


def validate(
    df: DataFrame,
    min_rows: int = settings.MIN_ROWS,
    max_null_rate: float = settings.MAX_NULL_RATE,
) -> ValidationReport:
    """Check the raw data against the rules above and return a report (never raises).

    The caller decides what to do with a failed report; in the pipeline it stops the run.
    """
    id_column = settings.ID_COLUMN
    required = [id_column, *RANGES]

    # 1. Structure: are all the columns we rely on present?
    missing = [column for column in required if column not in df.columns]
    if missing:
        return ValidationReport(row_count=0, failures=[f"Missing required columns: {missing}"])

    # 2. Volume: a file that is suddenly much smaller usually means a broken export.
    row_count = df.count()
    if row_count < min_rows:
        return ValidationReport(
            row_count=row_count,
            failures=[f"Only {row_count} rows; at least {min_rows} are required."],
        )

    # 3. Content: count missing and out-of-range values for every column in ONE pass over the
    #    data. Building all the expressions first and running a single agg() is much faster
    #    than running one Spark job per column.
    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"null:{c}") for c in required
    ]
    range_exprs = [
        F.sum(F.when((F.col(c) < low) | (F.col(c) > high), 1).otherwise(0)).alias(f"range:{c}")
        for c, (low, high) in RANGES.items()
    ]
    counts = df.agg(*null_exprs, *range_exprs).first().asDict()

    null_rates = {c: counts[f"null:{c}"] / row_count for c in required}
    out_of_range = {c: int(counts[f"range:{c}"]) for c in RANGES}
    duplicate_ids = row_count - df.select(id_column).distinct().count()

    # 4. Turn the numbers into readable failure messages. An empty list means "passed".
    failures: list[str] = []
    if null_rates[id_column] > 0:  # every row needs an ID: it becomes the entity key
        failures.append(f"{null_rates[id_column]:.1%} of rows have no {id_column}.")
    for column, rate in null_rates.items():
        if column != id_column and rate > max_null_rate:
            failures.append(f"{column}: {rate:.1%} missing values (limit {max_null_rate:.0%}).")
    for column, count in out_of_range.items():
        if count:
            failures.append(f"{column}: {count} values outside {RANGES[column]}.")
    if duplicate_ids:
        failures.append(f"{duplicate_ids} duplicate {id_column} values.")

    return ValidationReport(
        row_count=row_count,
        null_rates=null_rates,
        out_of_range=out_of_range,
        duplicate_ids=duplicate_ids,
        failures=failures,
    )
```

### File: `src/mlops/data.py`

This file downloads the data and prepares it with PySpark.

- **`file_sha256()`**: computes a fingerprint of the raw file. It is stored with every model,
  so you can always prove which exact data a model was trained on.
- **`download_raw()`**: downloads the Ames housing dataset **once** and reuses the file. The
  file is first written under a temporary name and then renamed, so a crash can never leave a
  half-written file behind.
- **`create_spark()`**: starts Spark in *local mode* (on your machine, using 2 CPU cores). The
  comments explain two settings that avoid common connection problems.
- **`load_raw()`**: reads every column as text, then converts the numeric ones explicitly.
  Values such as `"NA"` become *null* (missing) instead of breaking the load.
- **`prepare_data()`**: the main function: **validate first**, stop if the rules fail, then
  clean the data and add two important columns:
  - `house_id`, from the dataset's own `Id`: a **stable key** that means the same house every
    time;
  - `event_timestamp`, the **sale date**: it tells us *when* each fact was true. Step 6 and the
    train/test split in Step 8 both rely on it.

``` python
"""Download the raw dataset and prepare it with PySpark."""

import hashlib
import json
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from sklearn.datasets import fetch_openml

from mlops import settings
from mlops.validation import ValidationError, validate

REQUIRED_COLUMNS = [settings.ID_COLUMN, "YrSold", "MoSold", *settings.FEATURES, settings.TARGET]


def file_sha256(path: Path) -> str:
    """Fingerprint of a file: any change to the data, even one character, changes the hash."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw(force: bool = False) -> Path:
    """Download the Ames housing data once. The file is then reused, so its hash stays stable."""
    if settings.RAW_PATH.exists() and not force:
        return settings.RAW_PATH
    dataset = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
    frame = dataset.frame
    if frame is None or frame.empty:
        raise RuntimeError("OpenML returned an empty dataset.")
    if settings.ID_COLUMN not in frame.columns:
        # Some OpenML versions drop the row-id column. Rows keep the source order, and the raw
        # file is versioned by its hash, so this key is stable for a given raw file.
        frame.insert(0, settings.ID_COLUMN, range(1, len(frame) + 1))
    settings.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.RAW_PATH.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(settings.RAW_PATH)  # atomic: never leaves a half-written file
    return settings.RAW_PATH


def create_spark(app_name: str = "modern-mlops") -> SparkSession:
    # Spark workers must run the same Python as this process (with the same packages installed).
    # Without this, Spark starts whatever "python" is first on PATH, which fails on many machines.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.appName(app_name)
        .master(os.getenv("SPARK_MASTER", "local[2]"))
        .config("spark.driver.memory", settings.SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        # Local mode only needs loopback. Without this, Spark can resolve the machine name to
        # another address (e.g. Docker Desktop's host.docker.internal) and workers cannot connect.
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )


def load_raw(spark: SparkSession) -> DataFrame:
    # Read everything as text, then cast explicitly: values like "NA" become null.
    df = spark.read.option("header", True).csv(str(settings.RAW_PATH))
    for column in REQUIRED_COLUMNS:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast(DoubleType()))
    return df


def prepare_data() -> dict:
    """Validate the raw CSV, clean it and write it as Parquet. Returns a small summary.

    Raises ValidationError when the data breaks a rule, so no model is trained on bad data.
    """
    if not settings.RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {settings.RAW_PATH}.")

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    spark = create_spark()
    try:
        raw = load_raw(spark)
        # Validate the data as it arrived, BEFORE any cleaning, and always save the report
        # (also when it fails: that is exactly when someone needs to read it).
        report = validate(raw)
        settings.VALIDATION_REPORT_PATH.write_text(json.dumps(report.to_dict(), indent=2))
        if not report.passed:
            raise ValidationError("; ".join(report.failures))

        # The sale date (year + month) becomes the event time. Day 1 is used because the
        # dataset only records the month of the sale.
        sale_date = F.format_string(
            "%04d-%02d-01", F.col("YrSold").cast("int"), F.col("MoSold").cast("int")
        )
        clean = (
            # Validation guaranteed that at most 5% of values are missing, so dropping those
            # rows is safe. Without that check, this line could silently remove most of the data.
            raw.dropna(subset=REQUIRED_COLUMNS)
            .withColumn(settings.ENTITY_KEY, F.col(settings.ID_COLUMN).cast("long"))
            .withColumn(settings.TIMESTAMP, F.to_timestamp(sale_date, "yyyy-MM-dd"))
            .select(settings.ENTITY_KEY, settings.TIMESTAMP, *settings.FEATURES, settings.TARGET)
        )
        settings.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        clean.write.mode("overwrite").parquet(str(settings.PROCESSED_PATH))
        # Count what was actually written, not what we think we wrote.
        clean_rows = spark.read.parquet(str(settings.PROCESSED_PATH)).count()
    finally:
        spark.stop()

    summary = {
        "raw_sha256": file_sha256(settings.RAW_PATH),
        "raw_rows": report.row_count,
        "clean_rows": clean_rows,
        "dropped_rows": report.row_count - clean_rows,
    }
    settings.DATA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    download_raw()
    print(json.dumps(prepare_data(), indent=2))
```

PySpark writes Parquet as a **folder** containing one or more `part-*` files, not as a single
file. This is normal.

> On Windows, run the pipeline inside Docker (Step 19): native PySpark on Windows needs extra
> Hadoop tools to write files. On macOS/Linux you can also run it directly:
> `uv run python -m mlops.data`.

------------------------------------------------------------------------

## Step 6: Feature store: Feast

**Goal:** build the training data in a way that can never "see the future".

A **feature store** is a central place where features are defined once and served to both
training and prediction. Here Feast has one precise job: building the training set with a
**point-in-time join**.

Why does that matter? Imagine predicting a house's price using its "number of renovations". If
you join today's value onto a sale from 2008, renovations made *after* the sale leak into the
training data. The model learns from information it will never have at prediction time, looks
excellent in testing, and disappoints in reality. This is called **data leakage**, and it is one
of the most common silent bugs in machine learning. A point-in-time join only uses values that
were known **at the time of each sale**.

In this dataset the features never change, so the join looks trivial, but the pipeline is built
the right way from the start.

### File: `feature_repo/feature_store.yaml`

Feast's configuration: the project name, where Feast keeps its registry (the list of defined
features) and which stores to use. Everything is local files. The *online store* (used for
very fast lookups at prediction time) must be configured, but this project does not use it: the
API receives the features directly in each request.

``` yaml
project: modern_mlops
provider: local
registry: data/registry.db
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 2
```

### File: `src/mlops/features.py`

- **`house`** (an *entity*): the thing the features describe, here one house sale, identified by
  `house_id`.
- **`housing_source`**: where the feature values are stored, and which column holds the time.
- **`housing_features`** (a *feature view*): the five features of the house, from that source.
- **`build_feature_source()`**: writes the feature table, **without the price**. Labels and
  features are kept separate: the price is what we predict, not a feature.
- **`apply_definitions()`**: registers the definitions with Feast.
- **`get_training_frame()`**: takes each sale (ID, date, price) and asks Feast for the feature
  values valid at that date. The result is the training data.

``` python
"""Feast feature definitions, feature-source build and training-data retrieval."""

from datetime import timedelta

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float64

from mlops import settings

# An entity is "the thing features describe". Here: one house sale, identified by house_id.
house = Entity(
    name="house",
    join_keys=[settings.ENTITY_KEY],
    description="One house sale in the Ames housing dataset.",
)

# Where the feature values live, and which column says WHEN each value was true.
housing_source = FileSource(
    name="housing_source",
    path=settings.FEATURES_PATH.as_posix(),
    timestamp_field=settings.TIMESTAMP,
)

# A feature view groups features of one entity from one source. ttl = how old a feature value
# may be (relative to the event time) and still be used.
housing_features = FeatureView(
    name="housing_features",
    entities=[house],
    ttl=timedelta(days=30),
    schema=[Field(name=name, dtype=Float64) for name in settings.FEATURES],
    source=housing_source,
    online=False,  # the API receives features in the request; no online store is needed
)


def get_store() -> FeatureStore:
    return FeatureStore(repo_path=str(settings.FEATURE_REPO))


def build_feature_source() -> int:
    """Write the feature table (no labels) as a single Parquet file for Feast."""
    columns = [settings.ENTITY_KEY, settings.TIMESTAMP, *settings.FEATURES]
    frame = pd.read_parquet(settings.PROCESSED_PATH, columns=columns)
    settings.FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(settings.FEATURES_PATH, index=False)
    return len(frame)


def apply_definitions() -> None:
    get_store().apply([house, housing_features])


def get_training_frame() -> pd.DataFrame:
    """Labels come from the processed data; features come from Feast (point-in-time join)."""
    labels = pd.read_parquet(
        settings.PROCESSED_PATH, columns=[settings.ENTITY_KEY, settings.TIMESTAMP, settings.TARGET]
    )
    feature_refs = [f"{housing_features.name}:{name}" for name in settings.FEATURES]
    # For each (house_id, event_timestamp) in `labels`, Feast looks up the latest feature values
    # recorded at or before that time: a "point-in-time join". It never uses future values.
    frame = get_store().get_historical_features(entity_df=labels, features=feature_refs).to_df()

    missing = [c for c in [*settings.FEATURES, settings.TARGET] if c not in frame.columns]
    if missing:
        raise ValueError(f"Feast output is missing columns: {missing}")
    if frame[settings.FEATURES].isna().any().any():
        raise ValueError("Feast returned rows without features; check the TTL and timestamps.")
    return frame.sort_values(settings.TIMESTAMP).reset_index(drop=True)
```

------------------------------------------------------------------------

## Step 7: Evaluation and the promotion gate

**Goal:** define, in code, what "good enough to go live" means.

A metric such as RMSE tells you how wrong a model is **on average**. It does not tell you
whether the model behaves sensibly, or whether it is better than the model already in
production. The **promotion gate** combines three rules:

1. **Quality floor**: R² on the test data must be at least 0.70.
2. **Behaviour checks**: a bigger or better-built house must get a higher price. A model that
   breaks this rule is wrong, whatever its RMSE says.
3. **No regression**: the new model must not be worse than the current champion, measured on
   **the same** test data.

### File: `src/mlops/evaluation.py`

This module contains no database, no MLflow and no Spark: only calculations. That makes it fast
and easy to test (Step 14).

- **`regression_metrics()`**: MAE, RMSE, R² and MAPE. The docstring explains each one.
- **`REFERENCE_HOUSE`** and **`MONOTONIC_CHECKS`**: an ordinary house, and the changes that
  must increase its price.
- **`behaviour_checks()`**: predicts the reference house, then the same house made bigger or
  better, and reports any case where the price does not go up.
- **`GateDecision`** and **`promotion_gate()`**: apply the three rules and return *passed /
  failed* together with the reasons. A rejected model always says why.

``` python
"""Metrics, behaviour checks and the promotion gate. No I/O, easy to unit test."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops import settings


class Regressor(Protocol):
    """Anything with a predict() method. Lets the tests use a tiny fake model."""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


REFERENCE_HOUSE = {
    "OverallQual": 6.0,
    "GrLivArea": 1500.0,
    "GarageCars": 2.0,
    "TotalBsmtSF": 1000.0,
    "YearBuilt": 1990.0,
}
# Increasing these features must increase the predicted price.
MONOTONIC_CHECKS = {"GrLivArea": 500.0, "OverallQual": 2.0}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Standard regression metrics.

    mae:  average absolute error, in dollars. Easy to explain ("off by $20k on average").
    rmse: like MAE but punishes large errors more. Used by the promotion gate.
    r2:   share of the price variation the model explains (1.0 = perfect, 0 = useless).
    mape: average error as a percentage of the real price.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true))),
    }


def _predict_one(model: Regressor, house: dict[str, float]) -> float:
    frame = pd.DataFrame([house], columns=settings.FEATURES)
    return float(np.asarray(model.predict(frame))[0])


def behaviour_checks(model: Regressor) -> list[str]:
    """Sanity checks a good metric cannot replace. Returns a list of failures (empty = pass).

    We predict one ordinary "reference" house, then the same house made bigger or better, and
    require the price to go up.
    """
    base = _predict_one(model, REFERENCE_HOUSE)
    if not np.isfinite(base) or base <= 0:
        return [f"Reference house predicted at {base}; expected a positive price."]
    failures = []
    for feature, step in MONOTONIC_CHECKS.items():
        bigger = {**REFERENCE_HOUSE, feature: REFERENCE_HOUSE[feature] + step}
        if _predict_one(model, bigger) <= base:
            failures.append(f"Price did not increase when {feature} increased by {step:g}.")
    return failures


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def promotion_gate(
    candidate: dict[str, float],
    champion: dict[str, float] | None,
    behaviour_failures: list[str],
    min_r2: float,
) -> GateDecision:
    """Decide whether a candidate model may become the challenger.

    Every rule that fails adds a human-readable reason, so a rejected model always says why.
    `champion` is None for the very first model; then only the absolute rules apply.
    """
    reasons = list(behaviour_failures)
    if candidate["r2"] < min_r2:
        reasons.append(f"R2 {candidate['r2']:.3f} is below the minimum {min_r2:.2f}.")
    if champion is not None and candidate["rmse"] > champion["rmse"]:
        reasons.append(
            f"RMSE {candidate['rmse']:,.0f} is worse than the champion's {champion['rmse']:,.0f}."
        )
    return GateDecision(passed=not reasons, reasons=reasons)
```

------------------------------------------------------------------------

## Step 8: Training with MLflow

**Goal:** train a candidate model, record everything about it, and run the gate.

**MLflow** has two roles here:

- **Experiment tracking**: every training run records its parameters, metrics and files, so
  you can compare runs and reproduce any of them.
- **Model registry**: every registered model gets a **version number** (1, 2, 3, ...). Versions
  can carry **aliases**, movable labels such as `champion` and `challenger`. The API always asks
  for "the champion", so changing which model is live means moving a label, not changing code.

Three design choices to understand:

- **Time-based split.** The newest 20% of sales are kept for testing; the model trains on the
  older ones. A model is always used on houses sold *after* it was trained, so this is the
  honest way to measure it. A random split would let the model "see" sales from the test period.
- **Log of the price.** Prices are skewed (a few very expensive houses). Training on
  `log(price)` makes a 10% error on a cheap house count about as much as a 10% error on an
  expensive one.
- **The candidate never goes live automatically.** Passing the gate only earns the `challenger`
  alias. A person decides, in Step 9.

### File: `src/mlops/train.py`

- **`build_model()`**: the model: Ridge regression on `log(price)`, packaged with its
  preprocessing (filling missing values, scaling). Because the preprocessing is part of the
  saved model, prediction always uses exactly the same steps as training.
- **`time_split()`**: the date-based train/test split.
- **`load_champion()`**: loads the current champion, if one exists, to compare against.
- **`render_model_card()`**: writes a **model card**: a short document describing what the
  model is for, the data it was trained on, its results, the gate's decision and its known
  limitations. It is stored with the model in MLflow.
- **`train_candidate()`**: the main function. Its numbered comments follow the steps: get data,
  record what is being used, train, register a new version, compare with the champion, apply the
  gate and write the decision on the version.

``` python
"""Train a candidate model, compare it with the champion and register it in MLflow."""

import json
import logging
from dataclasses import dataclass
from importlib.metadata import version as package_version

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops import settings
from mlops.evaluation import GateDecision, behaviour_checks, promotion_gate, regression_metrics
from mlops.features import get_training_frame

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    run_id: str
    model_version: str
    metrics: dict[str, float]
    champion_version: str | None
    champion_metrics: dict[str, float] | None
    decision: GateDecision


def build_model(alpha: float) -> TransformedTargetRegressor:
    """Ridge regression on log(price), packaged with its preprocessing.

    Keeping imputation and scaling inside the pipeline means the exact same preprocessing is
    saved with the model and applied at prediction time: no training/serving mismatch.
    """
    regressor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def time_split(frame: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Older sales for training, the newest `test_fraction` of sales for testing."""
    cutoff = frame[settings.TIMESTAMP].quantile(1 - test_fraction)
    train = frame[frame[settings.TIMESTAMP] < cutoff]
    test = frame[frame[settings.TIMESTAMP] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("Time-based split produced an empty train or test set.")
    return train, test


def load_champion(client: MlflowClient):
    try:
        champion = client.get_model_version_by_alias(settings.MODEL_NAME, settings.CHAMPION_ALIAS)
    except MlflowException:
        logger.info("No champion model yet; the gate will only apply the absolute rules.")
        return None
    model = mlflow.sklearn.load_model(f"models:/{settings.MODEL_NAME}/{champion.version}")
    return str(champion.version), model


def render_model_card(
    version: str,
    metrics: dict[str, float],
    champion_version: str | None,
    champion_metrics: dict[str, float] | None,
    decision: GateDecision,
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_summary: dict,
) -> str:
    def period(frame: pd.DataFrame) -> str:
        dates = frame[settings.TIMESTAMP]
        return f"{dates.min():%Y-%m} to {dates.max():%Y-%m} ({len(frame)} sales)"

    rows = [f"| {name} | {value:,.3f} |" for name, value in metrics.items()]
    if champion_metrics:
        comparison = f"Champion v{champion_version} RMSE on the same holdout: "
        comparison += f"{champion_metrics['rmse']:,.0f}"
    else:
        comparison = "No champion existed when this version was trained."
    gate = "PASSED" if decision.passed else "FAILED: " + "; ".join(decision.reasons)
    lines = [
        f"# Model card: {settings.MODEL_NAME} v{version}",
        "",
        "## Intended use",
        "Estimates the sale price of a house in Ames, Iowa, from five basic attributes.",
        "Student project: not for real valuations, lending or pricing decisions.",
        "",
        "## Training data",
        "- Source: OpenML `house_prices` (Ames housing dataset)",
        f"- Raw file SHA-256: `{data_summary.get('raw_sha256', 'unknown')}`",
        f"- Training period: {period(train)}",
        f"- Holdout period: {period(test)}",
        f"- Features: {', '.join(settings.FEATURES)}",
        "",
        "## Model",
        "Ridge regression on log(price), with median imputation and standard scaling.",
        "",
        "## Holdout metrics",
        "| metric | value |",
        "|---|---|",
        *rows,
        "",
        comparison,
        "",
        "## Promotion gate",
        gate,
        "",
        "## Known limitations",
        "- Sales from 2006-2010 in one city; current prices are very different.",
        "- Five features only: no location, condition or lot size.",
        "- A linear model on log price extrapolates poorly for unusual houses.",
    ]
    return "\n".join(lines)


def train_candidate(alpha: float = settings.RIDGE_ALPHA) -> TrainingResult:
    """Train, evaluate, register and gate one candidate model. Never changes the champion."""
    # 1. Data: features from Feast, split by date.
    frame = get_training_frame()
    train, test = time_split(frame, settings.TEST_FRACTION)
    X_train, y_train = train[settings.FEATURES].astype(float), train[settings.TARGET]
    X_test, y_test = test[settings.FEATURES].astype(float), test[settings.TARGET]
    data_summary = {}
    if settings.DATA_SUMMARY_PATH.exists():
        data_summary = json.loads(settings.DATA_SUMMARY_PATH.read_text())

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.EXPERIMENT_NAME)
    client = MlflowClient()

    with mlflow.start_run() as run:
        # 2. Traceability: record exactly which data, code and settings produced this model.
        mlflow.set_tags(
            {
                "raw_data_sha256": data_summary.get("raw_sha256", "unknown"),
                "code_version": package_version("modern-mlops"),
                "split": "time-based",
            }
        )
        mlflow.log_params(
            {
                "estimator": "Ridge on log1p(price)",
                "alpha": alpha,
                "features": ",".join(settings.FEATURES),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "train_end": f"{train[settings.TIMESTAMP].max():%Y-%m-%d}",
                "test_start": f"{test[settings.TIMESTAMP].min():%Y-%m-%d}",
            }
        )
        if settings.VALIDATION_REPORT_PATH.exists():
            mlflow.log_artifact(str(settings.VALIDATION_REPORT_PATH), artifact_path="data")

        # 3. Train on the older sales, measure on the newest ones.
        model = build_model(alpha).fit(X_train, y_train)
        metrics = regression_metrics(y_test, model.predict(X_test))
        mlflow.log_metrics(metrics)
        behaviour_failures = behaviour_checks(model)

        # 4. Store the model and register it as a new version. The signature records the
        #    expected input columns and types, so a wrong input is caught when loading.
        example = X_train.head(3)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
        )
        registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", settings.MODEL_NAME)
        version = str(registered.version)

        # 5. Compare with the current champion on the SAME holdout rows: a fair comparison.
        champion_version, champion_metrics = None, None
        champion = load_champion(client)
        if champion is not None:
            champion_version, champion_model = champion
            champion_metrics = regression_metrics(y_test, champion_model.predict(X_test))
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})

        # 6. Apply the gate and write the decision onto the model version, where anyone can
        #    see it in MLflow. Passing only earns the "challenger" alias: a person promotes it.
        decision = promotion_gate(metrics, champion_metrics, behaviour_failures, settings.MIN_R2)
        tags = {
            "gate_passed": str(decision.passed).lower(),
            "gate_reasons": "; ".join(decision.reasons) or "none",
            "holdout_rmse": f"{metrics['rmse']:.0f}",
            "compared_with_champion": champion_version or "none",
        }
        for key, value in tags.items():
            client.set_model_version_tag(settings.MODEL_NAME, version, key, value)
        if decision.passed:
            client.set_registered_model_alias(
                settings.MODEL_NAME, settings.CHALLENGER_ALIAS, version
            )

        card = render_model_card(
            version,
            metrics,
            champion_version,
            champion_metrics,
            decision,
            train,
            test,
            data_summary,
        )
        mlflow.log_text(card, "model_card.md")

    return TrainingResult(
        run_id=run.info.run_id,
        model_version=version,
        metrics=metrics,
        champion_version=champion_version,
        champion_metrics=champion_metrics,
        decision=decision,
    )


if __name__ == "__main__":
    result = train_candidate()
    print(f"Registered version {result.model_version}; gate passed: {result.decision.passed}")
    for reason in result.decision.reasons:
        print(f"  - {reason}")
```

------------------------------------------------------------------------

## Step 9: Model registry commands: status, promote, rollback

**Goal:** release and roll back models with simple, safe, logged commands.

Promotion is a **human decision supported by automation**: the gate checks the numbers, and a
person makes the final call after reading the model card. The command enforces the rules: it
refuses any version that did not pass the gate.

Every promotion records who did it, when, and which version it replaced. That makes rollback
simple: the command already knows where to go back to.

### File: `src/mlops/registry.py`

- **`status()`**: shows which versions are champion and challenger, with their test RMSE.
- **`promote()`**: moves the `champion` alias to a version (the challenger by default), after
  checking that it passed the gate. It records the previous champion on the new version.
- **`rollback()`**: moves `champion` back to the previous version (or one you choose).
- **`main()`**: the command-line interface: `mlops-model status | promote | rollback`.

``` python
"""Command-line tool to inspect, promote and roll back registered models."""

import argparse
import getpass
from datetime import UTC, datetime

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlops import settings

NAME = settings.MODEL_NAME
CHAMPION = settings.CHAMPION_ALIAS
CHALLENGER = settings.CHALLENGER_ALIAS


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)


def _alias_version(client: MlflowClient, alias: str) -> str | None:
    """Version number an alias points to, or None if the alias is not set."""
    try:
        return str(client.get_model_version_by_alias(NAME, alias).version)
    except MlflowException:
        return None


def status() -> None:
    client = _client()
    for alias in (CHAMPION, CHALLENGER):
        version = _alias_version(client, alias)
        if version is None:
            print(f"{alias:<11} (none)")
            continue
        tags = client.get_model_version(NAME, version).tags
        print(
            f"{alias:<11} v{version}  holdout RMSE {tags.get('holdout_rmse', '?')}"
            f"  gate passed: {tags.get('gate_passed', '?')}"
        )


def promote(version: str | None) -> str:
    """Make a version the champion. Defaults to the current challenger.

    Refuses any version that did not pass the gate, and records who promoted it, when, and
    which version it replaced (so rollback knows where to go back to).
    """
    client = _client()
    version = version or _alias_version(client, CHALLENGER)
    if version is None:
        raise SystemExit("No challenger to promote. Run the training job first.")
    tags = client.get_model_version(NAME, version).tags
    if tags.get("gate_passed") != "true":
        raise SystemExit(f"v{version} did not pass the gate: {tags.get('gate_reasons')}")
    previous = _alias_version(client, CHAMPION)
    if previous == version:
        raise SystemExit(f"v{version} is already the champion.")

    client.set_registered_model_alias(NAME, CHAMPION, version)
    client.set_model_version_tag(NAME, version, "previous_champion", previous or "none")
    client.set_model_version_tag(NAME, version, "promoted_at", datetime.now(UTC).isoformat())
    client.set_model_version_tag(NAME, version, "promoted_by", getpass.getuser())
    if _alias_version(client, CHALLENGER) == version:
        client.delete_registered_model_alias(NAME, CHALLENGER)
    print(f"v{version} is now the champion (previous: {previous or 'none'}).")
    print("Restart the API to serve it: docker compose restart api")
    return version


def rollback(version: str | None) -> str:
    """Point the champion alias back to an older version. Defaults to the previous champion."""
    client = _client()
    current = _alias_version(client, CHAMPION)
    if current is None:
        raise SystemExit("There is no champion to roll back.")
    if version is None:
        version = client.get_model_version(NAME, current).tags.get("previous_champion")
        if not version or version == "none":
            raise SystemExit("The champion has no recorded previous version; pass --version.")
    client.get_model_version(NAME, version)  # fails loudly if the version does not exist

    client.set_registered_model_alias(NAME, CHAMPION, version)
    client.set_model_version_tag(NAME, version, "rolled_back_from", current)
    client.set_model_version_tag(NAME, version, "rolled_back_at", datetime.now(UTC).isoformat())
    print(f"Rolled back: v{version} is the champion again (was v{current}).")
    print("Restart the API to serve it: docker compose restart api")
    return version


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mlops-model", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show the champion and challenger versions.")
    for name in ("promote", "rollback"):
        command = commands.add_parser(name)
        command.add_argument("--version", help="Model version (default: the obvious one).")
    args = parser.parse_args(argv)

    if args.command == "status":
        status()
    elif args.command == "promote":
        promote(args.version)
    else:
        rollback(args.version)


if __name__ == "__main__":
    main()
```

You will use it inside the API container: `docker compose exec api mlops-model status`.

------------------------------------------------------------------------

## Step 10: Storing predictions in Postgres

**Goal:** keep a durable record of every prediction, so that monitoring can measure the model.

Why a database and not a CSV file? Several requests can arrive at the same time; a database
handles that safely. Each prediction gets an **ID**, so the real sale price can be attached
later (this is the `feedback`). And Grafana and the monitoring checks can query the data with
SQL.

### File: `src/mlops/db.py`

- **`SCHEMA`**: the three tables, described in the comment above it. `REFERENCES` links each
  feedback row to an existing prediction, so feedback for an unknown ID is rejected.
- **`PredictionStore`**: all database access in one class, through a **connection pool**:
  opening a database connection is slow, so a few connections are opened once and reused.
- **`_ensure_schema()`**: creates the tables on first use. Because it runs on first use rather
  than at startup, the API recovers on its own if Postgres was not ready when it started.
- **`log_prediction()`**, **`add_feedback()`**: write data. `add_feedback` returns `False`
  for an unknown prediction ID.
- **`recent_predictions()`**, **`labelled_predictions()`**: read data for the monitoring checks.
- **`record_check()`**: saves the result of a monitoring check.

``` python
"""Postgres storage for predictions, feedback and monitoring results."""

import threading
import uuid
from datetime import timedelta
from typing import Any

import pandas as pd
from psycopg import errors
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

# Three tables:
#   predictions         one row per /predict call (the inputs, the price, the model version)
#   feedback            the real sale price, sent later, linked to a prediction by its ID
#   monitoring_results  one row per monitoring check run (shown in Grafana)
# "IF NOT EXISTS" makes the schema safe to apply again on every start.
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   UUID PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version   TEXT NOT NULL,
    features        JSONB NOT NULL,
    predicted_price DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_created_at_idx ON predictions (created_at);

CREATE TABLE IF NOT EXISTS feedback (
    prediction_id UUID PRIMARY KEY REFERENCES predictions (prediction_id),
    actual_price  DOUBLE PRECISION NOT NULL CHECK (actual_price > 0),
    received_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS monitoring_results (
    id            BIGSERIAL PRIMARY KEY,
    checked_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    check_name    TEXT NOT NULL,
    model_version TEXT,
    value         DOUBLE PRECISION,
    threshold     DOUBLE PRECISION,
    passed        BOOLEAN NOT NULL,
    details       JSONB
);
"""


class PredictionStore:
    """All database access in one place, through a small pool of reusable connections."""

    def __init__(self, url: str, max_size: int = 5, timeout: float = 2.0) -> None:
        # timeout: how long a request waits for a connection before failing fast.
        self._pool = ConnectionPool(
            url,
            min_size=1,
            max_size=max_size,
            timeout=timeout,
            open=True,
            kwargs={"autocommit": True},
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def close(self) -> None:
        self._pool.close()

    def _ensure_schema(self) -> None:
        # Created lazily, so the API recovers if Postgres was down when it started.
        if self._schema_ready:
            return
        with self._schema_lock:
            if not self._schema_ready:
                with self._pool.connection() as conn:
                    conn.execute(SCHEMA)
                self._schema_ready = True

    def _query(self, sql: str, params: tuple) -> pd.DataFrame:
        self._ensure_schema()
        with self._pool.connection() as conn:
            cursor = conn.execute(sql, params)
            columns = [column.name for column in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=columns)

    def log_prediction(
        self,
        prediction_id: uuid.UUID,
        model_version: str,
        features: dict[str, Any],
        predicted_price: float,
    ) -> None:
        self._ensure_schema()
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO predictions (prediction_id, model_version, features, predicted_price)"
                " VALUES (%s, %s, %s, %s)",
                (prediction_id, model_version, Jsonb(features), predicted_price),
            )

    def add_feedback(self, prediction_id: uuid.UUID, actual_price: float) -> bool:
        """Store the real sale price. Returns False if the prediction ID is unknown."""
        self._ensure_schema()
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    "INSERT INTO feedback (prediction_id, actual_price) VALUES (%s, %s)"
                    " ON CONFLICT (prediction_id) DO UPDATE"
                    " SET actual_price = EXCLUDED.actual_price, received_at = now()",
                    (prediction_id, actual_price),
                )
        except errors.ForeignKeyViolation:
            return False
        return True

    def recent_predictions(self, days: int) -> pd.DataFrame:
        return self._query(
            "SELECT prediction_id, created_at, model_version, features, predicted_price"
            " FROM predictions WHERE created_at >= now() - %s",
            (timedelta(days=days),),
        )

    def labelled_predictions(self, days: int, model_version: str) -> pd.DataFrame:
        return self._query(
            "SELECT p.prediction_id, p.created_at, p.predicted_price, f.actual_price"
            " FROM predictions p JOIN feedback f USING (prediction_id)"
            " WHERE p.model_version = %s AND p.created_at >= now() - %s",
            (model_version, timedelta(days=days)),
        )

    def record_check(
        self,
        check_name: str,
        model_version: str | None,
        value: float | None,
        threshold: float,
        passed: bool,
        details: dict[str, Any],
    ) -> None:
        self._ensure_schema()
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO monitoring_results"
                " (check_name, model_version, value, threshold, passed, details)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (check_name, model_version, value, threshold, passed, Jsonb(details)),
            )
```

------------------------------------------------------------------------

## Step 11: The prediction API

**Goal:** a secure, observable web service that serves the champion model.

This is the part of the system that clients use. It has to be correct, but also secure,
measurable and resilient.

| Endpoint | Purpose |
|---|---|
| `GET /health` | *Liveness:* is the process running? |
| `GET /ready` | *Readiness:* is a model loaded, so requests can be served? |
| `POST /predict` | Predict a price (requires an API key). |
| `POST /feedback` | Send the real price for an earlier prediction (requires an API key). |
| `GET /metrics` | Metrics for Prometheus. |

Why both `/health` and `/ready`? A service can be running but not yet able to work, for example
while it waits for MLflow. Separating the two lets the infrastructure tell "restart it" apart
from "wait for it".

### File: `src/mlops/api.py`

The file is organised in sections:

- **Metrics**: counters and histograms that Prometheus reads: requests per route and status
  code, request duration, predictions per model version, predicted prices, failed database
  writes, and whether a model is loaded.
- **Model loading**: `load_champion()` asks MLflow which version is the champion and loads it,
  **once**. `ModelHolder` keeps it in memory and, if loading failed (for example because MLflow
  was still starting), tries again at most every 30 seconds instead of on every request.
- **Dependencies**: small functions that FastAPI runs before an endpoint: `get_model()` returns
  the model or answers 503, `get_store()` returns the database access, and `require_api_key()`
  checks the `X-API-Key` header. Keys are compared with `secrets.compare_digest`, which takes the
  same time whether the key is almost right or completely wrong, so timing reveals nothing. If no
  keys are configured at all, every request is refused: a mistake in configuration must never
  leave the API open.
- **`lifespan()`**: runs at startup and shutdown: sets up logging, the database pool and the
  first model load.
- **`observe_requests()`**: a *middleware*: code that runs around every request. It assigns the
  request ID, measures the duration, updates the metrics and writes one log line per request.
- **Schemas**: `HouseFeatures` describes a valid request. Pydantic rejects anything else
  automatically, with a clear error (HTTP 422): values out of range, wrong types or **unknown
  fields** (`extra="forbid"`). The limits match the data validation rules from Step 5.
- **Endpoints**: `/predict` returns a price together with a `prediction_id` and the
  `model_version`, so every answer can be traced. If the database is unavailable, it still
  answers, and counts the failure so that an alert fires.

``` python
"""FastAPI prediction service. Serves the champion model from the MLflow registry."""

import logging
import secrets
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import mlflow.sklearn
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.routing import Match

from mlops import settings
from mlops.db import PredictionStore
from mlops.logging_config import configure_logging, request_id_var

logger = logging.getLogger("mlops.api")

HTTP_REQUESTS = Counter("http_requests", "HTTP requests handled", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter("predictions", "Predictions served", ["model_version"])
PREDICTED_PRICE = Histogram(
    "predicted_sale_price",
    "Predicted sale prices",
    buckets=(50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 400_000, 600_000),
)
PREDICTION_LOG_FAILURES = Counter(
    "prediction_log_failures", "Predictions that could not be written to the database"
)
MODEL_LOADED = Gauge("model_loaded", "1 when a champion model is loaded, otherwise 0")


# ---------- model loading ----------


@dataclass
class LoadedModel:
    model: Any
    version: str


def load_champion() -> LoadedModel:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    champion = client.get_model_version_by_alias(settings.MODEL_NAME, settings.CHAMPION_ALIAS)
    # Load by explicit version, so the served version is known exactly.
    model = mlflow.sklearn.load_model(f"models:/{settings.MODEL_NAME}/{champion.version}")
    return LoadedModel(model=model, version=str(champion.version))


class ModelHolder:
    """Holds the loaded model and retries loading at most every `retry_seconds`."""

    def __init__(self, loader: Callable[[], LoadedModel], retry_seconds: float = 30.0) -> None:
        self._loader = loader
        self._retry_seconds = retry_seconds
        self._lock = threading.Lock()
        self._last_attempt = float("-inf")
        self.loaded: LoadedModel | None = None

    def _due(self) -> bool:
        return time.monotonic() - self._last_attempt >= self._retry_seconds

    def get(self) -> LoadedModel | None:
        if self.loaded is None and self._due():
            with self._lock:
                if self.loaded is None and self._due():
                    self._last_attempt = time.monotonic()
                    try:
                        self.loaded = self._loader()
                        logger.info("Champion loaded", extra={"model_version": self.loaded.version})
                    except Exception:
                        logger.exception("Could not load the champion model")
        MODEL_LOADED.set(1 if self.loaded else 0)
        return self.loaded


model_holder = ModelHolder(load_champion)


# ---------- dependencies ----------


def get_model() -> LoadedModel:
    loaded = model_holder.get()
    if loaded is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No champion model is loaded.")
    return loaded


def get_store(request: Request) -> PredictionStore | None:
    return getattr(request.app.state, "store", None)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    valid_keys = settings.api_keys()
    if not valid_keys:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API keys are not configured.")
    provided = (x_api_key or "").encode()
    if not any(secrets.compare_digest(provided, key.encode()) for key in valid_keys):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ---------- app ----------


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)
    app.state.store = PredictionStore(settings.DATABASE_URL) if settings.DATABASE_URL else None
    if app.state.store is None:
        logger.warning("DATABASE_URL is not set; predictions will not be stored.")
    model_holder.get()
    yield
    if app.state.store is not None:
        app.state.store.close()


app = FastAPI(title="House Price API", version="0.2.0", lifespan=lifespan)


def route_template(request: Request) -> str:
    """Use the route pattern (e.g. /predict), not the raw URL, to keep metric labels bounded."""
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", "unknown")
    return "unmatched"


@app.middleware("http")
async def observe_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex
    token = request_id_var.set(request_id)
    route = route_template(request)
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration = time.perf_counter() - started
        HTTP_REQUESTS.labels(request.method, route, str(status_code)).inc()
        HTTP_LATENCY.labels(route).observe(duration)
        if route != "/metrics":
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "route": route,
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 1),
                },
            )
        request_id_var.reset(token)


# ---------- schemas ----------


class HouseFeatures(BaseModel):
    """Limits match the training validation rules in mlops/validation.py."""

    model_config = ConfigDict(extra="forbid")

    OverallQual: int = Field(ge=1, le=10, description="Overall quality, 1-10")
    GrLivArea: float = Field(gt=0, le=10_000, description="Above-ground living area (sq ft)")
    GarageCars: int = Field(ge=0, le=10, description="Garage capacity in cars")
    TotalBsmtSF: float = Field(ge=0, le=10_000, description="Basement area (sq ft)")
    YearBuilt: int = Field(ge=1800, le=2100)


class PredictionResponse(BaseModel):
    prediction_id: uuid.UUID
    predicted_sale_price: float
    model_version: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: uuid.UUID
    actual_sale_price: float = Field(gt=0, le=10_000_000)


# ---------- endpoints ----------


# Docstrings of the endpoints below appear in the interactive docs at /docs.


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is running. Used by Docker's healthcheck."""
    return {"status": "ok"}


@app.get("/ready")
def ready(model: Annotated[LoadedModel, Depends(get_model)]) -> dict[str, str]:
    """Readiness: a champion model is loaded, so predictions can be served (503 otherwise)."""
    return {"status": "ready", "model_version": model.version}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus metrics in text format. Prometheus reads this page every 15 seconds."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(
    features: HouseFeatures,
    model: Annotated[LoadedModel, Depends(get_model)],
    store: Annotated[PredictionStore | None, Depends(get_store)],
) -> PredictionResponse:
    """Predict the sale price of one house. Requires the X-API-Key header.

    The response includes a prediction_id: keep it to send the real price to /feedback later.
    """
    values = features.model_dump()
    frame = pd.DataFrame([values], columns=settings.FEATURES).astype(float)
    price = float(model.model.predict(frame)[0])
    prediction_id = uuid.uuid4()
    PREDICTIONS.labels(model.version).inc()
    PREDICTED_PRICE.observe(price)

    # Store the prediction for monitoring. If the database is down, still answer the client
    # but count the failure: the PredictionLogFailures alert fires on this counter.
    if store is None:
        PREDICTION_LOG_FAILURES.inc()
    else:
        try:
            store.log_prediction(prediction_id, model.version, values, price)
        except Exception:
            PREDICTION_LOG_FAILURES.inc()
            logger.exception("Could not store prediction", extra={"prediction_id": prediction_id})

    return PredictionResponse(
        prediction_id=prediction_id, predicted_sale_price=price, model_version=model.version
    )


@app.post("/feedback", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_api_key)])
def feedback(
    body: FeedbackRequest,
    store: Annotated[PredictionStore | None, Depends(get_store)],
) -> dict[str, str]:
    """Send the real sale price for an earlier prediction. Used to measure live accuracy."""
    if store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Feedback storage unavailable.")
    try:
        found = store.add_feedback(body.prediction_id, body.actual_sale_price)
    except Exception as exc:
        logger.exception("Could not store feedback")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Feedback storage unavailable."
        ) from exc
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown prediction_id.")
    return {"status": "recorded"}
```

> The API runs with **one worker process** per container. Prometheus metrics live in the
> process's memory; with several workers, each would count separately. To handle more traffic,
> run more containers.

------------------------------------------------------------------------

## Step 12: Monitoring checks: drift and real accuracy

**Goal:** find out when the model stops matching reality, before users do.

A model is trained on the past. Two things can go wrong later:

- **Data drift**: the houses people ask about change (bigger, newer, a different area) and no
  longer look like the training data. The model is then working outside what it knows.
- **Performance decay**: the model's predictions become less accurate, for example because
  the market changed.

Drift is a **warning sign**; it does not prove the model got worse. Only comparing predictions
with **real outcomes** proves that. That is why the API has a `/feedback` endpoint, and why there
are two checks:

| Check | Question it answers | Needs |
|---|---|---|
| `input_drift` | Are recent requests different from the training data? | ≥ 50 predictions in the last 7 days |
| `live_rmse` | Is the champion accurate on real sales? | ≥ 30 predictions with feedback in the last 30 days |

When there is not enough data, a check is recorded as **skipped**, never as a made-up pass.

### File: `src/mlops/monitoring.py`

- **`CheckResult`**: the outcome of one check: passed or not, the measured value and the limit.
- **`champion_info()`**: the champion's version and the RMSE it had on its test data. That RMSE
  is the reference for the accuracy check.
- **`check_drift()`**: compares recent requests with the training data using **Evidently**,
  saves an HTML report you can open in a browser, and fails if more than half of the features
  have drifted.
- **`check_quality()`**: joins predictions with their real prices, computes the live RMSE and
  fails if it is more than 1.5 times the test RMSE.

``` python
"""Scheduled monitoring checks: input drift and real-world accuracy."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from mlflow.tracking import MlflowClient

from mlops import settings
from mlops.db import PredictionStore


@dataclass
class CheckResult:
    """Outcome of one monitoring check. value is None when the check was skipped."""

    name: str
    passed: bool
    value: float | None
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


def champion_info() -> tuple[str, float]:
    """Return the champion version and the RMSE it had on its holdout set."""
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
    champion = client.get_model_version_by_alias(settings.MODEL_NAME, settings.CHAMPION_ALIAS)
    holdout_rmse = client.get_run(champion.run_id).data.metrics["rmse"]
    return str(champion.version), float(holdout_rmse)


def drifted_share(snapshot) -> float:
    """Share of columns Evidently flagged as drifted. The output format is version-specific."""
    for metric in snapshot.dict()["metrics"]:
        # The name field differs between 0.7.x releases ("metric_name" or "metric_id").
        name = str(metric.get("metric_name") or metric.get("metric_id") or "")
        kind = str((metric.get("config") or {}).get("type", ""))
        if name.startswith("DriftedColumnsCount") or kind.endswith(":DriftedColumnsCount"):
            return float(metric["value"]["share"])
    raise RuntimeError("DriftedColumnsCount not found in Evidently output; check the version.")


def check_drift(
    store: PredictionStore,
    days: int = settings.DRIFT_WINDOW_DAYS,
    min_rows: int = settings.DRIFT_MIN_ROWS,
) -> CheckResult:
    """Compare the inputs the API received recently with the training data."""
    threshold = settings.DRIFT_SHARE_THRESHOLD
    predictions = store.recent_predictions(days)
    # Too few rows give meaningless statistics: report "skipped" instead of a fake result.
    if len(predictions) < min_rows:
        reason = f"only {len(predictions)} predictions in {days} days (need {min_rows})"
        return CheckResult("input_drift", True, None, threshold, {"skipped": reason})

    # The API stored each request's features as JSON; turn them back into columns.
    current = pd.DataFrame(predictions["features"].tolist())[settings.FEATURES].astype(float)
    reference = pd.read_parquet(settings.PROCESSED_PATH, columns=settings.FEATURES)
    snapshot = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = settings.REPORTS_DIR / f"drift-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.html"
    snapshot.save_html(str(report_path))
    share = drifted_share(snapshot)
    details = {"rows": len(current), "report": str(report_path)}
    return CheckResult("input_drift", share <= threshold, share, threshold, details)


def check_quality(
    store: PredictionStore,
    model_version: str,
    holdout_rmse: float,
    days: int = settings.QUALITY_WINDOW_DAYS,
    min_rows: int = settings.QUALITY_MIN_ROWS,
) -> CheckResult:
    """Measure the champion's real accuracy: predictions joined with the true prices."""
    # The limit is relative to how well the model did on its holdout set when it was trained.
    threshold = holdout_rmse * settings.QUALITY_RMSE_RATIO
    labelled = store.labelled_predictions(days, model_version)
    if len(labelled) < min_rows:
        reason = f"only {len(labelled)} labelled predictions in {days} days (need {min_rows})"
        return CheckResult("live_rmse", True, None, threshold, {"skipped": reason})

    errors = labelled["predicted_price"] - labelled["actual_price"]
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    details = {"rows": len(labelled), "holdout_rmse": holdout_rmse}
    return CheckResult("live_rmse", rmse <= threshold, rmse, threshold, details)
```

------------------------------------------------------------------------

## Step 13: Orchestration with Dagster

**Goal:** run the training and monitoring steps automatically, in the right order, on a
schedule, with a history of every run.

You could run each script by hand, but then nothing guarantees the order, nothing runs at 6 a.m.
on Monday, and nobody can see what happened last week. An **orchestrator** does that. Dagster
models the pipeline as **assets**: things that are produced (a file, a model version), with
dependencies between them.

| Job | Assets (in order) | Schedule |
|---|---|---|
| `training_job` | raw data → validated data → Feast features → candidate model | Mondays 06:00 UTC |
| `monitoring_job` | drift check, accuracy check | Every day 07:00 UTC |

Automating training is safe here because training only produces a **candidate**. Going live
always needs a person (Step 9).

### File: `src/mlops/orchestration.py`

- **Training assets**: one function per step, each calling the code you wrote earlier.
  `deps=[...]` sets the order. If validation fails, the run stops and no model is trained.
- **`candidate_model`**: trains and records the gate's decision as metadata visible in the
  Dagster UI.
- **Monitoring assets**: run a check, save the result to Postgres (for Grafana), and **fail
  the run** if the check failed, so it shows up red in Dagster.
- **Jobs and schedules**: group the assets into the two jobs and schedule them.

``` python
"""Dagster assets, jobs and schedules for training and monitoring."""

from dagster import (
    AssetExecutionContext,
    DefaultScheduleStatus,
    Definitions,
    Failure,
    MaterializeResult,
    MetadataValue,
    ScheduleDefinition,
    asset,
    define_asset_job,
)
from mlflow.exceptions import MlflowException

from mlops import settings
from mlops.data import download_raw, file_sha256, prepare_data
from mlops.db import PredictionStore
from mlops.features import apply_definitions, build_feature_source
from mlops.monitoring import CheckResult, champion_info, check_drift, check_quality
from mlops.train import train_candidate

# ---------- training ----------
# Each @asset is one step that produces something (a file, a model version). `deps=[...]` says
# which asset must be built first, so Dagster always runs them in the right order. A step that
# raises an exception stops the run, and later steps do not run.


@asset(group_name="training", description="Raw Ames housing CSV from OpenML.")
def raw_housing_data() -> MaterializeResult:
    path = download_raw()
    return MaterializeResult(metadata={"path": str(path), "sha256": file_sha256(path)})


@asset(
    group_name="training",
    deps=[raw_housing_data],
    description="Validated and cleaned Parquet data (PySpark). Fails on bad data.",
)
def processed_housing_data() -> MaterializeResult:
    summary = prepare_data()
    return MaterializeResult(metadata=summary)


@asset(group_name="training", deps=[processed_housing_data], description="Feast feature view.")
def feast_feature_view() -> MaterializeResult:
    rows = build_feature_source()
    apply_definitions()
    return MaterializeResult(metadata={"rows": rows})


@asset(
    group_name="training",
    deps=[feast_feature_view],
    description="New registered model version, checked by the promotion gate.",
)
def candidate_model(context: AssetExecutionContext) -> MaterializeResult:
    result = train_candidate()
    if result.decision.passed:
        context.log.info(f"v{result.model_version} passed the gate and is the new challenger.")
    else:
        context.log.warning(
            f"v{result.model_version} failed the gate: {'; '.join(result.decision.reasons)}"
        )
    return MaterializeResult(
        metadata={
            "model_version": result.model_version,
            "gate_passed": result.decision.passed,
            "gate_reasons": MetadataValue.text("; ".join(result.decision.reasons) or "none"),
            "rmse": result.metrics["rmse"],
            "r2": result.metrics["r2"],
            "mlflow_run_id": result.run_id,
        }
    )


# ---------- monitoring ----------


def _record(store: PredictionStore, check: CheckResult, model_version: str) -> MaterializeResult:
    """Save the result to Postgres (for Grafana), then fail the run if the check failed."""
    store.record_check(
        check.name, model_version, check.value, check.threshold, check.passed, check.details
    )
    metadata = {
        "passed": check.passed,
        "value": check.value if check.value is not None else "skipped",
        "threshold": check.threshold,
        "details": MetadataValue.json(check.details),
    }
    if not check.passed:
        raise Failure(
            description=f"{check.name} failed: {check.value:.3f} > {check.threshold:.3f}",
            metadata=metadata,
        )
    return MaterializeResult(metadata=metadata)


def _run_check(check_name: str) -> MaterializeResult:
    try:
        model_version, holdout_rmse = champion_info()
    except MlflowException:
        return MaterializeResult(metadata={"skipped": "no champion model yet"})
    store = PredictionStore(settings.require_database_url())
    try:
        if check_name == "drift":
            check = check_drift(store)
        else:
            check = check_quality(store, model_version, holdout_rmse)
        return _record(store, check, model_version)
    finally:
        store.close()


@asset(group_name="monitoring", description="Evidently drift check on recent requests.")
def input_drift_check() -> MaterializeResult:
    return _run_check("drift")


@asset(group_name="monitoring", description="Live RMSE of the champion, from feedback.")
def live_quality_check() -> MaterializeResult:
    return _run_check("quality")


# ---------- jobs and schedules ----------

training_job = define_asset_job(
    "training_job",
    selection=[raw_housing_data, processed_housing_data, feast_feature_view, candidate_model],
)
monitoring_job = define_asset_job(
    "monitoring_job", selection=[input_drift_check, live_quality_check]
)

defs = Definitions(
    assets=[
        raw_housing_data,
        processed_housing_data,
        feast_feature_view,
        candidate_model,
        input_drift_check,
        live_quality_check,
    ],
    jobs=[training_job, monitoring_job],
    schedules=[
        ScheduleDefinition(
            job=training_job,
            cron_schedule="0 6 * * 1",
            default_status=DefaultScheduleStatus.RUNNING,
        ),
        ScheduleDefinition(
            job=monitoring_job,
            cron_schedule="0 7 * * *",
            default_status=DefaultScheduleStatus.RUNNING,
        ),
    ],
)
```

### File: `docker/dagster.yaml`

Dagster's own settings: keep the run history in Postgres (so it survives restarts), and queue
runs so that only one runs at a time (two Spark jobs at once would fight for memory).

``` yaml
# Dagster's own configuration (it lives in DAGSTER_HOME inside the container).

# Run history, logs and schedules are kept in Postgres, so they survive restarts.
storage:
  postgres:
    postgres_db:
      hostname: postgres
      port: 5432
      db_name: dagster
      username:
        env: DAGSTER_PG_USER
      password:
        env: DAGSTER_PG_PASSWORD

# Runs wait in a queue and execute one at a time, so two Spark jobs never compete for
# memory on a laptop.
run_coordinator:
  module: dagster.core.run_coordinator
  class: QueuedRunCoordinator
  config:
    max_concurrent_runs: 1

# Each run is started as a separate process inside the daemon container.
run_launcher:
  module: dagster.core.launcher
  class: DefaultRunLauncher

telemetry:
  enabled: false
```

### File: `docker/workspace.yaml`

Tells Dagster where to find the code: the `defs` object in `mlops.orchestration`.

``` yaml
# Tells Dagster where our code is: the `defs` object in mlops/orchestration.py.
load_from:
  - python_module:
      module_name: mlops.orchestration
      working_directory: /app
```

------------------------------------------------------------------------

## Step 14: Unit tests

**Goal:** prove the code works before running anything in Docker.

Unit tests check small pieces of code in isolation, in seconds, without starting databases or
servers. Where a test needs something heavy (a trained model, a database), it uses a small
**fake** object with the same methods. This is why the code is organised the way it is: pure
calculations in `evaluation.py`, database access behind one class, and the model behind a
dependency function that a test can replace.

### File: `tests/conftest.py`

`conftest.py` is a special pytest file for shared *fixtures*: setup code that tests can request
by name. This one starts one small local Spark session for the whole test run (starting Spark
takes a few seconds, so it is done once).

``` python
import pytest


@pytest.fixture(scope="session")
def spark():
    from mlops.data import create_spark

    session = create_spark(app_name="modern-mlops-tests")
    yield session
    session.stop()
```

### File: `tests/test_validation.py`

Builds small tables in memory: one valid, then one per rule broken (a missing column, too few
rows, an out-of-range value, too many missing values, duplicate IDs). Each test checks that the
right rule catches the problem. Testing the **failures** matters as much as testing success: a
validation that never fails is useless.

``` python
from pyspark.sql.types import DoubleType, StructField, StructType

from mlops.validation import RANGES, validate

COLUMNS = ["Id", *RANGES]
SCHEMA = StructType([StructField(name, DoubleType(), nullable=True) for name in COLUMNS])
VALID = {
    "OverallQual": 7.0,
    "GrLivArea": 1800.0,
    "GarageCars": 2.0,
    "TotalBsmtSF": 900.0,
    "YearBuilt": 2005.0,
    "SalePrice": 200_000.0,
    "YrSold": 2008.0,
    "MoSold": 6.0,
}


def make_rows(count: int = 20, **overrides) -> list[dict]:
    return [{**VALID, "Id": float(i), **overrides} for i in range(count)]


def test_valid_data_passes(spark):
    report = validate(spark.createDataFrame(make_rows(), schema=SCHEMA), min_rows=10)
    assert report.passed, report.failures


def test_missing_column_fails(spark):
    df = spark.createDataFrame(make_rows(), schema=SCHEMA).drop("GrLivArea")
    report = validate(df, min_rows=10)
    assert not report.passed
    assert "GrLivArea" in report.failures[0]


def test_too_few_rows_fails(spark):
    report = validate(spark.createDataFrame(make_rows(5), schema=SCHEMA), min_rows=10)
    assert not report.passed


def test_out_of_range_value_fails(spark):
    report = validate(
        spark.createDataFrame(make_rows(OverallQual=42.0), schema=SCHEMA), min_rows=10
    )
    assert any("OverallQual" in failure for failure in report.failures)


def test_too_many_missing_values_fails(spark):
    rows = make_rows()
    for row in rows[:5]:  # 25% missing, limit is 5%
        row["GrLivArea"] = None
    report = validate(spark.createDataFrame(rows, schema=SCHEMA), min_rows=10)
    assert any("GrLivArea" in failure for failure in report.failures)


def test_duplicate_ids_fail(spark):
    report = validate(spark.createDataFrame(make_rows(Id=1.0), schema=SCHEMA), min_rows=10)
    assert report.duplicate_ids == 19
    assert not report.passed
```

### File: `tests/test_evaluation.py`

Uses a fake linear model whose behaviour we control, to check that the behaviour checks catch a
model that prices bigger houses lower, and that the gate accepts and rejects exactly when it
should.

``` python
import numpy as np
import pandas as pd

from mlops.evaluation import behaviour_checks, promotion_gate, regression_metrics


class LinearFakeModel:
    def __init__(self, area_effect: float) -> None:
        self.area_effect = area_effect

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        prices = 50_000 + self.area_effect * frame["GrLivArea"] + 10_000 * frame["OverallQual"]
        return prices.to_numpy()


def test_metrics_for_perfect_predictions():
    metrics = regression_metrics([100.0, 200.0, 300.0], [100.0, 200.0, 300.0])
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1


def test_sensible_model_passes_behaviour_checks():
    assert behaviour_checks(LinearFakeModel(area_effect=100)) == []


def test_model_pricing_bigger_houses_lower_fails():
    failures = behaviour_checks(LinearFakeModel(area_effect=-10))
    assert any("GrLivArea" in failure for failure in failures)


def test_gate_passes_first_model_above_floor():
    decision = promotion_gate({"r2": 0.8, "rmse": 30_000}, None, [], min_r2=0.7)
    assert decision.passed


def test_gate_rejects_low_r2():
    decision = promotion_gate({"r2": 0.5, "rmse": 30_000}, None, [], min_r2=0.7)
    assert not decision.passed


def test_gate_rejects_model_worse_than_champion():
    decision = promotion_gate(
        {"r2": 0.8, "rmse": 35_000}, {"r2": 0.82, "rmse": 30_000}, [], min_r2=0.7
    )
    assert not decision.passed
    assert "champion" in decision.reasons[0]


def test_gate_includes_behaviour_failures():
    decision = promotion_gate({"r2": 0.9, "rmse": 20_000}, None, ["bad"], min_r2=0.7)
    assert decision.reasons == ["bad"]
```

### File: `tests/test_api.py`

Replaces the model and the database with fakes (`dependency_overrides`), then calls the API like
a client would. It checks the security rules (no key, a wrong key, no keys configured), the input
validation, a successful prediction, the behaviour when the database is down, and feedback.

``` python
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops import api

HOUSE = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "TotalBsmtSF": 900,
    "YearBuilt": 2005,
}
HEADERS = {"X-API-Key": "test-key"}


class FakeModel:
    def predict(self, frame):
        return np.full(len(frame), 250_000.0)


class FakeStore:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.predictions: dict[uuid.UUID, float] = {}
        self.feedback: dict[uuid.UUID, float] = {}

    def log_prediction(self, prediction_id, model_version, features, predicted_price):
        if self.fail:
            raise RuntimeError("database is down")
        self.predictions[prediction_id] = predicted_price

    def add_feedback(self, prediction_id, actual_price) -> bool:
        if prediction_id not in self.predictions:
            return False
        self.feedback[prediction_id] = actual_price
        return True


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.setenv("API_KEYS", "test-key")
    api.app.dependency_overrides[api.get_model] = lambda: api.LoadedModel(FakeModel(), "7")
    api.app.dependency_overrides[api.get_store] = lambda: store
    # No "with" block: the lifespan (MLflow, Postgres) does not run in unit tests.
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_model_version(client):
    assert client.get("/ready").json()["model_version"] == "7"


def test_predict_requires_api_key(client):
    assert client.post("/predict", json=HOUSE).status_code == 401


def test_predict_rejects_wrong_api_key(client):
    response = client.post("/predict", json=HOUSE, headers={"X-API-Key": "nope"})
    assert response.status_code == 401


def test_no_configured_keys_rejects_everything(client, monkeypatch):
    monkeypatch.delenv("API_KEYS")
    assert client.post("/predict", json=HOUSE, headers=HEADERS).status_code == 503


def test_predict_rejects_out_of_range_input(client):
    response = client.post("/predict", json={**HOUSE, "OverallQual": 100}, headers=HEADERS)
    assert response.status_code == 422


def test_predict_rejects_unknown_fields(client):
    response = client.post("/predict", json={**HOUSE, "Pool": 1}, headers=HEADERS)
    assert response.status_code == 422


def test_predict_returns_price_and_stores_it(client, store):
    response = client.post("/predict", json=HOUSE, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_sale_price"] == 250_000.0
    assert body["model_version"] == "7"
    assert uuid.UUID(body["prediction_id"]) in store.predictions
    assert "X-Request-ID" in response.headers


def test_predict_still_answers_when_storage_fails(client):
    api.app.dependency_overrides[api.get_store] = lambda: FakeStore(fail=True)
    response = client.post("/predict", json=HOUSE, headers=HEADERS)
    assert response.status_code == 200


def test_feedback_for_known_prediction(client, store):
    prediction_id = client.post("/predict", json=HOUSE, headers=HEADERS).json()["prediction_id"]
    body = {"prediction_id": prediction_id, "actual_sale_price": 240_000}
    assert client.post("/feedback", json=body, headers=HEADERS).status_code == 201
    assert store.feedback[uuid.UUID(prediction_id)] == 240_000


def test_feedback_for_unknown_prediction(client):
    body = {"prediction_id": str(uuid.uuid4()), "actual_sale_price": 240_000}
    assert client.post("/feedback", json=body, headers=HEADERS).status_code == 404
```

### Check

``` bash
uv run ruff format .        # format every file consistently
uv run ruff check .         # lint: unused imports, style problems, likely bugs
uv run pytest -q            # run the tests
```

Expected: `All checks passed!` and **24 passed**. Fix any failure before continuing.

------------------------------------------------------------------------

## Step 15: Metrics, alerts and dashboards

**Goal:** see what the API is doing, and be told automatically when something is wrong.

- **Prometheus** collects the API's metrics every 15 seconds, stores them over time, and
  evaluates **alert rules**: conditions that should never be true for long.
- **Grafana** draws dashboards from Prometheus and from the monitoring results in Postgres.

Everything is configured with files, including the dashboard. That way the configuration is
versioned in Git, reviewed like code, and recreated identically on any machine.

### File: `monitoring/prometheus.yml`

Where to collect metrics from (the API, at its Compose service name `api`) and which rule file
to load.

``` yaml
# Prometheus pulls ("scrapes") metrics from the API every 15 seconds and evaluates the
# alert rules just as often.
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/alert_rules.yml

scrape_configs:
  - job_name: house-price-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]  # the Compose service name, not localhost
```

### File: `monitoring/alert_rules.yml`

Five alerts. Each has an expression, a duration (`for:`) that avoids alerting on a short blip, a
severity and a summary:

- **ApiDown**: Prometheus cannot reach the API.
- **ModelNotLoaded**: the API runs but has no model.
- **HighErrorRate**: more than 5% of predictions fail.
- **HighLatency**: the slowest 5% of predictions take more than 200 ms.
- **PredictionLogFailures**: predictions are not being stored, so monitoring data is lost.

``` yaml
groups:
  - name: house-price-api
    rules:
      - alert: ApiDown
        expr: up{job="house-price-api"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Prometheus cannot reach the prediction API."

      - alert: ModelNotLoaded
        expr: max(model_loaded) == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "The API is running but has no champion model."

      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{route="/predict", status=~"5.."}[5m]))
            / sum(rate(http_requests_total{route="/predict"}[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "More than 5% of /predict requests fail."

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum by (le) (rate(http_request_duration_seconds_bucket{route="/predict"}[5m]))
          ) > 0.2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "p95 /predict latency is above the 200 ms SLO."

      - alert: PredictionLogFailures
        expr: increase(prediction_log_failures_total[10m]) > 0
        labels:
          severity: warning
        annotations:
          summary: "Predictions are not being stored; monitoring data is being lost."
```

### File: `monitoring/grafana/provisioning/datasources/datasources.yml`

Connects Grafana to Prometheus and to the `monitoring` database. Grafana uses a **read-only**
database user: a dashboard should never be able to change data.

``` yaml
apiVersion: 1

datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

  - name: Monitoring DB
    uid: monitoring-db
    type: grafana-postgresql-datasource
    url: postgres:5432
    user: grafana_reader
    jsonData:
      database: monitoring
      sslmode: disable
      postgresVersion: 1600
    secureJsonData:
      password: $GRAFANA_DB_PASSWORD
```

### File: `monitoring/grafana/provisioning/dashboards/dashboards.yml`

Tells Grafana to load dashboards from a folder at startup.

``` yaml
apiVersion: 1

providers:
  - name: MLOps dashboards
    orgId: 1
    folder: MLOps
    type: file
    disableDeletion: true
    allowUiUpdates: false
    options:
      path: /etc/grafana/dashboards
```

### File: `monitoring/grafana/dashboards/house-price-api.json`

The dashboard itself: requests by status, latency, whether a model is loaded, storage failures,
predictions per model version, and a table of the latest monitoring checks. To change it, edit
it in the Grafana UI, export the JSON, and replace this file.

``` json
{
  "uid": "house-price-api",
  "title": "House Price API",
  "schemaVersion": 39,
  "version": 1,
  "refresh": "30s",
  "time": { "from": "now-6h", "to": "now" },
  "panels": [
    {
      "id": 1,
      "type": "timeseries",
      "title": "Requests per second by status",
      "gridPos": { "x": 0, "y": 0, "w": 12, "h": 8 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (status) (rate(http_requests_total{route=\"/predict\"}[5m]))",
          "legendFormat": "{{status}}"
        }
      ]
    },
    {
      "id": 2,
      "type": "timeseries",
      "title": "p95 /predict latency (seconds)",
      "gridPos": { "x": 12, "y": 0, "w": 12, "h": 8 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [
        {
          "refId": "A",
          "expr": "histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{route=\"/predict\"}[5m])))",
          "legendFormat": "p95"
        }
      ]
    },
    {
      "id": 3,
      "type": "stat",
      "title": "Model loaded",
      "gridPos": { "x": 0, "y": 8, "w": 6, "h": 4 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "refId": "A", "expr": "min(model_loaded)" }]
    },
    {
      "id": 4,
      "type": "stat",
      "title": "Prediction log failures (1h)",
      "gridPos": { "x": 6, "y": 8, "w": 6, "h": 4 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [{ "refId": "A", "expr": "sum(increase(prediction_log_failures_total[1h]))" }]
    },
    {
      "id": 5,
      "type": "timeseries",
      "title": "Predictions per model version",
      "gridPos": { "x": 12, "y": 8, "w": 12, "h": 4 },
      "datasource": { "type": "prometheus", "uid": "prometheus" },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (model_version) (rate(predictions_total[5m]))",
          "legendFormat": "v{{model_version}}"
        }
      ]
    },
    {
      "id": 6,
      "type": "table",
      "title": "Latest monitoring checks",
      "gridPos": { "x": 0, "y": 12, "w": 24, "h": 8 },
      "datasource": { "type": "grafana-postgresql-datasource", "uid": "monitoring-db" },
      "targets": [
        {
          "refId": "A",
          "format": "table",
          "rawQuery": true,
          "editorMode": "code",
          "rawSql": "SELECT checked_at, check_name, model_version, value, threshold, passed FROM monitoring_results ORDER BY checked_at DESC LIMIT 20"
        }
      ]
    }
  ]
}
```

------------------------------------------------------------------------

## Step 16: Container images

**Goal:** package the code so it runs the same way on any machine.

A **Docker image** contains the operating system libraries, Python, the packages and the code.
A **container** is a running image. We build two images, each with only what it needs:

| Image | Contents | Used by |
|---|---|---|
| `api` | Python and the API's packages. No Java, no Spark. | `api` |
| `pipeline` | Java, PySpark, Feast, Evidently, Dagster, MLflow server | `mlflow`, `dagster-webserver`, `dagster-daemon` |

A smaller image starts faster, downloads faster and contains fewer components that could have
security problems. Both images:

- install packages from `uv.lock` (`--frozen`), so they get exactly the tested versions,
- install the packages **before** copying the code, so that changing your code does not
  reinstall every package (Docker reuses the cached layer),
- run as a **normal user**, not `root`.

### File: `docker/api.Dockerfile`

A **multi-stage** build: the first stage installs everything, the second stage copies only the
finished virtual environment into a clean image. The `HEALTHCHECK` lets Docker know when the
API stops responding.

``` dockerfile
# syntax=docker/dockerfile:1
ARG UV_VERSION=0.8.0
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# Stage 1 (builder): install dependencies into /app/.venv. Build tools stay in this stage.
FROM python:3.11-slim-bookworm AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: this layer is cached until the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Stage 2 (runtime): a clean image with only the virtual environment copied in.
FROM python:3.11-slim-bookworm
# Run as a normal user, not root: limits the damage if the app is compromised.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app
USER app
EXPOSE 8000
# Docker marks the container unhealthy if /health stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["uvicorn", "mlops.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
```

### File: `docker/pipeline.Dockerfile`

Adds Java (needed by Spark) and the project's configuration files for Feast and Dagster.

``` dockerfile
# syntax=docker/dockerfile:1
ARG UV_VERSION=0.8.0
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.11-slim-bookworm
# Java is required by Spark; procps provides `ps`, which Spark's start scripts use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (cached until the lockfile changes), then our code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group pipeline --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --group pipeline --no-editable

COPY feature_repo/feature_store.yaml ./feature_repo/feature_store.yaml
COPY docker/workspace.yaml ./workspace.yaml
COPY docker/dagster.yaml /opt/dagster/home/dagster.yaml

# A normal user that owns only the folders it must write to.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/artifacts /app/feature_repo/data \
       /opt/dagster/home/storage /mlflow/artifacts \
    && chown -R app:app /app/data /app/artifacts /app/feature_repo /opt/dagster /mlflow

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    DAGSTER_HOME=/opt/dagster/home
USER app
```

------------------------------------------------------------------------

## Step 17: Docker Compose: the whole stack

**Goal:** start every service with one command.

**Docker Compose** describes several containers and how they connect. Inside Compose, services
reach each other **by name**: the API talks to `http://mlflow:5000`, and Prometheus collects
from `api:8000`. Only your own machine can reach the services, because every port is bound to
`127.0.0.1`.

### File: `docker/postgres-init.sh`

Runs **once**, when the database is first created. It creates one user and one database per
service (MLflow, Dagster, the app), plus a read-only user for Grafana.

``` sh
#!/bin/sh
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<EOSQL
CREATE USER mlflow WITH PASSWORD '${MLFLOW_DB_PASSWORD}';
CREATE DATABASE mlflow OWNER mlflow;

CREATE USER dagster WITH PASSWORD '${DAGSTER_DB_PASSWORD}';
CREATE DATABASE dagster OWNER dagster;

CREATE USER app WITH PASSWORD '${APP_DB_PASSWORD}';
CREATE DATABASE monitoring OWNER app;

CREATE USER grafana_reader WITH PASSWORD '${GRAFANA_DB_PASSWORD}';
GRANT CONNECT ON DATABASE monitoring TO grafana_reader;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname monitoring <<EOSQL
GRANT USAGE ON SCHEMA public TO grafana_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE app IN SCHEMA public GRANT SELECT ON TABLES TO grafana_reader;
EOSQL
```

### File: `compose.yaml`

Key parts:

- **`x-pipeline-image`** and **`x-dagster`**: shared settings, defined once and reused by
  several services (the `<<: *name` lines).
- **`${VARIABLE:?...}`**: values read from `.env`. The `:?` makes Compose **refuse to start**
  if a value is missing, instead of silently using an empty password.
- **`depends_on` with `service_healthy`**: a service only starts once the services it needs
  are actually ready, not just started.
- **`healthcheck`**: how Docker tests that a service works.
- **`deploy.resources.limits.memory`**: a memory limit per service, so one service cannot take
  all of your laptop's memory.
- **`volumes`**: named volumes keep the data (database, models, dashboards) when containers are
  recreated.

``` yaml
name: modern-mlops

x-pipeline-image: &pipeline-image
  build:
    context: .
    dockerfile: docker/pipeline.Dockerfile
  image: modern-mlops-pipeline:local

x-dagster: &dagster
  <<: *pipeline-image
  environment:
    PROJECT_ROOT: /app
    MLFLOW_TRACKING_URI: http://mlflow:5000
    DATABASE_URL: postgresql://app:${APP_DB_PASSWORD:?set in .env}@postgres:5432/monitoring
    DAGSTER_PG_USER: dagster
    DAGSTER_PG_PASSWORD: ${DAGSTER_DB_PASSWORD:?set in .env}
    SPARK_DRIVER_MEMORY: 1g  # ample for this dataset (1,460 rows)
    RIDGE_ALPHA: ${RIDGE_ALPHA:-1.0}
  volumes:
    - ./data:/app/data
    - ./artifacts:/app/artifacts
    - dagster-storage:/opt/dagster/home/storage
  depends_on:
    postgres:
      condition: service_healthy
    mlflow:
      condition: service_healthy
  restart: unless-stopped

services:
  postgres:
    image: postgres:16.4-alpine
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?set in .env}
      MLFLOW_DB_PASSWORD: ${MLFLOW_DB_PASSWORD:?set in .env}
      DAGSTER_DB_PASSWORD: ${DAGSTER_DB_PASSWORD:?set in .env}
      APP_DB_PASSWORD: ${APP_DB_PASSWORD:?set in .env}
      GRAFANA_DB_PASSWORD: ${GRAFANA_DB_PASSWORD:?set in .env}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./docker/postgres-init.sh:/docker-entrypoint-initdb.d/init.sh:ro
    ports:
      - "127.0.0.1:${POSTGRES_HOST_PORT:-5432}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 3s
      retries: 20
    deploy:
      resources:
        limits:
          memory: 512m
    restart: unless-stopped

  mlflow:
    <<: *pipeline-image
    command: >
      mlflow server
      --backend-store-uri postgresql+psycopg2://mlflow:${MLFLOW_DB_PASSWORD}@postgres:5432/mlflow
      --artifacts-destination /mlflow/artifacts
      --serve-artifacts
      --host 0.0.0.0
      --port 5000
      --workers 1
    volumes:
      - mlflow-artifacts:/mlflow/artifacts
    ports:
      - "127.0.0.1:5000:5000"
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 20s
    deploy:
      resources:
        limits:
          memory: 1g
    restart: unless-stopped

  dagster-webserver:
    <<: *dagster
    command: ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000", "-w", "/app/workspace.yaml"]
    ports:
      - "127.0.0.1:3000:3000"
    deploy:
      resources:
        limits:
          memory: 1g

  dagster-daemon:
    <<: *dagster
    command: ["dagster-daemon", "run", "-w", "/app/workspace.yaml"]
    deploy:
      resources:
        limits:
          memory: 3g  # training runs (Spark) execute inside the daemon container

  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    image: modern-mlops-api:local
    environment:
      PROJECT_ROOT: /app
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MLFLOW_HTTP_REQUEST_MAX_RETRIES: "2"
      MLFLOW_HTTP_REQUEST_TIMEOUT: "10"
      DATABASE_URL: postgresql://app:${APP_DB_PASSWORD:?set in .env}@postgres:5432/monitoring
      API_KEYS: ${API_KEYS:?set in .env}
    ports:
      - "127.0.0.1:8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      mlflow:
        condition: service_healthy
    deploy:
      resources:
        limits:
          memory: 1g
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.54.1
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.retention.time=15d
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "127.0.0.1:9090:9090"
    depends_on:
      - api
    deploy:
      resources:
        limits:
          memory: 512m
    restart: unless-stopped

  grafana:
    image: grafana/grafana-oss:11.2.0
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:?set in .env}
      GF_USERS_ALLOW_SIGN_UP: "false"
      GF_ANALYTICS_REPORTING_ENABLED: "false"
      GRAFANA_DB_PASSWORD: ${GRAFANA_DB_PASSWORD:?set in .env}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/etc/grafana/dashboards:ro
    ports:
      - "127.0.0.1:3001:3000"
    depends_on:
      - prometheus
      - postgres
    deploy:
      resources:
        limits:
          memory: 512m
    restart: unless-stopped

volumes:
  postgres-data:
  mlflow-artifacts:
  dagster-storage:
  prometheus-data:
  grafana-data:
```

### Check

``` bash
docker compose config --quiet
```

No output means the file is valid and every variable from `.env` is set.

------------------------------------------------------------------------

## Step 18: Helper scripts

**Goal:** tools to test the running system and to simulate real usage.

### File: `scripts/smoke_test.py`

A **smoke test** is a quick check run after every deployment: "does the basic path work?" It
calls each endpoint once and checks the security rules. It exits with an error code if anything
fails, so it can be used in automation.

``` python
"""Smoke test against a running stack.

Usage: uv run python scripts/smoke_test.py --api-key KEY [--base-url http://127.0.0.1:8000]
"""

import argparse
import sys

import httpx

SAMPLE = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "TotalBsmtSF": 900,
    "YearBuilt": 2005,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    results: list[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail if not ok else ''}")
        results.append(ok)

    headers = {"X-API-Key": args.api_key}
    with httpx.Client(base_url=args.base_url, timeout=10) as client:
        check("health", client.get("/health").status_code == 200)

        response = client.get("/ready")
        check("ready (champion loaded)", response.status_code == 200, response.text)

        response = client.post("/predict", json=SAMPLE, headers=headers)
        check("predict", response.status_code == 200, response.text)
        if response.status_code == 200:
            prediction_id = response.json()["prediction_id"]
            body = {"prediction_id": prediction_id, "actual_sale_price": 210_000}
            response = client.post("/feedback", json=body, headers=headers)
            check("feedback", response.status_code == 201, response.text)

        response = client.post("/predict", json=SAMPLE, headers={"X-API-Key": "wrong"})
        check("rejects a wrong API key", response.status_code == 401)

        response = client.post("/predict", json={**SAMPLE, "OverallQual": 100}, headers=headers)
        check("rejects invalid input", response.status_code == 422)

        check("metrics exposed", "http_requests_total" in client.get("/metrics").text)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
```

### File: `scripts/replay_holdout.py`

Simulates real traffic: it sends the most recent real sales to the API, then sends each one's
true price to `/feedback`, exactly as a real client would once the sale is known. The
`--price-shift` option multiplies the true prices, to simulate a market change in the exercises.

``` python
"""Replay recent real sales through the API, then send their true prices as feedback.

Usage: uv run python scripts/replay_holdout.py --api-key KEY [--rows 200] [--price-shift 1.0]
"""

import argparse

import httpx
import pandas as pd

from mlops import settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument(
        "--price-shift",
        type=float,
        default=1.0,
        help="Multiply real prices by this factor to simulate a market change.",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(settings.PROCESSED_PATH)
    recent = frame.sort_values(settings.TIMESTAMP).tail(args.rows)

    headers = {"X-API-Key": args.api_key}
    with httpx.Client(base_url=args.base_url, timeout=10, headers=headers) as client:
        for record in recent.to_dict("records"):
            payload = {name: record[name] for name in settings.FEATURES}
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            feedback = {
                "prediction_id": response.json()["prediction_id"],
                "actual_sale_price": record[settings.TARGET] * args.price_shift,
            }
            client.post("/feedback", json=feedback).raise_for_status()
    print(f"Sent {len(recent)} predictions with feedback (price shift {args.price_shift}).")


if __name__ == "__main__":
    main()
```

### File: `loadtest/locustfile.py`

A **load test** sends many requests at once to measure speed under pressure. It uses real houses
from the dataset, so it does not distort the drift check.

``` python
"""Load test.

Usage (macOS/Linux):
  API_KEY=... uv run locust -f loadtest/locustfile.py --headless -u 20 -r 5 -t 2m \
    --host http://127.0.0.1:8000
PowerShell: $env:API_KEY="..."; then the same command on one line.
"""

import os
import random

import pandas as pd
from locust import HttpUser, between, task

from mlops import settings

HOUSES = pd.read_parquet(settings.PROCESSED_PATH, columns=settings.FEATURES).to_dict("records")


class PredictionUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        self.client.headers["X-API-Key"] = os.environ["API_KEY"]

    @task
    def predict(self) -> None:
        self.client.post("/predict", json=random.choice(HOUSES))
```

------------------------------------------------------------------------

## Step 19: First run

**Goal:** start everything, train the first model, release it and see it work.

**1. Start the stack**

``` bash
docker compose up --build -d
docker compose ps
```

The first build takes several minutes. Then every service should be `running`, and `postgres`,
`mlflow` and `api` should become `healthy`. Open:

| Service | URL |
|---|---|
| API docs | <http://localhost:8000/docs> |
| MLflow | <http://localhost:5000> |
| Dagster | <http://localhost:3000> |
| Prometheus | <http://localhost:9090> |
| Grafana | <http://localhost:3001> (user `admin`, password `GRAFANA_ADMIN_PASSWORD` from `.env`) |

Open <http://localhost:8000/ready>: it answers **503 "No champion model is loaded"**. That is
correct: no model has been trained yet.

**2. Train the first model**

In Dagster, open **Jobs → training_job** and launch it (**Materialize all**). The first run
downloads the data and starts Spark: allow a few minutes. All four steps should turn green.

In MLflow, look at what was recorded: the run's parameters and metrics, the `model_card.md` file
(read it!), and **model version 1** with the alias `challenger`.

**3. Release it**

``` bash
docker compose exec api mlops-model status
docker compose exec api mlops-model promote
docker compose restart api
```

Open <http://localhost:8000/ready> again: it now shows `"model_version": "1"`.

**4. Smoke test**

Copy your API key from the `API_KEYS` line in `.env`, then:

``` bash
uv run python scripts/smoke_test.py --api-key "<your key>"
```

Every line must say `PASS`.

**5. Try the API yourself**

In <http://localhost:8000/docs>, click **Authorize**, enter your key, open `POST /predict` →
**Try it out**, and send:

``` json
{"OverallQual": 7, "GrLivArea": 1800, "GarageCars": 2, "TotalBsmtSF": 900, "YearBuilt": 2005}
```

Note the `prediction_id` and `model_version` in the answer. Then try a value out of range
(`"OverallQual": 42`) and look at the 422 error.

**6. Send realistic traffic and run monitoring**

``` bash
uv run python scripts/replay_holdout.py --api-key "<your key>" --rows 200
```

In Dagster, launch **monitoring_job**. Both checks should pass. In Grafana, open
**Dashboards → MLOps → House Price API**: you should see the requests and the
"Latest monitoring checks" table.

### Check

- `/ready` shows a model version.
- The smoke test passes.
- `monitoring_job` succeeded, and Grafana shows its results.
- In Prometheus (**Status → Targets**), `house-price-api` is **UP**.

------------------------------------------------------------------------

## Step 20: Exercises: prove the safety mechanisms work

A safety mechanism you have never seen work is a safety mechanism you cannot trust. Do each
exercise and **write down what you observe**: what happened, where you saw it, and why.

1. **Accuracy alert.** Simulate a market where real prices are 60% higher than the model
   expects:
   `uv run python scripts/replay_holdout.py --api-key "<key>" --rows 200 --price-shift 1.6`.
   Run `monitoring_job`. The `live_quality_check` must fail (red run), and Grafana's table must
   show `passed = false`.
2. **Gate rejection.** Set `RIDGE_ALPHA=1000` in `.env` (a much weaker model), run
   `docker compose up -d` to apply it, then run `training_job`. The new version must fail the gate
   ("worse than the champion") and must **not** become challenger. Try to promote it anyway with
   `docker compose exec api mlops-model promote --version <n>`: the command must refuse. Set
   `RIDGE_ALPHA` back to `1.0` and run `docker compose up -d` again.
3. **Promotion and rollback.** Run `training_job` again (it passes: equal RMSE is not worse).
   Promote the new version, restart the API, and check `/ready`. Then run
   `docker compose exec api mlops-model rollback`, restart, and confirm `/ready` shows the old
   version again.
4. **Database outage.** Run `docker compose stop postgres`, then send a prediction: it must still
   return a price. Within about 10 minutes, the `PredictionLogFailures` alert appears at
   <http://localhost:9090/alerts>. Start Postgres again with `docker compose start postgres`.
5. **Bad data.** Edit `data/raw/housing.csv` and set a few `OverallQual` values to `42`. Run
   `training_job`: it must stop at `processed_housing_data`, and
   `artifacts/reports/validation.json` must explain why. To restore the data, delete
   `data/raw/housing.csv` and run the job again: it downloads the original file.

### Stopping the stack

``` bash
docker compose down        # stops everything and KEEPS all data
docker compose down -v     # also DELETES all data: database, models, dashboards
```

------------------------------------------------------------------------

## Step 21: CI/CD with GitHub Actions

**Goal:** every change is checked automatically, and every release is a versioned image.

**CI** (continuous integration) runs the checks on every push, so a broken change is caught
before it is merged. **CD** (continuous delivery) turns a release into a versioned artifact.

### File: `.github/workflows/ci.yaml`

Two jobs:

- **checks**: installs the exact locked dependencies, then runs the linter, the formatting
  check, the tests, and `pip-audit`, which looks for packages with known security
  vulnerabilities.
- **images**: builds both images and scans them with **Trivy**. The API image must have no
  fixable high or critical vulnerabilities; the pipeline image is scanned and reported.

``` yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"

      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - run: uv python install 3.11
      - run: uv sync --frozen --all-groups
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest -q

      - name: Audit dependencies for known vulnerabilities
        run: |
          uv export --frozen --all-groups --no-hashes --no-emit-project \
            --format requirements-txt > requirements-audit.txt
          uvx pip-audit -r requirements-audit.txt --disable-pip --no-deps

  images:
    needs: checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build images
        run: |
          docker build -f docker/api.Dockerfile -t modern-mlops-api:ci .
          docker build -f docker/pipeline.Dockerfile -t modern-mlops-pipeline:ci .

      - name: Scan the API image (fails on fixable HIGH/CRITICAL issues)
        run: |
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy:0.56.2 image --severity HIGH,CRITICAL --ignore-unfixed \
            --exit-code 1 modern-mlops-api:ci

      - name: Scan the pipeline image (report only)
        run: |
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy:0.56.2 image --severity HIGH,CRITICAL --ignore-unfixed \
            --exit-code 0 modern-mlops-pipeline:ci
```

### File: `.github/workflows/release.yaml`

When you push a version tag such as `v0.2.0`, both images are built and published to the GitHub
Container Registry. A release is then a fixed, named image, not "whatever was on my laptop".

``` yaml
name: Release

on:
  push:
    tags: ["v*"]

permissions:
  contents: read
  packages: write

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        run: |
          for target in api pipeline; do
            image="ghcr.io/${GITHUB_REPOSITORY,,}-${target}:${GITHUB_REF_NAME}"
            docker build -f "docker/${target}.Dockerfile" -t "$image" .
            docker push "$image"
          done
```

### File: `.github/dependabot.yml`

Dependabot checks every week for newer versions of the packages, images and actions, and opens a
pull request for each update. CI then tests it before you merge.

``` yaml
version: 2
updates:
  - package-ecosystem: uv
    directory: /
    schedule:
      interval: weekly
  - package-ecosystem: docker
    directory: /docker
    schedule:
      interval: weekly
  - package-ecosystem: github-actions
    directory: /
    schedule:
      interval: weekly
```

------------------------------------------------------------------------

## Step 22: The runbook

**Goal:** written instructions that let anyone operate the system without reading the code.

A **runbook** answers "what do I do when...?". It contains the service level objectives (SLOs:
the targets the service must meet), the release and rollback procedures, and what to do for each
alert. Write it while you still remember how everything works; you will thank yourself later.

### File: `docs/RUNBOOK.md`

```` markdown
# House Price API: Runbook

## Service level objectives (measured over 7 days)

| SLO | Target | Where to see it |
|---|---|---|
| Availability | ≥ 99% of /predict requests return non-5xx | Grafana: requests by status |
| Latency | p95 /predict < 200 ms | Grafana: p95 latency; HighLatency alert |
| Monitoring freshness | monitoring_job succeeded in the last 26 h | Dagster: runs |
| Model quality | live RMSE ≤ 1.5 × holdout RMSE | Grafana: latest monitoring checks |

## Daily operations

- Start: `docker compose up -d`. Stop: `docker compose down` (never add `-v` unless you mean to delete all data)
- Status: `docker compose ps`, `docker compose exec api mlops-model status`
- Logs: `docker compose logs -f api` (JSON; search by `request_id`)

## Release a new model

1. Run `training_job` in Dagster (or wait for Monday's scheduled run).
2. In MLflow, read the new version's `model_card.md` and gate tags.
3. `docker compose exec api mlops-model promote`
4. `docker compose restart api`
5. `uv run python scripts/smoke_test.py --api-key <key>`: every line must be PASS.
6. Watch the dashboard for 15 minutes.

## Roll back a model

1. `docker compose exec api mlops-model rollback` (or `--version <n>`)
2. `docker compose restart api`
3. Smoke test. Confirm `/ready` shows the expected version.

## Alert → action

| Alert / signal | First checks | Typical fix |
|---|---|---|
| ApiDown | `docker compose ps`, `docker compose logs api` | Restart api; check memory limits |
| ModelNotLoaded | `docker compose logs api`; is MLflow healthy? does a champion exist? | Fix MLflow, promote a version, restart api |
| HighErrorRate | API logs for exceptions (search `"level": "ERROR"`) | Roll back the model if it started after a release |
| HighLatency | CPU/memory (`docker stats`); is a training run active? | Wait for training to finish; investigate slow requests |
| PredictionLogFailures | `docker compose ps postgres`, Postgres logs | Restart Postgres; check disk space |
| input_drift failed | Open the HTML drift report in `artifacts/reports/` | Investigate which features drifted; retrain if the change is real |
| live_rmse failed | Compare predictions and feedback in Postgres | Retrain on recent data; roll back if a recent release caused it |
| Validation failed in training_job | `artifacts/reports/validation.json` | Fix or reject the source data. Never loosen a rule just to make it pass |

## Backups

    docker compose exec -T postgres pg_dumpall -U postgres > backup-$(date +%F).sql
    docker run --rm -v modern-mlops_mlflow-artifacts:/data -v "$PWD":/backup alpine:3.20 \
      tar czf /backup/mlflow-artifacts-$(date +%F).tgz -C /data .

Restore by recreating the volumes and loading the dump. Practise a restore at least once.

## Rotate an API key

1. Add the new key to `API_KEYS` (comma-separated), `docker compose up -d api`.
2. Move clients to the new key.
3. Remove the old key, `docker compose up -d api`.
````

### The retraining policy

- **Scheduled:** `training_job` runs every Monday and produces a candidate.
- **On signal:** run it by hand when the drift or accuracy check fails.
- **Promotion:** always by a person, only for versions that passed the gate, and only after
  reading the model card.
- **Rollback:** when the error rate, latency or live accuracy gets worse after a release.

------------------------------------------------------------------------

## Troubleshooting

### `docker compose config` says a variable is not set

`.env` is missing a value or is not in the project folder. Compare it with `.env.example`.

### "ports are not available ... 5432"

Port 5432 is already used or blocked on your machine (for example by another Postgres
installation). Set `POSTGRES_HOST_PORT=5433` in `.env`, and use the same port in `DATABASE_URL`.
The containers are not affected: inside Docker, Postgres always uses 5432.

### Postgres fails to start the init script, or a service cannot log in to the database

- The `.sh` file must have Unix line endings (LF). In VS Code, check the `LF`/`CRLF` indicator
  in the bottom-right corner of the window.
- The script only runs when the database is first created. If you changed passwords afterwards,
  reset the database with `docker compose down -v` (this deletes all data) and start again.

### "Permission denied" on `/app/data` or `/app/artifacts` (Linux)

The containers run as user ID 1000. Create `data/` and `artifacts/` yourself before starting
Compose. If your user ID is not 1000: `sudo chown -R 1000:1000 data artifacts`.

### `/ready` returns 503

No model is loaded. Run `docker compose exec api mlops-model status`. If there is no champion,
train and promote one (Step 19). If there is, read the API's logs with
`docker compose logs api`; the API retries loading every 30 seconds, or you can restart it.

### Git Bash: `/app/workspace.yaml` becomes `C:/Program Files/Git/app/...`

Git Bash on Windows rewrites arguments that look like Linux paths. Put `MSYS_NO_PATHCONV=1` in
front of the command, or use PowerShell.

### Spark tests fail with "Python worker failed to connect back"

Check that you run the tests with `uv run pytest`, so Spark uses the project's Python. If it
still fails, check your free memory: Spark and its workers need about 1–2 GB.

### Out of memory, or everything is very slow

Close other applications, check usage with `docker stats --no-stream`, and make sure Docker has
at least 4 GB. You do not need every service at once: to train, only `postgres`, `mlflow` and the
Dagster services are needed.

### Prometheus shows no API metrics

Open <http://localhost:9090/targets>. Inside Compose the target must be `api:8000`, not
`localhost:8000`.

### Grafana: the "Monitoring DB" data source fails

The read-only user is created by the init script when the database is first created. If the
database existed before, reset it with `docker compose down -v` (this deletes all data).

### `uv lock` fails

Read the whole error message: it names the two packages that conflict. Change only those version
limits, then run the tests again. Do not delete `uv.lock` to "start fresh".

------------------------------------------------------------------------

## Final checklist

You are done when you can demonstrate **and explain** each item.

**Setup and quality**

-   [ ] `uv sync --all-groups` works and `uv.lock` is committed.
-   [ ] `ruff check`, `ruff format --check` and `pytest` pass on your machine and in CI.
-   [ ] CI builds both images and runs the security scans.
-   [ ] `.env` is not in Git, and every password was changed from the template.

**Data and training**

-   [ ] The pipeline stops with a clear report when the data breaks a rule (exercise 5).
-   [ ] Training data comes from Feast, and the train/test split is by date.
-   [ ] Each model version in MLflow has metrics, the data's fingerprint, the validation report
    and a model card.
-   [ ] The gate rejects a worse model, and `promote` refuses it (exercise 2).

**Serving**

-   [ ] The API serves the champion; `/ready` shows its version.
-   [ ] A request without a valid API key gets 401; invalid input gets 422.
-   [ ] Every prediction is stored with an ID, and feedback can be attached to it.
-   [ ] Promotion and rollback work, and `/ready` shows the change (exercise 3).

**Monitoring and operations**

-   [ ] Prometheus collects the API's metrics, and the dashboard loads automatically.
-   [ ] The `PredictionLogFailures` alert fires when Postgres is down (exercise 4).
-   [ ] The accuracy check fails when prices shift (exercise 1).
-   [ ] The smoke test passes after a release; you recorded the load test's p95 latency.
-   [ ] The runbook exists, and you took a backup and restored it once.
-   [ ] You can stop the stack without deleting data.

------------------------------------------------------------------------

## Going further (optional)

Once everything above works, here are good next challenges, roughly from easiest to hardest:

- **Alert delivery**: add Alertmanager and send alerts by email or to a chat channel.
- **HTTPS**: put a reverse proxy (Caddy or Traefik) in front of the API.
- **Better model**: try more features or another algorithm; the gate tells you if it is
  really better.
- **Hot reload**: let the API pick up a new champion without a restart.
- **Canary release**: send 10% of traffic to the challenger and compare.
- **Feast online store**: let the API look up features by `house_id` instead of receiving
  them in the request.
- **Cloud deployment**: run the same images on a managed container service.

Whatever you add, follow the rule this project is built on: **one change at a time, and prove
each one works with a check you can show.**