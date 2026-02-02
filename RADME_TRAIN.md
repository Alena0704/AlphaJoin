# How to Run AlphaJoin Training

## Dependencies (required)

Install and configure the following before running training or data collection.

### Python (for training)

- **Python 3** with packages: `torch`, `numpy` (e.g. `pip install torch numpy`, or use the project `venv`).

### Java + PostgreSQL (for tools and data collection)

- **JDK** (to compile and run the Java tools).
- **PostgreSQL JDBC driver** (e.g. `postgresql-42.x.x.jar`). Put it in `tools/` or on the classpath when running `getruntimerandom`, `gethintresult`, `getoriginruntime`, etc.
- **`AlphaJoin/tools/JDBCUtil.java`** — provides the DB connection for all Java tools. Edit the constants inside to match your PostgreSQL instance:
  - `URL` — e.g. `jdbc:postgresql://localhost:5496/postgres`
  - `NAME`, `PASSWORD` — DB user and password
  Compile it together with the other tools: `javac JDBCUtil.java getruntimerandom.java` (and add the JDBC JAR to the classpath if needed: `javac -cp .:postgresql-42.x.x.jar JDBCUtil.java getruntimerandom.java`).

### Database setup scripts

- **`job-create`** — shell script that creates and loads the JOB database (schema, indexes, data). You **must** run it (or equivalent steps) once so that PostgreSQL has the JOB schema and data. Edit the script to set:
  - `INSTDIR` — path to PostgreSQL binaries (e.g. `psql`, `pg_ctl`)
  - `PGDATA`, `PORT` — data directory and port
  - `QUERY_DIR` — directory containing `schema.sql`, `fkindexes.sql`, `copy.sql`
  Then run: `./job-create` (or `bash job-create`).

- **`job_test_first`** — optional shell script for baseline timing (runs EXPLAIN and queries, appends results to e.g. `original_time.csv`). Edit `INSTDIR`, `PGDATA`, `PGPORT`, and the paths to `source/queries/*.sql` and output files as needed. Run with: `./job_test_first` (or `bash job_test_first`).

### Summary

| Dependency | Purpose |
|------------|--------|
| Python 3, torch, numpy | Training (AlphaJoin1.0) |
| JDK + PostgreSQL JDBC JAR | Compile/run Java tools in `tools/` |
| `tools/JDBCUtil.java` | DB connection; edit URL/user/password |
| `job-create` | Create and load JOB database (required for data collection) |
| `job_test_first` | Optional baseline timing over JOB queries |

---

## 1. Prepare Resources (once)

From the **AlphaJoin1.0** directory:

```bash
cd AlphaJoin1.0
python3 2.getQueryEncode.py
```

**Required:** JOB query files (`.sql`) in `../resource/jobquery/`.
**Created:** `../resource/jobtablename/`, `../resource/shorttolong`, `predicatesEncodedDict`, `queryEncodedDict`.

(If needed, run `python3 1.getResource.py` first or use the DB — see `1.getResource.py`.)

## 2. Collect Training Data (if you don’t have a CSV yet)

Run the Java data collector from the **tools** directory (requires PostgreSQL, `JDBCUtil`, `dropCache.sh`):

```bash
cd tools
javac JDBCUtil.java getruntimerandom.java
java getruntimerandom
```

**Output:** `./t6.sql` (lines like `queryName,hint,runtime,...`) and the `../data/` directory with plans.

Training expects a file with columns **queryName, hint, runtime** (other columns are ignored). You can use the Java output directly:

```bash
mkdir -p AlphaJoin1.0/data
# If data was collected in tools/t6.sql, use it as the dataset:
# when running training, pass: --data-file ../tools/t6.sql
```

Or place your own file at `AlphaJoin1.0/data/training.csv` (format: lines `queryName,hint,runtime` or with extra columns).

## 3. Run Training

From the **AlphaJoin1.0** directory:

```bash
cd AlphaJoin1.0
source ../venv/bin/activate   # if using venv
python3 6.train_network.py
```

Data is loaded automatically from the file given by `--data-file` (default: `./data/training.csv`).

Example with an explicit data path and model directory:

```bash
python3 6.train_network.py --data-file ./data/training.csv --save-dir saved_models/
```

If your data is in `tools/t6.sql`:

```bash
python3 6.train_network.py --data-file ../tools/t6.sql
```

**Arguments** (from `0.arguments.py`):

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-file` | Path to CSV (queryName,hint,runtime,...) | `./data/training.csv` |
| `--save-dir` | Directory to save the model | `saved_models/` |
| `--timeout-value` | Value used when runtime is "timeout" | 1e9 |
| `--train-steps` | Number of SGD steps | 300000 |
| `--lr` | Learning rate | 0.01 |
| `--test-every` | Run test_network every N steps | 1000 |
| `--save-every` | Save model every N steps | 200000 |

The model is saved as `saved_models/supervised.pt`.

## Quick Sequence (when data already exists)

```bash
cd AlphaJoin1.0
python3 2.getQueryEncode.py          # if not done yet
python3 6.train_network.py --data-file /path/to/your/training.csv
```

## When You Don’t Have Data Yet

1. `python3 2.getQueryEncode.py`
2. `cd ../tools && javac *.java && java getruntimerandom` (requires a running PostgreSQL instance).
3. `cd ../AlphaJoin1.0 && python3 6.train_network.py --data-file ../tools/t6.sql`
