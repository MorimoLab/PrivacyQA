"""Shared helpers: explicit failures, file input and output, fixed public orderings and progress records."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path

DOMAINS = ('movie', 'beauty', 'recipe_text')
RATING_DOMAINS = ('movie', 'beauty')
ARMS = ('main', 'random')
VIEWS = ('C-only', 'C+weight')
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ContractError(ValueError):
    """An input or state that the pipeline does not accept; `code` names the failed check."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition:
        raise ContractError(code, message)


def atoms(values):
    """Canonical aspect::direction constraints: unique and sorted."""
    require(isinstance(values, (list, tuple)) and all(isinstance(v, str) and v.strip() == v and '::' in v
                                                      and all(v.split('::', 1)) for v in values),
            'invalid_atoms', 'Constraints are explicit aspect::direction strings')
    return tuple(sorted(set(values)))


def identifier(value, field):
    require(isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip(),
            'missing_identity', f'{field} must be a nonempty identifier')
    return str(value).strip()


# ---------------------------------------------------------------- fixed orderings

def rank(seed, *parts):
    """Order of users, releases, partial histories and test requests under a fixed seed."""
    return hashlib.sha256(json.dumps([seed, *parts], separators=(',', ':')).encode()).hexdigest()


def public_rank(seed, value):
    """Order of cores in the request banks under a fixed seed."""
    return hashlib.sha256((str(seed) + ':' + json.dumps(value, ensure_ascii=False, separators=(',', ':'))).encode()).hexdigest()


def key(*parts):
    """Identity and order of QA records."""
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- files

def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, payload):
    with Path(path).open('x', encoding='utf-8') as handle:
        os.chmod(path, 0o600)
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def write_rows(path, rows):
    with Path(path).open('x', encoding='utf-8') as handle:
        os.chmod(path, 0o600)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def iter_rows(path, format):
    """Rows of a table in an explicitly named format."""
    if format == 'parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            yield from batch.to_pylist()
    elif format == 'jsonl':
        with Path(path).open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif format in ('csv', 'tsv', 'csv.gz'):
        opener = gzip.open if format == 'csv.gz' else open
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
            yield from csv.DictReader(handle, delimiter='\t' if format == 'tsv' else ',')
    else:
        raise ContractError('invalid_table_format', 'Use parquet, jsonl, csv, csv.gz or tsv')


def write_parquet(path, rows, schema):
    import pyarrow as pa
    import pyarrow.parquet as pq
    require(not Path(path).exists(), 'artifact_exists', 'Outputs are never overwritten')
    count, batch = 0, []
    with pq.ParquetWriter(path, schema, compression='zstd') as writer:
        os.chmod(path, 0o600)
        for row in rows:
            batch.append(row)
            if len(batch) == 4096:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                count += len(batch)
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            count += len(batch)
    return count


def save_arrays(path, **arrays):
    import numpy as np
    with Path(path).open('xb') as stream:
        os.chmod(path, 0o600)
        np.savez_compressed(stream, **arrays)


def load_arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k].copy() for k in values.files}


def save_sparse(path, matrix):
    from scipy import sparse
    with Path(path).open('xb') as stream:
        os.chmod(path, 0o600)
        sparse.save_npz(stream, matrix)


def load_sparse(path):
    from scipy import sparse
    return sparse.load_npz(path).tocsr()


def progress(output, stage, **facts):
    path = Path(output) / 'progress.jsonl'
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(), stage=stage, **facts)) + '\n')
    path.chmod(0o600)
