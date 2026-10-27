"""Accès Postgres (Neon) : schéma et écritures idempotentes."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import (Column, DateTime, Float, MetaData, String, Table, create_engine,
                        func, select)
from sqlalchemy.engine import Engine

from dday.config import database_url, to_sqlalchemy_url

metadata = MetaData()

DATASET_COLS = ["y", "mean", "min", "max", "median", "std", "q1", "q3"]

# Données d'entraînement : consommation + statistiques de température, pas horaire (UTC)
dataset_hourly = Table(
    "dataset_hourly", metadata,
    Column("ds", DateTime, primary_key=True),
    *[Column(c, Float) for c in DATASET_COLS],
)

# Prévisions servies par l'API : sert au monitoring (prévu vs réel)
predictions = Table(
    "predictions", metadata,
    Column("ds", DateTime, primary_key=True),
    Column("yhat", Float, nullable=False),
    Column("model_version", String, primary_key=True),
    Column("created_at", DateTime, server_default=func.now()),
)


def get_engine(url: str | None = None) -> Engine:
    return create_engine(to_sqlalchemy_url(url) if url else database_url(), pool_pre_ping=True)


def init_schema(engine: Engine) -> None:
    metadata.create_all(engine)


def _upsert(engine: Engine, table: Table, rows: list[dict], keys: list[str]) -> int:
    if not rows:
        return 0
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:  # sqlite (tests)
        from sqlalchemy.dialects.sqlite import insert
    stmt = insert(table).values(rows)
    update_cols = {c.name: stmt.excluded[c.name] for c in table.columns
                   if c.name not in keys and c.name != "created_at"}
    stmt = stmt.on_conflict_do_update(index_elements=keys, set_=update_cols)
    with engine.begin() as conn:
        conn.execute(stmt)
    return len(rows)


def _clean_rows(df: pd.DataFrame) -> list[dict]:
    df = df.astype(object).where(df.notna(), None)
    rows = df.to_dict("records")
    for r in rows:
        r["ds"] = pd.Timestamp(r["ds"]).to_pydatetime()
    return rows


def upsert_dataset(engine: Engine, df: pd.DataFrame) -> int:
    """Insère ou met à jour les lignes (clé : ds). Rejouable sans doublon."""
    return _upsert(engine, dataset_hourly, _clean_rows(df[["ds"] + DATASET_COLS]), ["ds"])


def upsert_predictions(engine: Engine, df: pd.DataFrame, model_version: str) -> int:
    d = df[["ds", "yhat"]].copy()
    d["model_version"] = str(model_version)
    return _upsert(engine, predictions, _clean_rows(d), ["ds", "model_version"])


def read_dataset(engine: Engine, since: pd.Timestamp | None = None) -> pd.DataFrame:
    q = select(dataset_hourly).order_by(dataset_hourly.c.ds)
    if since is not None:
        q = q.where(dataset_hourly.c.ds >= since.to_pydatetime())
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def read_predictions(engine: Engine, since: pd.Timestamp | None = None) -> pd.DataFrame:
    q = select(predictions).order_by(predictions.c.ds)
    if since is not None:
        q = q.where(predictions.c.ds >= since.to_pydatetime())
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    df["ds"] = pd.to_datetime(df["ds"])
    return df
