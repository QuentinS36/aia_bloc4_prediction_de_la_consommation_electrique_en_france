import pandas as pd
import numpy as np
import warnings
from typing import Union


def _agg_series(series: pd.Series, func: str):
    """Aggregate a numeric or datetime series according to the given function."""
    non_null = series.dropna()
    if pd.api.types.is_datetime64_any_dtype(series):
        tz = series.dt.tz
        if func == "min":
            return non_null.min()
        elif func == "max":
            return non_null.max()
        elif func == "mean":
            ns = np.array([t.value for t in non_null])
            return pd.Timestamp(int(ns.mean()), unit="ns", tz=tz)
        elif func == "median":
            ns = np.array([t.value for t in non_null])
            return pd.Timestamp(int(np.median(ns)), unit="ns", tz=tz)
    else:
        return {"mean": non_null.mean, "min": non_null.min,
                "max": non_null.max, "median": non_null.median}[func]()


def aggregate_to_hourly(
    df: pd.DataFrame,
    start_date: str = "start_date",
    end_date: str = "end_date",
    agg_func: Union[str, dict] = "mean",
) -> pd.DataFrame:
    """
    Aggregate a quarter-hourly DataFrame into an hourly DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the `start_date` and `end_date` columns (configurable names).
    start_date : str
        Name of the column containing the start timestamp of each period.
    end_date : str
        Name of the column containing the end timestamp of each period.
    agg_func : str or dict
        - str  : function applied to all columns except start_date and end_date.
        - dict : mapping {column_name: function} for per-column control.
                 Only the listed columns will appear in the output DataFrame.
        Allowed functions: "mean", "min", "max", "median".
        Columns of string type that cannot be parsed as datetime cannot be
        aggregated and will raise a ValueError.

    Returns
    -------
    pd.DataFrame
        Aggregated hourly DataFrame.
    """
    VALID_FUNCS = {"mean", "min", "max", "median"}

    # --- Validate and build the aggregation dictionary ---
    if isinstance(agg_func, str):
        if agg_func not in VALID_FUNCS:
            raise ValueError(
                f"agg_func='{agg_func}' is invalid. Allowed values: {VALID_FUNCS}"
            )
        value_cols = [c for c in df.columns if c not in (start_date, end_date)]
        agg_dict = {col: agg_func for col in value_cols}
    elif isinstance(agg_func, dict):
        for col, func in agg_func.items():
            if func not in VALID_FUNCS:
                raise ValueError(
                    f"agg_func['{col}']='{func}' is invalid. Allowed values: {VALID_FUNCS}"
                )
        agg_dict = agg_func
    else:
        raise TypeError(
            f"agg_func must be a str or a dict, not {type(agg_func).__name__}"
        )

    df = df.copy()

    # --- Convert start_date and end_date columns if necessary ---
    for col in (start_date, end_date):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], utc=True)

    # --- Inspect and convert columns to be aggregated ---
    # col_meta records whether a column was originally a datetime string
    col_meta = {}
    for col in agg_dict:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is missing from the DataFrame.")
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            col_meta[col] = {"is_datetime_str": False}
        elif dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce")
            original_nulls = df[col].isna().sum()
            parsed_nulls = parsed.isna().sum()
            if parsed_nulls == original_nulls:
                # All non-null values are valid datetimes
                df[col] = parsed
                col_meta[col] = {"is_datetime_str": True}
            else:
                raise ValueError(
                    f"Column '{col}' contains strings that cannot be parsed as "
                    f"datetime and therefore cannot be aggregated."
                )
        else:
            col_meta[col] = {"is_datetime_str": False}

    # --- Group by calendar hour ---
    df["_hour_group"] = df[start_date].dt.floor("h")

    results = []

    for hour, group in df.groupby("_hour_group", sort=True):
        row = {
            start_date: hour,
            end_date:   hour + pd.Timedelta(hours=1),
        }
        for col, func in agg_dict.items():
            row[col] = _agg_series(group[col], func)
        results.append(row)

    output_cols = [start_date, end_date] + list(agg_dict.keys())
    df_out = pd.DataFrame(results, columns=output_cols)

    # --- Check temporal continuity of the output DataFrame ---
    if len(df_out) > 1:
        for i in range(len(df_out) - 1):
            e = df_out[end_date].iloc[i]
            s = df_out[start_date].iloc[i + 1]
            if e != s:
                warnings.warn(
                    f"Temporal gap detected between row {i} "
                    f"(end_date={e}) and row {i + 1} (start_date={s})."
                )

    # --- Convert back to string columns that were originally of string type ---
    for col, meta in col_meta.items():
        if meta["is_datetime_str"] and col in df_out.columns:
            df_out[col] = df_out[col].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            )

    return df_out
