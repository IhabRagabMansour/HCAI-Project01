from io import BytesIO

import pandas as pd

HEAD_PREVIEW_ROWS = 10
CLASSIFICATION_UNIQUE_THRESHOLD = 10


def read_csv_safely(file_field) -> pd.DataFrame:
    """Read a CSV from a Django FileField into a DataFrame.
    Tries utf-8-sig then latin-1. Raises ValueError on failure.
    """
    file_field.open("rb")
    try:
        raw = file_field.read()
    finally:
        file_field.close()

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(BytesIO(raw), encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty or contains no parseable data.")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV could not be parsed: {e}")

    raise ValueError("File encoding is not supported (tried utf-8, latin-1).")


def humanize_dtype(dtype) -> str:
    name = str(dtype)
    if name.startswith("int"):
        return "integer"
    if name.startswith("float"):
        return "float"
    if name == "bool":
        return "boolean"
    if name.startswith("datetime"):
        return "datetime"
    if name == "object":
        return "string"
    return name


def infer_problem_type(df: pd.DataFrame) -> str:
    """Return 'classification' | 'regression' | 'unknown' based on last column."""
    if df.shape[1] < 2:
        return "unknown"
    target = df.iloc[:, -1]
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    n_unique = target.nunique(dropna=True)
    if n_unique <= CLASSIFICATION_UNIQUE_THRESHOLD:
        return "classification"
    return "regression"


def extract_metadata(df: pd.DataFrame) -> dict:
    """Extract all structural info to persist on the Dataset model."""
    columns = [
        {"name": col, "dtype": humanize_dtype(df[col].dtype)}
        for col in df.columns
    ]
    head_df = df.head(HEAD_PREVIEW_ROWS)
    head_rows = (
        head_df.astype(object)
               .where(head_df.notna(), None)
               .values.tolist()
    )
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": columns,
        "head": head_rows,
        "target_name": str(df.columns[-1]) if df.shape[1] else None,
        "problem_type": infer_problem_type(df),
    }
