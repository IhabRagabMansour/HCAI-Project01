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
    if name in ("object", "str", "string"):
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


MAX_CLASSES_FOR_PLOT = 20


def numeric_column_names(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def build_chart_data(df: pd.DataFrame, x_col: str, y_col: str, mode: str, target_name: str | None) -> dict:
    """Return Chart.js-ready datasets list."""
    TAB10 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]

    # Select x/y as independent Series to avoid duplicate-column issues when x_col == y_col
    x_vals = df[x_col]
    y_vals = df[y_col]
    valid = x_vals.notna() & y_vals.notna()
    x_clean = x_vals[valid]
    y_clean = y_vals[valid]

    def to_point(xi, yi):
        return {"x": float(xi), "y": float(yi)}

    if mode == "classification" and target_name and target_name in df.columns:
        labels = df[target_name][valid].astype(str)
        unique_labels = sorted(labels.unique())
        if len(unique_labels) > MAX_CLASSES_FOR_PLOT:
            mode = "regression"
        else:
            datasets = []
            for i, label in enumerate(unique_labels):
                mask = labels == label
                points = [to_point(xi, yi) for xi, yi in zip(x_clean[mask], y_clean[mask])]
                datasets.append({
                    "label": label,
                    "data": points,
                    "backgroundColor": TAB10[i % len(TAB10)],
                    "pointRadius": 4,
                })
            return {"datasets": datasets}

    points = [to_point(xi, yi) for xi, yi in zip(x_clean, y_clean)]
    return {"datasets": [{"label": f"{y_col} vs {x_col}", "data": points,
                           "backgroundColor": TAB10[0], "pointRadius": 4}]}


def build_histogram_data(df: pd.DataFrame, col: str, bins: int = 20) -> dict:
    import numpy as np
    values = df[col].dropna().astype(float).values
    counts, edges = np.histogram(values, bins=bins)
    midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    labels = [f"{m:.3g}" for m in midpoints]
    return {
        "labels": labels,
        "datasets": [{
            "label": col,
            "data": counts.tolist(),
            "backgroundColor": "#1f77b4bb",
            "borderColor": "#1f77b4",
            "borderWidth": 1,
        }],
    }


def build_boxplot_data(df: pd.DataFrame, col: str, target_name: str | None, mode: str) -> dict:
    if mode == "classification" and target_name and target_name in df.columns:
        grouped = df.groupby(df[target_name].astype(str))[col]
        labels = sorted(grouped.groups.keys())
        data = [grouped.get_group(lbl).dropna().tolist() for lbl in labels]
    else:
        labels = [col]
        data = [df[col].dropna().tolist()]
    return {
        "labels": labels,
        "datasets": [{
            "label": col,
            "data": data,
            "backgroundColor": "#1f77b455",
            "borderColor": "#1f77b4",
            "borderWidth": 2,
            "outlierColor": "#d62728",
        }],
    }


def build_heatmap_data(df: pd.DataFrame, cols: list[str]) -> dict:
    corr = df[cols].corr()
    col_names = corr.columns.tolist()
    matrix = [
        {"x": x_col, "y": y_col, "v": round(float(corr.loc[y_col, x_col]), 3)}
        for y_col in col_names
        for x_col in col_names
    ]
    return {"cols": col_names, "data": matrix}


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
