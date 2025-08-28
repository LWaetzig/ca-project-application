import os
from typing import Any

import pandas as pd
import streamlit as st
import pyarrow.parquet as pq


def preload_table_content(
    file_path: str, columns: list[str] | None = None, downcast: bool = True
) -> pd.DataFrame:
    """Load table content from Parquet file with optional column selection and downcasting.

    Args:
        file_path (str): The path to the Parquet file.
        columns (list[str] | None, optional): The columns to load from the file. Defaults to None.
        downcast (bool, optional): Whether to downcast numeric columns to more efficient types. Defaults to True.

    Returns:
        pd.DataFrame: The loaded DataFrame
    """

    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()

        df = pd.read_parquet(file_path, columns=columns)
        if downcast and not df.empty:
            # Downcast integer and float columns where possible
            for col in df.select_dtypes(
                include=["int", "int64", "float", "float64"]
            ).columns:
                try:
                    if pd.api.types.is_integer_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], downcast="integer")
                    elif pd.api.types.is_float_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], downcast="float")
                except Exception:
                    pass

            # Convert low-cardinality object columns to category (heuristic)
            for col in df.select_dtypes(include=["object"]).columns:
                try:
                    n = len(df)
                    nunique = df[col].nunique(dropna=True)
                    if n > 0 and nunique > 0 and nunique / n <= 0.5:
                        df[col] = df[col].astype("category")
                except Exception:
                    pass
        return df
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()


def get_parquet_metadata(file_path: str) -> dict[str, Any]:
    """Get basic Parquet metadata (columns and row count) without loading the whole file.

    Args:
        file_path (str): The path to the Parquet file.

    Returns:
        dict[str, Any]: A dictionary containing the metadata information.
    """

    meta: dict[str, Any] = {"columns": [], "row_count": 0}

    try:
        try:
            pf = pq.ParquetFile(file_path)
            meta["columns"] = [s.name for s in pf.schema]
            meta["row_count"] = pf.metadata.num_rows if pf.metadata is not None else 0

        except Exception:
            # Fallback: load only schema via pandas if pyarrow fails
            df_head = pd.read_parquet(file_path, columns=None)
            meta["columns"] = list(df_head.columns)
            meta["row_count"] = len(df_head)

    except Exception:
        pass
    return meta


def get_table_info(df: pd.DataFrame) -> dict[str, Any]:
    """Extract table information from DataFrame

    Args:
        df (pd.DataFrame): The input DataFrame to extract information from.

    Returns:
        dict[str, Any]: A dictionary containing the table information.
    """

    if df.empty:
        return {"columns": [], "column_names": [], "row_count": 0, "memory_usage_mb": 0}

    columns_info = []
    for col in df.columns:
        columns_info.append(
            {
                "column_name": col,
                "data_type": str(df[col].dtype),
                "non_null_count": df[col].count(),
                "null_count": df[col].isnull().sum(),
            }
        )

    return {
        "columns": columns_info,
        "column_names": df.columns.tolist(),
        "row_count": len(df),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }


def get_sample_data(df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    """Get a random sample from DataFrame

    Args:
        df (pd.DataFrame): The input DataFrame to sample from.
        sample_size (int, optional): The number of rows to sample. Defaults to 1000.

    Returns:
        pd.DataFrame: The sampled DataFrame
    """

    if df.empty or len(df) <= sample_size:
        return df

    return df.sample(n=sample_size, random_state=42)


def filter_dataframe(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    filters: dict | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Filter DataFrame with optional column selection, filters, and limit

    Args:
        df (pd.DataFrame): The input DataFrame to filter.
        columns (list[str] | None, optional): Columns to select. Defaults to None.
        filters (dict | None, optional): Filters to apply. Defaults to None.
        limit (int | None, optional): Maximum number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """

    if df.empty:
        return df

    result_df = df

    # apply filters
    if filters:
        for column, filter_config in filters.items():
            if column not in result_df.columns:
                continue

            filter_type = filter_config.get("type")
            filter_value = filter_config.get("value")

            if filter_type == "equals":
                result_df = result_df[result_df[column] == filter_value]
            elif filter_type == "contains":
                result_df = result_df[
                    result_df[column].str.contains(filter_value, na=False)
                ]
            elif filter_type == "greater_than":
                result_df = result_df[result_df[column] > filter_value]
            elif filter_type == "less_than":
                result_df = result_df[result_df[column] < filter_value]
            elif filter_type == "between":
                min_val, max_val = filter_value
                result_df = result_df[
                    (result_df[column] >= min_val) & (result_df[column] <= max_val)
                ]

    # select columns
    if columns:
        available_columns = [col for col in columns if col in result_df.columns]
        if available_columns:
            result_df = result_df[available_columns]

    # apply limit
    if limit and len(result_df) > limit:
        result_df = result_df.head(limit)

    return result_df


def calculate_memory_usage() -> dict[str, float]:
    """Calculate memory usage of all loaded tables

    Returns:
        dict[str, float]: Memory usage of each table in MB
    """
    memory_usage: dict[str, float] = {}

    if "db_tables" not in st.session_state:
        return memory_usage

    # calculate memory usage for each table
    for table_name in st.session_state["db_tables"]:
        if table_name in st.session_state:
            df = st.session_state[table_name]
            if isinstance(df, pd.DataFrame):
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                memory_usage[table_name] = memory_mb

    return memory_usage
