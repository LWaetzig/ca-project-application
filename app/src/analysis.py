from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.utils import (
    calculate_memory_usage,
    filter_dataframe,
    get_sample_data,
    get_table_info,
)


def dashboard_view() -> None:
    """Display the dashboard overview interface of all loaded data"""

    st.markdown("## Dashboard Overview")

    if "db_tables" not in st.session_state:
        st.error("No tables loaded")
        return

    total_rows = sum(
        len(st.session_state[table]) for table in st.session_state["db_tables"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tables Loaded", len(st.session_state["db_tables"]))
    with col2:
        st.metric("Total Records", f"{total_rows:,}")
    with col3:
        largest_table = max(
            st.session_state["db_tables"], key=lambda x: len(st.session_state[x])
        )
        st.metric(
            "Largest Table",
            f"{largest_table} ({len(st.session_state[largest_table]):,} rows)",
        )

    st.markdown("### Database Details")

    table_overview = []
    for table_name in st.session_state["db_tables"]:
        df = st.session_state[table_name]
        table_overview.append(
            {
                "Table": table_name,
                "Rows": f"{len(df):,}",
                "Columns": len(df.columns),
                "Missing Values": f"{df.isnull().sum().sum():,}",
            }
        )

    overview_df = pd.DataFrame(table_overview)
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    st.markdown("### Quick Insights")

    col1, col2 = st.columns(2)

    # display visualization of table sizes
    with col1:
        table_sizes = [
            len(st.session_state[table]) for table in st.session_state["db_tables"]
        ]
        fig = px.bar(
            x=st.session_state["db_tables"], y=table_sizes, title="Records per Table"
        )
        fig.update_xaxes(title="Tables")
        fig.update_yaxes(title="Record Count", type="log")
        st.plotly_chart(fig, use_container_width=True)

    # display visualization of memory usage per table
    with col2:
        memory_usage = calculate_memory_usage()
        memory_values = [
            memory_usage.get(table, 0) for table in st.session_state["db_tables"]
        ]
        fig = px.pie(
            values=memory_values,
            names=st.session_state["db_tables"],
            title="Memory Usage by Table",
        )
        st.plotly_chart(fig, use_container_width=True)


def table_explorer():
    """Display table exploration interface"""
    st.markdown("## Table Explorer")

    if "db_tables" not in st.session_state:
        st.error("No tables loaded")
        return
    select_col, _ = st.columns([1, 2])
    with select_col:
        selected_table = st.selectbox(
            "Select Table", st.session_state["db_tables"], index=None
        )

    if not selected_table:
        return

    st.divider()

    df = st.session_state.get(selected_table, pd.DataFrame())

    if df.empty:
        st.warning(f"No data available for {selected_table}")
        return

    col1, col2 = st.columns([1, 2])

    # display table information
    with col1:
        st.markdown("### Table Information")
        table_info = get_table_info(df)

        col_metrics1, col_metrics2 = st.columns(2)
        with col_metrics1:
            st.metric("Total Rows", f"{table_info['row_count']:,}")
            st.metric("Total Columns", len(table_info["columns"]))
        with col_metrics2:
            st.metric("Memory Usage", f"{table_info['memory_usage_mb']:.1f} MB")
            missing_values = sum(col["null_count"] for col in table_info["columns"])
            st.metric("Missing Values", f"{missing_values:,}")

        # Column information
        st.markdown("### Column Details")
        columns_df = pd.DataFrame(table_info["columns"])
        st.dataframe(columns_df, use_container_width=True, hide_index=True)

    # display data preview
    with col2:
        st.subheader(f"Data Preview: {selected_table}")

        col_columns, col_sample = st.columns(2)

        # allow filtering of columns
        with col_columns:
            show_columns = st.multiselect(
                "Select Columns", df.columns.tolist(), default=df.columns.tolist()[:5]
            )

        # change sample size
        with col_sample:
            sample_size_mapping = {
                100: 100,
                500: 500,
                1000: 1000,
                5000: 5000,
                "All (this could take a while)": -1,
            }
            sample_size = st.selectbox(
                "Sample Size", list(sample_size_mapping.keys()), index=2
            )
            if sample_size == "All (this could take a while)":
                display_df = df
            else:
                display_df = get_sample_data(df, sample_size_mapping[sample_size])

        # display table data based on filter
        if show_columns:
            display_df = display_df[show_columns]
        st.dataframe(display_df, use_container_width=True)


def variable_analysis():
    """Variable analysis interface"""

    st.markdown("## Variable Analysis")

    if "db_tables" not in st.session_state:
        st.error("No tables loaded")
        return

    st.markdown("### Analysis Setup")
    setup_col, sampling_col = st.columns(2)
    with setup_col:
        # Table and column selection
        col1, col2 = st.columns(2)

        with col1:
            selected_table = st.selectbox(
                "Select Table",
                st.session_state["db_tables"],
                index=None,
                key="var_table",
            )

        if not selected_table:
            return

        df = st.session_state.get(selected_table, pd.DataFrame())
        if df.empty:
            st.warning(f"No data available for {selected_table}")
            return

        with col2:
            selected_columns = st.multiselect(
                "Select Variables",
                df.columns.tolist(),
                key="var_columns",
            )

        if not selected_columns:
            st.info("Please select at least one variable for analysis")
            return

    # Sample size control
    with sampling_col:
        analysis_sampling_mapping = {
            1000: 1000,
            5000: 5000,
            10000: 10000,
            25000: 25000,
            "Full Dataset (this could take a while)": -1,
        }
        analysis_sample_size = st.selectbox(
            "Choose Sample Size",
            list(analysis_sampling_mapping.keys()),
            index=2,
        )

        # Prepare analysis data
        if analysis_sample_size == "Full Dataset (this could take a while)":
            analysis_df = df[selected_columns]
        else:
            analysis_df = get_sample_data(
                df[selected_columns], analysis_sampling_mapping[analysis_sample_size]
            )

    # Analysis type selection
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        [
            "Descriptive Statistics",
            "Distribution Analysis",
            "Text Analysis",
        ],
    )

    st.divider()

    analysis_df = df[selected_columns]
    if analysis_type == "Descriptive Statistics":
        descriptive_statistics(analysis_df)
    elif analysis_type == "Distribution Analysis":
        distribution_analysis(analysis_df, selected_columns)
    elif analysis_type == "Text Analysis":
        text_analysis(analysis_df, selected_columns)


def descriptive_statistics(df: pd.DataFrame):
    """Display comprehensive descriptive statistics"""

    st.markdown("### Descriptive Statistics")

    # separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category"], exclude=["datetime"]
    ).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    # numeric statistics
    if numeric_cols:
        st.markdown("#### Numerical Variables")
        numeric_stats = df[numeric_cols].describe()

        # Add additional statistics
        additional_stats = pd.DataFrame(
            {
                col: {
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis(),
                    "variance": df[col].var(),
                    "missing_count": df[col].isnull().sum(),
                    "missing_percentage": (df[col].isnull().sum() / len(df)) * 100,
                }
                for col in numeric_cols
            }
        ).T

        full_stats = pd.concat([numeric_stats.T, additional_stats], axis=1)
        st.dataframe(full_stats, use_container_width=True)

    # Categorical statistics
    if categorical_cols:
        st.markdown("#### Categorical Variables")

        for col in categorical_cols:
            with st.expander(label=col):
                col_stats1, col_stats2 = st.columns(2)

                with col_stats1:
                    st.markdown("**SUMMARY**")
                    summary = {
                        "Unique Values": df[col].nunique(),
                        "Most Frequent": (
                            df[col].mode().iloc[0]
                            if not df[col].mode().empty
                            else "N/A"
                        ),
                        "Missing Values": df[col].isnull().sum(),
                        "Missing %": f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%",
                    }
                    for key, value in summary.items():
                        st.metric(key, value)

                with col_stats2:
                    st.markdown("**TOP 10 VALUES**")
                    value_counts = df[col].value_counts().head(10)
                    st.dataframe(value_counts.reset_index())

    # DateTime statistics
    if datetime_cols:
        st.markdown("#### DateTime Variables")

        for col in datetime_cols:
            with st.expander(label=col):
                st.markdown("**SUMMARY**")
                date_stats = {
                    "Earliest Date": df[col].min().strftime("%Y-%m-%d"),
                    "Latest Date": df[col].max().strftime("%Y-%m-%d"),
                    "Date Range": str(df[col].max() - df[col].min()),
                    "Missing Values": str(df[col].isnull().sum()),
                }

                for key, value in date_stats.items():
                    st.metric(key, value)


def distribution_analysis(df: pd.DataFrame, columns: List[str]):
    """Advanced distribution analysis"""
    st.markdown("### Distribution Analysis")

    # Column selection for detailed analysis
    focus_column = st.selectbox("Focus Column", columns, key="dist_col")

    if not focus_column:
        return

    col1, col2 = st.columns(2)

    # Determine column type
    is_numeric = pd.api.types.is_numeric_dtype(df[focus_column])
    is_datetime = pd.api.types.is_datetime64_any_dtype(df[focus_column])

    with col1:
        if is_numeric:
            # Histogram with customization
            st.write("**Distribution Shape**")

            bins = st.slider("Number of Bins", 10, 100, 30, key="hist_bins")
            log_scaled = st.checkbox("Log Scale Y-Axis", key="log_scale")

            fig = px.histogram(
                df, x=focus_column, nbins=bins, title=f"Distribution of {focus_column}"
            )
            if log_scaled:
                fig.update_yaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)

        elif is_datetime:
            # Time-based histogram
            fig = px.histogram(
                df, x=focus_column, title=f"Temporal Distribution of {focus_column}"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Categorical bar chart
            value_counts = df[focus_column].value_counts().head(20)
            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation="h",
                title=f"Top 20 Values in {focus_column}",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if is_numeric:
            # Box plot and violin plot
            plot_type = st.radio(
                "Plot Type", ["Box Plot", "Violin Plot"], key="box_type"
            )

            if plot_type == "Box Plot":
                fig = px.box(df, y=focus_column, title=f"Box Plot of {focus_column}")
            else:
                fig = px.violin(
                    df, y=focus_column, title=f"Violin Plot of {focus_column}"
                )

            st.plotly_chart(fig, use_container_width=True)

        else:
            # Category analysis
            st.write("**Category Analysis**")

            # Frequency analysis
            total_categories = df[focus_column].nunique()

            st.metric("Total Categories", total_categories)

            # Category distribution
            if total_categories < 50:
                fig = px.pie(
                    values=df[focus_column].value_counts().values,
                    names=df[focus_column].value_counts().index,
                    title=f"Distribution of {focus_column}",
                )
                st.plotly_chart(fig, use_container_width=True)


def text_analysis(df: pd.DataFrame, columns: List[str]):
    """Advanced text analysis for speech data"""
    st.markdown("### Text Analysis")

    # find text columns
    text_cols = [col for col in columns if df[col].dtype == "object"]

    if not text_cols:
        st.info("No text columns found in selected variables")
        return

    selected_text_col = st.selectbox("Select Text Column", text_cols, key="text_col")

    if not selected_text_col:
        return

    # remove null values
    text_data = df[selected_text_col].dropna()

    if text_data.empty:
        st.warning("No text data available in selected column")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Text length analysis
        st.markdown("#### Text Length Analysis")

        # count characters
        text_lengths = text_data.astype(str).str.len()
        log_scale = st.checkbox(
            "Log Scale Y-Axis", value=False, key="log_scale_text_length"
        )

        fig = px.histogram(
            x=text_lengths,
            title=f"Text Length Distribution - {selected_text_col}",
            labels={"x": "Character Count", "y": "Frequency"},
        )
        if log_scale:
            fig.update_yaxes(type="log")

        st.plotly_chart(fig, use_container_width=True)

        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Avg Length", f"{text_lengths.mean():.0f} chars")
            st.metric("Min Length", f"{text_lengths.min()} chars")
        with col_stat2:
            st.metric("Max Length", f"{text_lengths.max()} chars")
            st.metric("Empty Entries", (text_data.str.len() == 0).sum())

    with col2:
        st.markdown("#### Word Analysis")

        # count words
        word_counts = text_data.astype(str).str.split().str.len()
        log_scale = st.checkbox(
            "Log Scale Y-Axis", value=False, key="log_scale_word_count"
        )

        fig = px.histogram(
            x=word_counts,
            title=f"Word Count Distribution - {selected_text_col}",
            labels={"x": "Word Count", "y": "Frequency"},
        )
        if log_scale:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

        col_word1, col_word2 = st.columns(2)
        with col_word1:
            st.metric("Avg Words", f"{word_counts.mean():.0f}")
            st.metric("Min Words", word_counts.min())
        with col_word2:
            st.metric("Max Words", word_counts.max())
            st.metric("Median Words", f"{word_counts.median():.0f}")


def advanced_analysis():
    """Advanced analysis with cross-table functionality"""
    st.markdown("## Advanced Analysis")

    if "db_tables" not in st.session_state:
        st.error("No tables loaded")
        return

    analysis_type = st.selectbox(
        "Analysis Type",
        [
            "Custom Filtering",
            "Speech Content Analysis",
            "Cross-Table Analysis",
        ],
    )

    if analysis_type == "Custom Filtering":
        custom_filtering_analysis()
    elif analysis_type == "Speech Content Analysis":
        speech_content_analysis()
    elif analysis_type == "Cross-Table Analysis":
        cross_table_analysis()


def cross_table_analysis():
    """Analysis across multiple tables"""
    st.markdown("## Cross-Table Analysis")

    # Select tables for cross-table analysis
    selected_tables = st.multiselect(
        "Select Tables",
        st.session_state["db_tables"],
        default=st.session_state["db_tables"][:2],
    )

    if len(selected_tables) < 2:
        st.info("Please select at least 2 tables for cross-table analysis")
        return

    # Simple join analysis (if common columns exist)
    for i, table1 in enumerate(selected_tables):
        for table2 in selected_tables[i + 1 :]:
            df1 = st.session_state[table1]
            df2 = st.session_state[table2]

            # Show sample merge
            join_col1, join_col2 = st.columns(2)
            with join_col1:
                join_col_left = st.selectbox(
                    f"Select Join Column for {table1}",
                    list(df1.columns),
                    index=None,
                    key=f"join_{table1}_{table2}_left",
                )
            with join_col2:
                join_col_right = st.selectbox(
                    f"Select Join Column for {table2}",
                    list(df2.columns),
                    index=None,
                    key=f"join_{table1}_{table2}_right",
                )

            if join_col_left and join_col_right:
                try:
                    merged = pd.merge(
                        df1.head(100),
                        df2,
                        left_on=join_col_left,
                        right_on=join_col_right,
                        how="inner",
                        suffixes=("_1", "_2"),
                    )
                    st.write(f"Sample merge result: {len(merged)} rows")
                    st.dataframe(merged.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Merge failed: {str(e)}")


def speech_content_analysis():
    """Specialized analysis for speech content"""
    st.markdown("### Speech Content Analysis")

    if "speeches" not in st.session_state:
        st.error("Speeches table not found")
        return

    speeches_df = st.session_state["speeches"]

    # Basic speech statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Speeches", len(speeches_df))

    with col2:
        if "speech_content" in speeches_df.columns:
            avg_length = speeches_df["speech_content"].str.len().mean()
            st.metric("Avg Speech Length", f"{avg_length:.0f} chars")

    with col3:
        if "date" in speeches_df.columns:
            date_range = speeches_df["date"].max() - speeches_df["date"].min()
            st.metric("Date Range", str(date_range).split(",")[0])

    # Speech analysis options
    analysis_option = st.selectbox(
        "Analysis Focus", ["Length Analysis", "Temporal Patterns"]
    )

    if analysis_option == "Length Analysis" and "speech_content" in speeches_df.columns:
        length_type = st.selectbox(
            "Select Length Type", ["Characters", "Words"], index=None, key="length_type"
        )

        if length_type == "Characters":
            speech_lengths = speeches_df["speech_content"].str.len()
        else:
            speech_lengths = speeches_df["speech_content"].str.split().str.len()

        fig = px.histogram(
            x=speech_lengths,
            nbins=50,
            title=f"Distribution of Speech Lengths (by {length_type})",
            labels={"x": length_type, "y": "Frequency"},
        )
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_option == "Temporal Patterns" and "date" in speeches_df.columns:
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(speeches_df["date"]):
            speeches_df["date"] = pd.to_datetime(speeches_df["date"])

        monthly_counts = speeches_df.groupby(pd.Grouper(key="date", freq="ME")).size()

        fig = px.line(
            x=monthly_counts.index,
            y=monthly_counts.values,
            title="Speeches per Month",
            labels={"x": "Date", "y": "Number of Speeches"},
        )
        st.plotly_chart(fig, use_container_width=True)


def custom_filtering_analysis():
    """Custom filtering and analysis"""

    st.markdown("### Custom Filtering Analysis")

    # Select table for custom filtering
    selected_table = st.selectbox(
        "Select Table for Custom Analysis",
        st.session_state["db_tables"],
        key="custom_table",
    )

    if not selected_table:
        return

    df = st.session_state[selected_table]

    # init session state for filters
    if "filter_state" not in st.session_state:
        st.session_state["filter_state"] = [{} for _ in range(3)]

    st.markdown("### Apply Filters")

    filters = {}

    col1, col2, col3 = st.columns(3)
    filter_columns = [col1, col2, col3]

    for i in range(3):
        with filter_columns[i]:
            st.markdown(f"**Filter {i+1}**")
            prev_col = st.session_state["filter_state"][i].get("filter_col", "None")
            filter_col = st.selectbox(
                f"Column",
                ["None"] + df.columns.tolist(),
                key=f"filter_col_{i}",
                index=(
                    (["None"] + df.columns.tolist()).index(prev_col)
                    if prev_col in ["None"] + df.columns.tolist()
                    else 0
                ),
            )
            st.session_state["filter_state"][i]["filter_col"] = filter_col

            if filter_col != "None":
                if pd.api.types.is_numeric_dtype(df[filter_col]):
                    prev_type = st.session_state["filter_state"][i].get(
                        "filter_type", "greater_than"
                    )
                    filter_type = st.selectbox(
                        f"Filter Type",
                        ["greater_than", "less_than", "equals", "between"],
                        key=f"filter_type_{i}",
                        index=(
                            ["greater_than", "less_than", "equals", "between"].index(
                                prev_type
                            )
                            if prev_type
                            in ["greater_than", "less_than", "equals", "between"]
                            else 0
                        ),
                    )
                    st.session_state["filter_state"][i]["filter_type"] = filter_type

                    if filter_type == "between":
                        prev_min = st.session_state["filter_state"][i].get(
                            "min_val", float(df[filter_col].min())
                        )
                        prev_max = st.session_state["filter_state"][i].get(
                            "max_val", float(df[filter_col].max())
                        )
                        min_val = st.number_input(
                            f"Min Value",
                            value=prev_min,
                            key=f"filter_min_{i}",
                        )
                        max_val = st.number_input(
                            f"Max Value",
                            value=prev_max,
                            key=f"filter_max_{i}",
                        )
                        st.session_state["filter_state"][i]["min_val"] = min_val
                        st.session_state["filter_state"][i]["max_val"] = max_val
                        filters[filter_col] = {
                            "type": filter_type,
                            "value": (min_val, max_val),
                        }
                    else:
                        prev_val = st.session_state["filter_state"][i].get(
                            "filter_val", float(df[filter_col].min())
                        )
                        filter_val = st.number_input(
                            "Value", value=prev_val, key=f"filter_val_{i}"
                        )
                        st.session_state["filter_state"][i]["filter_val"] = filter_val
                        filters[filter_col] = {"type": filter_type, "value": filter_val}

                else:
                    prev_type = st.session_state["filter_state"][i].get(
                        "filter_type", "equals"
                    )
                    filter_type = st.selectbox(
                        f"Filter Type",
                        ["equals", "contains"],
                        key=f"filter_type_{i}",
                        index=(
                            ["equals", "contains"].index(prev_type)
                            if prev_type in ["equals", "contains"]
                            else 0
                        ),
                    )
                    st.session_state["filter_state"][i]["filter_type"] = filter_type
                    prev_val = st.session_state["filter_state"][i].get("filter_val", "")
                    filter_val = st.text_input(
                        f"Value", value=prev_val, key=f"filter_val_{i}"
                    )
                    st.session_state["filter_state"][i]["filter_val"] = filter_val
                    if filter_val:
                        filters[filter_col] = {"type": filter_type, "value": filter_val}

    # Apply filters and show results
    # Clear filters button
    if st.button("Apply Filters"):
        st.session_state["last_filtered"] = {
            "filtered_df": filter_dataframe(df, filters=filters),
            "filters": filters,
        }
    if st.button("Clear All Filters"):
        st.session_state["filter_state"] = [{} for _ in range(3)]
        st.rerun()

    # Retain filtered results
    filtered_df = None
    if "last_filtered" in st.session_state:
        filtered_df = st.session_state["last_filtered"]["filtered_df"]
        st.dataframe(filtered_df.head(100), use_container_width=True)

        # Quick visualization
        if len(filtered_df) > 0:
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                viz_col = st.selectbox(
                    "Select Column for Quick Visualization",
                    numeric_cols,
                    index=None,
                    key=f"viz_col_{selected_table}",  # Stable key includes table name
                )
                if viz_col:
                    fig = px.histogram(
                        filtered_df,
                        x=viz_col,
                        title=f"Distribution of {viz_col} (Filtered)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
