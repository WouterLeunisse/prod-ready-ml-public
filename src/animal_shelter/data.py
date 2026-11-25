"""Data loading and column-name normalization utilities for animal shelter data."""

import logging
import re

import pandas as pd

logger = logging.getLogger("animal_shelter")


def load_data(path: str) -> pd.DataFrame:
    """Load shelter data and normalize column names.

    Parameters
    ----------
    path : str
        Path to a CSV file.

    Returns
    -------
    pandas.DataFrame
        Data with parsed datetimes, normalized snake_case columns,
        and missing values filled with ``"Unknown"``.

    """
    logger.info(f"Loading data from {path}")

    df = (
        pd.read_csv(path, parse_dates=["DateTime"])
        .rename(columns=lambda x: x.replace("upon", "Upon"))
        .rename(columns=convert_camel_case)
        .fillna("Unknown")
    )

    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    logger.debug(f"Columns: {list(df.columns)}")

    return df


def convert_camel_case(name: str) -> str:
    """Convert a camelCase or PascalCase string to snake_case.

    Parameters
    ----------
    name : str
        Input column name.

    Returns
    -------
    str
        Column name converted to snake_case.

    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
