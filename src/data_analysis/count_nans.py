import pandas as pd


def count_nans_per_column(df: pd.DataFrame) -> dict:
    """
    Prints and returns the number of NaN values for each column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to inspect.

    Returns
    -------
    dict
        A dictionary mapping each column name to the count of NaN values.
    """
    # pandas Series of NaN counts per column
    nan_counts_series = df.isna().sum()

    # Convert that Series to a dictionary
    nan_counts_dict = nan_counts_series.to_dict()

    # Print results
    print("NaN counts per column:")
    for col, count in nan_counts_dict.items():
        print(f"  {col}: {count}")

    # Return the dictionary for further use
    return nan_counts_dict
