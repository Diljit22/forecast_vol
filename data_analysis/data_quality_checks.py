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

def describe_df(df: pd.DataFrame, ticker_col: str = 'ticker'):
    """
    Prints the following metrics for a DataFrame df:
    1. Number of rows grouped by ticker_col (only if ticker_col exists).
    2. Number of NaNs per column (only if > 0). Prints 'No NaNs' if all zeros.
    3. Shape of the entire DataFrame.
    4. Shape of each group (only if ticker_col exists).
    5. Datatype of each column.
    """
    print("======= DataFrame Description =======")

    # 1) and 3) Shape of entire df
    df_shape = df.shape
    print(f"DataFrame shape: {df_shape}")

    # 2) Count NaNs
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()

    # If the ticker_col is present, group by it
    if ticker_col in df.columns:
        group_sizes = df.groupby(ticker_col).size()
        
        print("\nNumber of rows by ticker_col:")
        for ticker_value, size in group_sizes.items():
            print(f"  {ticker_value}: {size}")

        # 4) Shape of each group
        print("\nShape of each group (ticker_col -> (rows, cols)):")
        for ticker_value, group_data in df.groupby(ticker_col):
            print(f"  {ticker_value}: {group_data.shape}")
    else:
        print(f"\nColumn '{ticker_col}' not found. Skipping group-by statistics.")

    # 2) Print columns with NaNs if any
    print("\nNaN counts (only showing columns with NaNs):")
    if total_nans == 0:
        print("  No NaNs detected in any column.")
    else:
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count}")

    # 5) Datatypes
    print("\nColumn data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    print("=====================================\n")

