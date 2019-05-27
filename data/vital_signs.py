import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
import argparse

pbar = ProgressBar().register()


def convert_dask_columns_to_numeric(dd_df: dd.DataFrame,
                                    columns: list,
                                    numeric_dtype: str = 'float',
                                    errors: str = 'coerce') -> dd.DataFrame:
    """Convert one or more columns in a dask dataframe to a numeric format.

    Parameters
    ----------
    dd_df : dd.DataFrame
        Dask dataframe containing the columns to convert.
    columns : list
        List of column names to convert.
    numeric_dtype : str, optional
        Desired datatype of the resulting columns.
    errors : str, optionl
        If 'raise', then invalid parsing will raise an exception -
        If 'coerce', then invalid parsing will be set as NaT - If 'ignore', then invalid parsing
        will return the input.

    Returns
    -------
    type : dd.DataFrame
        Dask dataframe with converted columns.
    """
    dd_out = dd_df.copy()
    for col in columns:
        dd_out[col] = dd_out[col].map_partitions(pd.to_numeric, errors=errors, meta=numeric_dtype)
    return dd_out


def main(path_to_mimic_data):
    usecols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'VALUE']
    chartevents_df = dd.read_csv(f'{path_to_mimic_data}/CHARTEVENTS.csv', usecols=usecols,
                                               dtype={'ICUSTAY_ID': 'float64',
                                                     'VALUE': 'object'})

    chartevents_features = pd.read_csv('./data/chartevents_numerical_features.tsv', sep='\t',
                                       squeeze=True)

    chartevents_df = dd.merge(chartevents_df, chartevents_features, on='ITEMID').drop('ITEMID',
                                                                                      axis=1)
    chartevents_df = convert_dask_columns_to_numeric(chartevents_df, ['VALUE'])

    chartevents_df = chartevents_df.groupby(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
                                             'NAME']).agg(['mean', 'std']).compute()

    chartevents_df = chartevents_df.unstack(level=3)

    chartevents_df.columns = ['_'.join(col[1:]).strip() for col in chartevents_df.columns.values]

    for column in chartevents_df.columns:
        if 'arterial_bp' in column:
            combined_name = column.replace('arterial_bp', 'combined_bp')
            ni_name = column.replace('arterial_bp', 'ni_bp')
            chartevents_df[combined_name] = chartevents_df[column].fillna(chartevents_df[ni_name])

            chartevents_df = chartevents_df.drop(ni_name, axis=1)

    print(chartevents_df.shape)
    chartevents_df.to_csv('./data/interim/vital_signs.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/data/MIMIC/', type=str)
    args = parser.parse_args()
    main(args.data_path)
