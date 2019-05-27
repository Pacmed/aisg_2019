import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def dummify(df: pd.DataFrame,
            features_to_dummify,
            drop_first: bool = True,
            dummy_na: bool = False):
    """Create dummy variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe of features
    features_to_dummify : list[str]
        List of feature on which to apply the dummification
    drop_first : bool, optional
        Set to True to apply one-hot-encoding (do not drop first category)
    dummy_na : bool, optional
        Whether to dummify NaN as their own class. If False, ignore them.

    Returns
    -------
    type: Tuple[pd.DataFrame, List[str]]

    """
    prefixes = [x for x in features_to_dummify]

    df_dummies = pd.get_dummies(df[features_to_dummify], columns=features_to_dummify,
                                prefix=prefixes, drop_first=drop_first, dummy_na=dummy_na)

    df = pd.concat([df, df_dummies], axis=1, sort=False)

    return df, df_dummies.columns.tolist()


def remove_outliers(df, ctu, n_iqrs=8):
    print(f"Length before removing outliers using {n_iqrs} interquartile ranges: {len(df)}")
    df_use = df[ctu]
    q1 = df_use.quantile(0.25)
    q3 = df_use.quantile(0.75)
    iqr = q3 - q1
    df = df[~((df_use < (q1 - n_iqrs * iqr)) | (df_use > (q3 + n_iqrs * iqr))).any(axis=1)]
    print(f"Length after: {len(df)}")
    return df


def main(path_to_interim_data, path_to_processed_data):

    lab_results = pd.read_csv(f'{path_to_interim_data}/lab_results.csv')
    vital_signs = pd.read_csv(f'{path_to_interim_data}/vital_signs.csv')
    patient_characteristics = pd.read_csv(f'{path_to_interim_data}/patient_characteristics.csv')

    df_model = pd.merge(lab_results, patient_characteristics,
                        on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])
    df_model = pd.merge(df_model, vital_signs, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])

    dummy_cols = ['GENDER']

    df_model, dummified = dummify(df_model, dummy_cols)
    df_model = df_model.drop(dummy_cols, axis=1)

    print(df_model.columns)
    ctu = df_model.drop(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'EXPIRED_THIS_ICUSTAY',
                         'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'ETHNICITY',
                         'DIAGNOSIS'],
                        axis=1).columns

    df_newborns = df_model[df_model['ADMISSION_TYPE'] == 'NEWBORN']
    df_model = df_model[df_model['ADMISSION_TYPE'] != 'NEWBORN']

    df_model = remove_outliers(df_model, ctu)

    # Custom cleaning.

    df_model = df_model[(df_model[['mean_combined_bp_dia', 'mean_combined_bp_sys']] > 10).all(axis=1)]

    df_model = df_model[df_model['time_at_hosp_pre_ic_admission'] > 0]

    print(df_model.min())

    print(df_model.max())

    print(df_model.shape)

    for col in list(ctu) + ['EXPIRED_THIS_ICUSTAY']:

        if 'std' in col:
            df_model[col] = df_model[col].fillna(0)

        df_model[col] = pd.to_numeric(df_model[col], errors='raise')

    X_train_raw, X_test_raw = train_test_split(df_model, test_size=0.1)

    y_train = X_train_raw['EXPIRED_THIS_ICUSTAY']
    y_test = X_test_raw['EXPIRED_THIS_ICUSTAY']

    X_train_processed = X_train_raw[ctu]
    X_test_processed = X_test_raw[ctu]

    imputer = SimpleImputer(strategy='median')
    scaler = MinMaxScaler()

    X_train_processed.loc[:] = imputer.fit_transform(X_train_processed)
    X_test_processed.loc[:] = imputer.transform(X_test_processed)

    X_train_processed.loc[:] = scaler.fit_transform(X_train_processed)
    X_test_processed.loc[:] = scaler.transform(X_test_processed)

    assert X_train_processed.isna().sum().sum() == 0
    assert X_test_processed.isna().sum().sum() == 0

    assert y_train.isna().sum().sum() == 0
    assert y_test.isna().sum().sum() == 0

    X_train_processed.to_csv(f'{path_to_processed_data}/X_train_processed.csv', index=False)
    X_test_processed.to_csv(f'{path_to_processed_data}/X_test_processed.csv', index=False)

    X_train_raw.to_csv(f'{path_to_processed_data}/X_train_raw.csv', index=False)
    X_test_raw.to_csv(f'{path_to_processed_data}/X_test_raw.csv', index=False)

    y_train.to_csv(f'{path_to_processed_data}/y_train.csv', index=False, header=True)
    y_test.to_csv(f'{path_to_processed_data}/y_test.csv', index=False, header=True)

    y_newborns = df_newborns['EXPIRED_THIS_ICUSTAY']

    X_newborns = df_newborns[X_train_processed.columns]
    X_newborns.loc[:] = imputer.transform(X_newborns)
    X_newborns.loc[:] = scaler.transform(X_newborns)

    y_newborns.to_csv(f'{path_to_processed_data}/y_newborns.csv', index=False, header=True)
    X_newborns.to_csv(f'{path_to_processed_data}/X_newborns.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interim_path', default='./data/interim', type=str)
    parser.add_argument('--processed_path', default='./data/processed', type=str)
    args = parser.parse_args()
    main(args.interim_path, args.processed_path)