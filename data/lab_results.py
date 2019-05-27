import pandas as pd
import argparse


def main(data_dir):

    usecols = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUE']
    labevents_features = pd.read_csv('./data/labevents_numerical_features.tsv', sep='\t',
                                       squeeze=True)

    labevents_df = pd.read_csv(f'{data_dir}/LABEVENTS.csv', usecols=usecols, parse_dates=[
        'CHARTTIME'])

    labevents_df = pd.merge(labevents_df, labevents_features, on='ITEMID').drop('ITEMID', axis=1)

    usecols = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']
    icustays_df = pd.read_csv(f'{data_dir}/ICUSTAYS.csv.gz', usecols=usecols, parse_dates=[
        'INTIME', 'OUTTIME'])

    processed_labevents = pd.merge(labevents_df, icustays_df, on=['SUBJECT_ID', 'HADM_ID'], how='outer')

    mask = processed_labevents['CHARTTIME'].between(processed_labevents['INTIME'], processed_labevents['OUTTIME'])
    processed_labevents = processed_labevents[mask]

    processed_labevents = processed_labevents.drop(['CHARTTIME', 'INTIME', 'OUTTIME'], axis=1)

    processed_labevents['VALUE'] = pd.to_numeric(processed_labevents['VALUE'], errors='coerce')

    processed_labevents = processed_labevents.groupby(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
                                                       'NAME']).agg(['mean', 'std'])
    processed_labevents = processed_labevents.unstack(level=3)

    processed_labevents.columns = ['_'.join(col[1:]).strip() for col in processed_labevents.columns.values]

    processed_labevents.to_csv('./data/interim/lab_results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/MIMIC/')
    args = parser.parse_args()

    main(args.data_dir)
