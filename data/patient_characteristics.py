"""Get patient characteristics data."""
import pandas as pd
from pacmagic.general_utils.datetime_utils import calculate_timediff_in_hours
from pacmagic.general_utils.constants import DAYS_IN_YEAR, HOURS_IN_DAY
import argparse


def main(data_dir):
    # # All columns: ['ROW_ID', 'SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN',
    #               'EXPIRE_FLAG']

    usecols = ['SUBJECT_ID', 'GENDER', 'DOB']
    patients_df = pd.read_csv('/data/MIMIC/PATIENTS.csv.gz', usecols=usecols, parse_dates=['DOB'])

    # All columns: ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME',
    #                'DEATHTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION',
    #                'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION',
    #                'MARITAL_STATUS', 'ETHNICITY', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS',
    #                'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA']

    usecols = ['SUBJECT_ID', 'HADM_ID', 'ADMISSION_TYPE', 'ADMITTIME', 'HOSPITAL_EXPIRE_FLAG',
               'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'ETHNICITY', 'DIAGNOSIS']

    admissions_df = pd.read_csv(f'{data_dir}/ADMISSIONS.csv.gz', usecols=usecols,
                                parse_dates=['ADMITTIME'])

    admissions_df['emergency_admission_is_yes'] = admissions_df['ADMISSION_TYPE'] == 'EMERGENCY'

    patient_char_df = pd.merge(patients_df, admissions_df, on='SUBJECT_ID', how='outer')

    usecols = ['LOS', 'INTIME', 'SUBJECT_ID', 'ICUSTAY_ID', 'HADM_ID']
    icustays_df = pd.read_csv(f'{data_dir}/ICUSTAYS.csv.gz', usecols=usecols, parse_dates=['INTIME'])

    patient_char_df = pd.merge(patient_char_df, icustays_df, on=['SUBJECT_ID', 'HADM_ID'], how='outer')

    final_stays = patient_char_df.groupby('HADM_ID')['INTIME'].max()

    patient_char_df['FINAL_ICUSTAY'] = patient_char_df.set_index('HADM_ID')['INTIME'].isin(
        final_stays).values

    patient_char_df['EXPIRED_THIS_ICUSTAY'] = patient_char_df['FINAL_ICUSTAY'] & patient_char_df[
        'HOSPITAL_EXPIRE_FLAG']

    age_df = admissions_df.sort_values('ADMITTIME').drop_duplicates('SUBJECT_ID')

    age_df = pd.merge(age_df, patients_df, on='SUBJECT_ID')

    age_df['age'] = calculate_timediff_in_hours(age_df['ADMITTIME'], age_df['DOB']) / \
                    DAYS_IN_YEAR / HOURS_IN_DAY

    age_df = age_df[['SUBJECT_ID', 'age']]

    age_df[age_df['age'] < -200] = 90

    patient_char_df = pd.merge(patient_char_df, age_df, on='SUBJECT_ID', how='outer')

    patient_char_df = patient_char_df.drop(['DOB'], axis=1)

    # Columns: ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'DBSOURCE',
    #           'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID',
    #           'INTIME', 'OUTTIME', 'LOS']


    patient_char_df['time_at_hosp_pre_ic_admission'] = calculate_timediff_in_hours(
        patient_char_df['INTIME'], patient_char_df['ADMITTIME']
    )

    columns_to_keep = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'EXPIRED_THIS_ICUSTAY',
                       'GENDER','age', 'LOS', 'time_at_hosp_pre_ic_admission', 'ADMISSION_TYPE',
                       'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'ETHNICITY', 'DIAGNOSIS']


    # columns_to_keep = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'EXPIRED_THIS_ICUSTAY',
    #                    'GENDER', 'LOS', 'time_at_hosp_pre_ic_admission', 'ADMISSION_TYPE',
    #                    'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'ETHNICITY', 'DIAGNOSIS']

    patient_char_df = patient_char_df[columns_to_keep]

    print(patient_char_df.shape)

    print(patient_char_df['ADMISSION_TYPE'].value_counts())

    patient_char_df.to_csv('./data/interim/patient_characteristics.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/MIMIC/')
    args = parser.parse_args()
    main(args.data_dir)
