'''
File to connect to SQL data based and read relvant table
Author: Amir Yazdavar
'''


import pandas as pd
import numpy as np
from datetime import datetime
import datetime as dt
from collections import defaultdict
import logging
import sys
import time

# SET LOGGER
logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

# ESTABLISH A DIRECTORY PATH
PATH = '/home/ubuntu/amir/depression_sdh/depression_sdh/Joe_results/'


# IMPORT DATA
hosp = pd.read_csv(PATH+'cdrn_psych_hosp_bef_18.csv', encoding='utf-8', names=['person_id', 'inpt_hosp_date', 'visit_occurrence_id', 'visit_concept_id', 'condition_type_concept_id'],delimiter="\t") # cdrn_psych_primary_hosp_bef_18.csv
diagnosis = pd.read_csv(PATH+'cdrn_psych_dx_bef_18.csv', encoding='utf-8', names=['person_id', 'psych_dx_date'],delimiter="\t")
visits = pd.read_csv(PATH+'cdrn_all_visits_1218.csv', encoding='utf-8', names=['person_id', 'visit_occurrence_id', 'visit_start_date', 'visit_concept_id'],delimiter="\t")
no_hosp = pd.read_csv(PATH+'cdrn_no_psych_hosp_bef_18.csv', encoding='utf-8', names=['person_id', 'condition_start_date', 'condition_source_value'],delimiter="\t")
demographics = pd.read_csv(PATH+'cdrn_pt_demographics.csv', encoding='utf-8', names=['person_id', 'gender_concept_id', 'birth_date'],delimiter="\t")


# CALCULATE DAYS BETWEEN X-Y
def days_between(d1, d2):
    d1 = datetime.strptime(d1, '%Y-%m-%d')
    d2 = datetime.strptime(d2, '%Y-%m-%d')
    return abs((d2 - d1).days)

# CREATE A CONTROL DF
def control_find(w, x, z):
    #input
    # w:demographics, x:no_hosp, z:visits

    # select 1 random visit (w/ psych dx) from each patient
    control = x.groupby('person_id').apply(lambda y: y.sample(n=1, random_state=1).reset_index(drop=True))

    # Reset the Index
    control = control.reset_index(drop=True)

    #nake the time series frm the df
    control['condition_start_date'] = pd.to_datetime(control['condition_start_date'])

    # print (   )
    # sys.exit()

    # establish index date based on psych dx date
    dates = {"visit_win_start":[],"visit_win_end":[]}
    for _ in control['condition_start_date']:
        if _ < dt.datetime(2013,1,1,00,00,00):
            delta = (dt.datetime(2013, 1, 1, 00, 00, 00) - _).days
            if delta == 366:
                d2 = dt.datetime(2013, 1, 1, 00, 00, 00)
                d1 = d2 - dt.timedelta(366)
                dates["visit_win_end"].append(d2)
                dates["visit_win_start"].append(d1)
            else:
                random = np.random.randint(delta, 366)
                d2 = _ + np.timedelta64(random,'D')
                d1 = d2 - np.timedelta64(366, 'D')
                dates["visit_win_end"].append(d2)
                dates["visit_win_start"].append(d1)
        elif dt.datetime(2013,1,1,00,00,00) <= _ <= dt.datetime(2016,12,31,00,00,00):
            random = np.random.randint(0, 366)
            d2 = _ + np.timedelta64(random, 'D')
            d1 = d2 - np.timedelta64(366, 'D')
            dates["visit_win_end"].append(d2)
            dates["visit_win_start"].append(d1)
        else:
            delta = (dt.datetime(2017, 12, 31, 00, 00, 00) - _).days # update to 12.31.2017 with larger sample size
            if delta == 0:
                d2 = _
                d1 = d2 - np.timedelta64(366, 'D')
                dates["visit_win_end"].append(d2)
                dates["visit_win_start"].append(d1)
            else:
                random = np.random.randint(0, delta)
                d2 = _ + np.timedelta64(random, 'D')
                d1 = d2 - np.timedelta64(366, 'D')
                dates["visit_win_end"].append(d2)
                dates["visit_win_start"].append(d1)

    window = pd.DataFrame(dates)
    control_2 = pd.merge(control, window, left_index=True, right_index=True)

    control_3 = control_2.merge(z, on='person_id', how='left')
    control_3 = control_3.sort_values(by=['person_id', 'visit_win_end'])
    control_3['visit_start_date'] = pd.to_datetime(control_3['visit_start_date'])
    control_3 = control_3[(control_3.visit_start_date >= control_3.visit_win_start) & (control_3.visit_start_date < control_3.visit_win_end)]

    control_4 = control_3.groupby('person_id').visit_start_date.agg(['first', 'last'])
    control_4 = control_4.reset_index()
    first_last = {"first":[], "last":[]}
    for i in control_4['first']:
        date1 = pd.to_datetime(i)
        date2 = date1.date()
        first_last["first"].append(date2)
    for i in control_4['last']:
        date1 = pd.to_datetime(i)
        date2 = date1.date()
        first_last["last"].append(date2)
    first_last = pd.DataFrame(first_last)
    control_4_ = pd.merge(control_4, first_last, left_index=True, right_index=True)
    control_4_['date_diff'] = control_4_.apply(lambda row: days_between(str(row['first_y']), str(row['last_y'])), axis=1)

    control_v = control_3.groupby('person_id').visit_occurrence_id.agg('count')
    control_v = control_v.reset_index()
    control_v.rename(columns={'visit_occurrence_id': 'visit_count'}, inplace=True)

    control_5 = control_4_.merge(control_2, on='person_id', how='left')
    control_5 = control_5.merge(control_v, on='person_id', how='left')
    control_5.rename(columns={'visit_win_end': 'index_date', 'visit_win_start':'one_year_date'}, inplace=True)

    control_6 = control_5.merge(w, on='person_id', how='inner')
    control_6['age'] = control_6.apply(lambda row: round((days_between(str(row['last_y']), str(row['birth_date'])) / 365)),
                                   axis=1)

    features = ['person_id', 'age', 'gender_concept_id', 'index_date', 'one_year_date', 'date_diff', 'visit_count']
    control_6 = control_6.loc[:,features]
    control_6['enc'] = 99
    print("Number of controls: {}".format(len(control_6)))

    return control_6

# CREATE A CASE DF
def case_find(w, x, y, z):
    # Creates var 'hosp_num': successive count of psych hosp per 'person_id' (ascending order by date)
    # Identifies the first psych hospitalization, but is not necessarily the first psych hosp ever
    x['hosp_num'] = x.groupby('person_id').inpt_hosp_date.apply(pd.Series.rank)

    # Creates new df where only the first psychiatric hospitalization 2013-2017 is retained
    df = x[x['hosp_num'] < 2.0] # if >1 hosp on a single day, 1 < hosp_num < 2
    df = df.drop_duplicates(subset='person_id', keep='first')
    df = df[df['inpt_hosp_date'] > '2012-12-31'] # selects only those patients w/ first hosp after 2012


    df = df.merge(y, on='person_id', how='inner')
    df['one_year_date'] = (df['inpt_hosp_date'].values.astype('datetime64[D]') - np.timedelta64(366, 'D'))
    cases = df[(df.psych_dx_date.values.astype('datetime64[D]') >= df.one_year_date.values.astype('datetime64[D]')) & (
                df.psych_dx_date.values.astype('datetime64[D]') < df.inpt_hosp_date.values.astype('datetime64[D]'))]
    cases = cases.drop_duplicates(subset='person_id', keep='first')

    cases_2 = cases.merge(z, on='person_id', how='left')
    cases_2 = cases_2.sort_values(by=['person_id', 'visit_start_date'])
    cases_2 = cases_2[(cases_2.visit_start_date.values.astype('datetime64[D]') >= cases_2.one_year_date.values.astype('datetime64[D]')) & (
            cases_2.visit_start_date.values.astype('datetime64[D]') < cases_2.inpt_hosp_date.values.astype('datetime64[D]'))]

    cases_3 = cases_2.groupby('person_id').visit_start_date.agg(['first', 'last'])
    cases_3 = cases_3.reset_index()
    cases_3['date_diff'] = cases_3.apply(lambda row: days_between(str(row['first']),str(row['last'])), axis=1)

    cases_v = cases_2.groupby('person_id').visit_occurrence_id_y.agg('count')
    cases_v = cases_v.reset_index()
    cases_v.rename(columns = {'visit_occurrence_id_y':'visit_count'}, inplace=True)


    cases_4 = cases_3.merge(cases, on='person_id', how='left')
    cases_4 = cases_4.merge(cases_v, on='person_id', how='left')
    cases_4['inpt_hosp_date'] = cases_4['inpt_hosp_date'].values.astype('datetime64[D]')
    cases_4.rename(columns={'inpt_hosp_date': 'index_date'}, inplace=True)


    cases_5 = cases_4.merge(w, on='person_id', how='inner')
    cases_5['age'] = cases_5.apply(lambda row: round((days_between(str(row['last']), str(row['birth_date']))/365)), axis=1)
    cases_5['enc'] = cases_5.apply(lambda row: 1 if row.condition_type_concept_id == 44786627 else 2, axis=1)

    features = ['person_id', 'age', 'gender_concept_id', 'index_date', 'one_year_date', 'date_diff', 'visit_count', 'enc']
    cases_5 = cases_5.loc[:,features]

    print("Number of cases: {}".format(len(cases_5)))

    return cases_5

# RECODE GENDER VARIABLES TO BINARY
def gender_var(df):
    person_list = []
    for _ in df['person_id']:
        person_list.append(_)

    gender_list = []
    for _ in df['gender_concept_id']:
        if _ == 8532:
            gender_list.append(1)
        else: gender_list.append(0)

    person_gender = pd.DataFrame({"person_id":person_list,"sex":gender_list})
    df = pd.merge(df,person_gender, how='left', on='person_id')

    df = df.drop(columns='gender_concept_id')

    return df

# JOIN CASE-CONTROL DFs
def case_control(df1, df2):

    df3 = df1.append(df2, ignore_index=True)

    df4 = gender_var(df3)

    pd.DataFrame(df4).to_csv(PATH+'cdrn_psych_casecontrol2.csv', encoding='utf-8',sep = "\t")

    return df4



controls = control_find(demographics, no_hosp, visits)

cases = case_find(demographics, hosp, diagnosis, visits)

cdrn_psych = case_control(cases, controls)
