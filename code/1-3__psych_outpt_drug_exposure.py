'''
File to connect to SQL data based and read relvant table
Author: Amir Yazdavar
'''


import pandas as pd
PATH = '/home/ubuntu/amir/depression_sdh/depression_sdh/Joe_results/'

case_control = pd.read_csv(PATH+'cdrn_psych_casecontrol2.csv',delimiter="\t",
                      encoding='utf-8')

def outpt_drugs(df):
    de = pd.read_csv(PATH+'cdrn_drug_exposure_1218.csv', encoding='utf-8', names=['person_id', 'drug_exposure_start_date', 'concept_name', 'drug_class'],delimiter="\t")

    de = de.merge(df, on='person_id', how='inner')
    de1 = de[(de.drug_exposure_start_date >= de.one_year_date) & (de.drug_exposure_start_date < de.index_date)]

    de_rank = de1.groupby(['drug_class']).size().reset_index(name='counts').sort_values(by='counts',ascending=False).reset_index(drop=True)
    de_rank['cum_freq_pct'] = (de_rank['counts'].cumsum() / len(de1) * 100)
    de_rank = de_rank[de_rank['cum_freq_pct'] <= 95]

    meds = {"drug_class":[], "recode":[]}
    for _ in de_rank['drug_class']:
        if _.strip() in('MACROCRYSTALS 25 MG / Nitrofurantoin', 'USP 8.6 MG Oral Tablet'):
            continue
        else:
            meds['drug_class'].append(_)
            sub = _[0:16].casefold().strip().replace(" ", "_").replace("/", "").replace("-", "")
            meds['recode'].append('med_' + sub)

    meds = pd.DataFrame(meds)
    de2 = de1.merge(meds, on='drug_class', how='inner')
    de2['labels'] = de2.loc[:, 'recode']

    features = ['person_id', 'recode', 'labels']
    de2 = de2.loc[:, features]

    de3 = pd.pivot_table(de2, values='recode', index='person_id', columns='labels', aggfunc='count').fillna(0).reset_index()
    
    pd.DataFrame(meds).to_csv(PATH+'meds_data_dictionary.csv', encoding='utf-8') # saves the data dictionary
    pd.DataFrame(de3).to_csv(PATH+'psych_outpt_drug_exp_1218.csv', encoding='utf-8') # saves drug exposure df

outpt_drugs(case_control)
