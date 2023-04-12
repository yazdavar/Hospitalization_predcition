'''
File to connect to SQL data based and read relvant table
Author: Amir Yazdavar
'''


import io
import sys
sys.path.append("/Users/amir/code/joe/OMOP_Hospital_Predict")
print(sys.path)
import SQL_Connect


def main():
        cursor = SQL_Connect.connect_DB()

        output_file = '/Users/amir/code/joe/OMOP_Hospital_Predict/Joe_results/cdrn_psych_dx_bef_18.csv'

# Selects all patients in the cohort and reports their gender, birth date, and age at '2014-01-01'
        sql = "select v.person_id, v.visit_start_date from staging.visit_occurrence as v join staging.condition_occurrence as co on (v.visit_occurrence_id = co.visit_occurrence_id and v.person_id = co.person_id) where (v.visit_concept_id != 9201) and ((co.condition_source_value like 'ICD9CM:293.81%') or (co.condition_source_value like 'ICD9CM:293.82%') or (co.condition_source_value like 'ICD9CM:295%') or (co.condition_source_value like 'ICD9CM:296%') or (co.condition_source_value like 'ICD9CM:297%') or (co.condition_source_value like 'ICD9CM:298%') or (co.condition_source_value like 'ICD9CM:300.4%') or (co.condition_source_value like 'ICD9CM:311%') or (co.condition_source_value like 'ICD10CM:F06.0%') or (co.condition_source_value like 'ICD10CM:F06.2%') or (co.condition_source_value like 'ICD10CM:F20%') or (co.condition_source_value like 'ICD10CM:F21%') or (co.condition_source_value like 'ICD10CM:F22%') or (co.condition_source_value like 'ICD10CM:F23%')  or (co.condition_source_value like 'ICD10CM:F24%') or (co.condition_source_value like 'ICD10CM:F25%') or (co.condition_source_value like 'ICD10CM:F28%') or (co.condition_source_value like 'ICD10CM:F29%') or (co.condition_source_value like 'ICD10CM:F30%') or (co.condition_source_value like 'ICD10CM:F31%') or (co.condition_source_value like 'ICD10CM:F32%') or (co.condition_source_value like 'ICD10CM:F33%') or (co.condition_source_value like 'ICD10CM:F34.1%') or (co.condition_source_value like 'ICD10CM:F34.8%') or (co.condition_source_value like 'ICD10CM:F39%')) and co.condition_type_concept_id in(44786627, 44786629) and (v.visit_start_date < '2018-01-01') union select v.person_id, v.visit_start_date from mdd_control.visit_occurrence as v join mdd_control.condition_occurrence as co on (v.visit_occurrence_id = co.visit_occurrence_id and v.person_id = co.person_id) where (v.visit_concept_id != 9201) and ((co.condition_source_value like 'ICD9CM:293.81%') or (co.condition_source_value like 'ICD9CM:293.82%') or (co.condition_source_value like 'ICD9CM:295%') or (co.condition_source_value like 'ICD9CM:296%') or (co.condition_source_value like 'ICD9CM:297%') or (co.condition_source_value like 'ICD9CM:298%') or (co.condition_source_value like 'ICD9CM:300.4%') or (co.condition_source_value like 'ICD9CM:311%') or (co.condition_source_value like 'ICD10CM:F06.0%') or (co.condition_source_value like 'ICD10CM:F06.2%') or (co.condition_source_value like 'ICD10CM:F20%') or (co.condition_source_value like 'ICD10CM:F21%') or (co.condition_source_value like 'ICD10CM:F22%') or (co.condition_source_value like 'ICD10CM:F23%')  or (co.condition_source_value like 'ICD10CM:F24%') or (co.condition_source_value like 'ICD10CM:F25%') or (co.condition_source_value like 'ICD10CM:F28%') or (co.condition_source_value like 'ICD10CM:F29%') or (co.condition_source_value like 'ICD10CM:F30%') or (co.condition_source_value like 'ICD10CM:F31%') or (co.condition_source_value like 'ICD10CM:F32%') or (co.condition_source_value like 'ICD10CM:F33%') or (co.condition_source_value like 'ICD10CM:F34.1%') or (co.condition_source_value like 'ICD10CM:F34.8%') or (co.condition_source_value like 'ICD10CM:F39%')) and co.condition_type_concept_id in(44786627, 44786629) and (v.visit_start_date < '2018-01-01')"

        cursor.execute(sql)

        with io.open(output_file, 'w', encoding='utf-8') as f:
            for index, row in enumerate(cursor):
                print(index)
                person_id = str(row[0])
                visit_start_date = str(row[1])

                line = '\t'.join([
                    person_id, visit_start_date])

                f.write(line + '\n')

        print("Finished Reading!")

if __name__=='__main__':
    main()
