'''
File to connect to SQL data based and read relvant table
Author: Amir Yazdavar
'''


import io
import sys
sys.path.append("/Users/amir/code/joe/OMOP_Hospital_Predict")
print(sys.path)
import io
import SQL_Connect

def main():
        cursor = SQL_Connect.connect_DB()

        output_file = '/Users/amir/code/joe/OMOP_Hospital_Predict/Joe_results/cdrn_all_visits_1218.csv'

# Selects all patient diagnoses that match the inclusion criteria: visit in 2012, 2013, 2014
        sql = "select vi.person_id, vi.visit_occurrence_id, vi.visit_start_date, vi.visit_concept_id from staging.visit_occurrence as vi where vi.visit_start_date between '2012-01-01' and '2017-12-31' union all select vi.person_id, vi.visit_occurrence_id, vi.visit_start_date, vi.visit_concept_id from mdd_control.visit_occurrence as vi where vi.visit_start_date between '2012-01-01' and '2017-12-31'"

        cursor.execute(sql)

        with io.open(output_file, 'w', encoding='utf-8') as f:
            for index, row in enumerate(cursor):
                print(index)
                person_id = str(row[0])
                visit_occurrence_id = str(row[1])
                visit_start_date = str(row[2])
                visit_concept_id = str(row[3])

                line = '\t'.join(
                    [person_id, visit_occurrence_id, visit_start_date, visit_concept_id])

                f.write(line + '\n')

        print("Finished Reading!")

if __name__=='__main__':
    main()
