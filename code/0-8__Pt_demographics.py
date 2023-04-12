'''
File to connect to SQL data based and read relvant table
Author: Amir Yazdavar
'''

import io
import sys
sys.path.append("/Users/amir/code/OMOP_Hospital_Predict")
print(sys.path)


import io
import SQL_Connect

def main():
        cursor = SQL_Connect.connect_DB()

        output_file = '/Users/amir/code/OMOP_Hospital_Predict/Joe_results/cdrn_pt_demographics.csv'

# Selects all patients in the database and reports their gender and birthdate
        sql = "select distinct person_id, gender_concept_id, make_date(year_of_birth, month_of_birth, day_of_birth) as birth_date from staging.person union select distinct person_id, gender_concept_id, make_date(year_of_birth, month_of_birth, day_of_birth) as birth_date from mdd_control.person"

        cursor.execute(sql)

        with io.open(output_file, 'w', encoding='utf-8') as f:
            for index, row in enumerate(cursor):
                print(index)
                person_id = str(row[0])
                gender_concept_id = str(row[1])
                birth_date = str(row[2])

                line = '\t'.join([
                    person_id, gender_concept_id, birth_date])

                f.write(line + '\n')

        print("Finished Reading!")

if __name__=='__main__':
    main()
