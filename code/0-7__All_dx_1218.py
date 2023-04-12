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

        output_file = '/Users/amir/code/joe/OMOP_Hospital_Predict/Joe_results/cdrn_all_individual_dx_1218.csv'

# Selects all patient diagnoses between 2012-2017
        sql = "select co.person_id, co.condition_start_date, co.condition_source_value from staging.condition_occurrence as co where co.condition_start_date between '2012-01-01' and '2017-12-31' union all select co.person_id, co.condition_start_date, co.condition_source_value from mdd_control.condition_occurrence as co where co.condition_start_date between '2012-01-01' and '2017-12-31'"

        cursor.execute(sql)

        with io.open(output_file, 'w', encoding='utf-8') as f:
            for index, row in enumerate(cursor):
                print(index)
                person_id = str(row[0])
                condition_start_date = str(row[1])
                condition_source_value = str(row[2])

                line = '\t'.join(
                    [person_id, condition_start_date, condition_source_value])

                f.write(line + '\n')

        print("Finished Reading!")

if __name__=='__main__':
    main()
