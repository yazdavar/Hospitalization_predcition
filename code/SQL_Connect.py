'''
File to connect to SQL data based and read relvant table
Author: Amir Yazdavar
'''


import sys
sys.path.append("/Users/amir/OMOP_Hospital_Predict")
print(sys.path)

import psycopg2
import psycopg2.extras
import pprint


# /Users/amir/code/joe/OMOP_Hospital_Predict/Joe_results
def connect_DB():
    #Define our connection string
    # conn_string = "XXXX' dbname='XXXX' user='XXXX' password='XXXX'"
    conn_string = "host = vits-hpr-postgres dbname=mdd user=ahy4001 password=Ahy41$19"

    #print the connection string we will use to connect
    print("Connecting to database\n ->%s" % (conn_string))

    #get a connection, is a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)

    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor('cursor_unique_name', cursor_factory=psycopg2.extras.DictCursor)

    return cursor


if __name__ == "__main__":
    connect_DB()
    print ("curser created")
