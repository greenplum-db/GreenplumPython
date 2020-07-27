import pandas as pd
from greenplumpython.core.gpdatabase import GPDatabase
from greenplumpython.core.dataframe_wrapper import DataFrameWrapper
from greenplumpython.core.gptable_metadata import GPTableMetadata
import inspect
from greenplumpython.utils.apply_utils import *

def pythonExec(df, funcName, typeName, index, output, extra_args):
    func_type_name = []
    internal_select = []
    for i, col in enumerate(df.table_metadata.signature):
        for j, column in enumerate(col):
            internal_select.append("array_agg(" + column + ") AS " + column)
            func_type_name.append(column)

    for key, value in extra_args.items(): 
        func_type_name.append(str(value[0])) 
    select_func = ""
    joined_type_name = ",".join(func_type_name)
    if output.name == None or output.name == "":
        select_func = "WITH gpdbtmpa AS (\nSELECT (%s(%s)) AS gpdbtmpb FROM (SELECT %s FROM %s GROUP BY %s) tmptbl\n)\nSELECT (gpdbtmpb::%s).* FROM gpdbtmpa;" % (funcName, joined_type_name, ",".join(internal_select), df.table_metadata.name, index, typeName)
    else:
        if output.case_sensitive:
            output_name = '"'+output.name+'"'
        else:
            output_name = output.name
        select_func = "CREATE TABLE " + output_name + " AS \n" \
            + "WITH gpdbtmpa AS ( \n" \
            + "SELECT (" + funcName + "(" + joined_type_name +")) AS gpdbtmpb FROM (SELECT " \
            + ",".join(internal_select) + " FROM " + df.table_metadata.name + " GROUP BY " + index + ") tmptbl \n ) \n" \
            + "SELECT (gpdbtmpb::" + typeName + ").* FROM gpdbtmpa " + output.distribute_on_str + ";"
    return select_func

def gptApply(dataframe, index, py_func, db, output, clear_existing = True, runtime_id = 'plc_python', runtime_type = 'plpythonu', **kwargs):
    if py_func == None:
        raise ValueError("No input function provided")
    if callable(py_func) == False:
        raise ValueError("Wrong input function provided")

    s = inspect.getsource(py_func)
    function_name = randomString()
    typeName = randomStringType()
    params = []
    columns = inspect.getfullargspec(py_func)[0]

    if dataframe == None:
        raise ValueError("No input dataframe provided")

    for i, col in enumerate(dataframe.table_metadata.signature):
        for j, column in enumerate(col):
            params.append(column+" "+col[column] + "[]")
    
    rest_args_num = len(columns) - len(params)
    args_index = 0 - rest_args_num

    for key, value in kwargs.items(): 
        params.append(columns[args_index] + " " + str(value[1]))
        args_index = args_index + 1

    if output == None or output.signature == None:
        raise ValueError("Output.signature must be provided")

    if index == None or index == "":
        raise ValueError("Groupby index must be provided")

    create_type_sql = createTypeFunc(output.signature, typeName)
    runtime_id_str = ''
    if runtime_type == 'plcontainer':
        runtime_id_str = '# container: %s' % (runtime_id)
    function_body = "CREATE OR REPLACE FUNCTION %s(%s) RETURNS %s AS $$\n%s\n%s\nreturn %s(%s) $$ LANGUAGE %s;" % (function_name,",".join(params),typeName,runtime_id_str,s,py_func.__name__,",".join(columns),runtime_type)
    select_sql = pythonExec(dataframe, function_name, typeName, index, output, kwargs)
    drop_sql = "DROP TYPE " + typeName + " CASCADE;"

    if db == None:
        raise ValueError("No database connection provided")

    try:
        with db.run_transaction() as trans:
            if clear_existing and output.name != None and output.name != "":
                if output.case_sensitive:
                    output_name = '"'+output.name+'"'
                else:
                    output_name = output.name
                drop_table_sql = "drop table if exists %s;" % output.name
                trans.execute(drop_table_sql)
            trans.execute(create_type_sql)
            trans.execute(function_body)
            res = None
            if output.name == None or output.name == "":
                res = db.execute_transaction_query(trans, select_sql)
            else:
                trans.execute(select_sql)
            trans.execute(drop_sql)

    except Exception as e:
        raise e

    return res
