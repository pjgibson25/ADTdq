# Import[ant] libaries @('_')@
import hl7
import pandas as pd
import numpy as np
import regex as re
import os
import math
import time
import datetime

# Plotting!
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors

# Plotly--yeet
import plotly.figure_factory as ff
import plotly
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Clears display if you need to manual loop
from IPython.display import clear_output

# For accessing supporting docs within supporting folder
import pkg_resources





###################################################

def NoError(func, *args, **kw):
    '''
    Determine whether or not a function and its arguments gives an error
    For purposes of this HL7 Project, it is typically used in conjunction with the functions index(),index_n(), or exec()
    
    Parameters
    ----------
    func: function, required
    *args: varies, required
    
    Returns
    -------
    bool
        True if function does not cause error.
	False if function causes error.
        
    Requirements
    ------------
    -none
    '''
    try:
        func(*args, **kw)
        return True
    except Exception:
        return False
    
def index(m,ind):
    '''
    Simple function to return m[ind]
    For purposes of this HL7 parsing project, this is typically used in conjunction with the NoError() function.
    '''
    return m[ind]

def LIKE(array,word):
    '''
    Finds all parts of list that have a word in them
    
    Parameters
    ----------
    array : list/array type, required
    word : str, required
    
    Returns
    -------
    np.array
        An array which is a subset of the original containing the word
        
    Requirements
    ------------
    -import numpy as np
    
    '''
    # Convert to numpy array.  Everything's easier with numpy
    array = np.array(array)
    
    # Create in-condition.  List of True/False for each element
    cond = np.array([str(word) in array[i] for i in np.arange(0,len(array))])
    
    # Enact that condition 
    subset = array[cond]
    
    # Return the subset
    return subset

###################################################

def completeness_facvisits(df, Timed = False):
    
    '''
    1. Read in Pandas Dataframe outputted from NSSP_Element_Grabber() function.
    2. Group events by Facility->Patient MRN->Patient Visit Num
        to find unique visits
    3. Return Dataframe.
        dataframe.index -> Facility Name, Number of Visits
        dataframe.frame -> Percents of visits within hospital with
            non-null values in specified column
    
    Parameters
    ----------
    df : pandas.DataFrame, required
        should have format outputted from NSSP_Element_Grabber() function
    *Timed : bool, optional
        If True, gives completion time in seconds
    
    Returns
    -------
    DataFrame
        A pandas dataframe object is returned as a two dimensional data
        structure with labeled axes.
        
    Requirements
    ------------
    *Libraries*
    -from pj_funcs import *
 
    '''

    start_time = time.time()
    
    # Make a visit indicator that combines facility|mrn|visit_num
    df['VISIT_INDICATOR'] = df[['FACILITY_NAME', 'PATIENT_MRN', 'PATIENT_VISIT_NUMBER']].astype(str).agg('|'.join, axis=1)

    # Create array of Falses.  Useful down the road 
    false_array = np.array([False] * len(df.columns))

    # Create empty dataframe we will eventually insert into
    empty = pd.DataFrame(columns=df.columns)

    # Create empty lists for facility_names (facs) and number of patients in a facility (num_patients)
    # These lists will serve as our output's descriptive indexes
    num_visits = []
    facs = []

    # First sort our data by Facility Name.  Sort=False speeds up runtime
    fac_sort = df.groupby('FACILITY_NAME',sort=False)

    # Iterate through the groupby object
    for facility, df1 in fac_sort:

        # Append facility name to empty list
        facs.append(facility)

        # Initiate visit count
        visit_count = 0

        # Sort by Patient MRN
        MRN_sort = df1.groupby(['VISIT_INDICATOR'],sort=False)

        # Initiate list of 0s.  Each column gets +1 for each visit with a non-null column value.
        countz = false_array.copy().astype(int)

        for visit, df3 in MRN_sort:


            # Initiate array of falses
            init = false_array.copy()

            # Looping through the visits ADT data rows, look for non_null values.  True if non-null. 
            #       Use OR-logic to replace 0s in init with 1s and keep 1s as 1s for each iterated row.
            for i in np.arange(0,len(df3)):
                init = init | (df3.iloc[i].notnull())

            # Add information on null (0) vs. non-null (1) columns to countz which is initially all 0 but updates for each patient.
            countz += init.astype(int)

            # Show that the number of visits has increased
            visit_count += 1


        # Append visit number to empty list
        num_visits.append(visit_count)

        # Update empty dataframe with information on completeness (out of 100%) we had for each column
        # * note countz is a 1D array that counts how many visits have non-null values in each column.
        empty.loc[facility,:] = (countz/visit_count)*100


    # Clarify and Create index information for output Dataframe
    empty['Num_Visits'] = num_visits
    empty['Facility'] = facs
    empty = empty.set_index(['Facility','Num_Visits'])
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
    
    # Return filled dataframe.
    return empty

################################################################

def to_hours(item):
    '''
    Takes a datetime object and converts them to the time in hours,
    as a float rounded to the 3rd decimal.
    
    Input
    -----
    item - DateTime object, required
    
    Output
    -----
    Time in hours (dtype: Float)
    
    Requirements
    ------------
    *Libraries*
    -import datetime
    
    *Functions*
    none
    
    '''
    return round((datetime.timedelta.total_seconds(item) / (60*60)),3)

#####################################################################

def to_days(item):
    '''
    Takes a datetime object and converts them to the time in days,
    as a float rounded to the 3rd decimal.
    
    Input
    -----
    item - DateTime object, required
    
    Output
    -----
    Time in days (dtype: Float)
    
    Requirements
    ------------
    *Libraries*
    -import datetime
    
    *Functions*
    none
    
    '''
    return round((datetime.timedelta.total_seconds(item) / (24*60*60)),3)

####################################################################

def timeliness_facvisits_days(df, Timed = False):
    
    '''
    1. Read in Pandas Dataframe straight from PHESS SQL Query-pulled file.
    2. Group events by Facility->Patient MRN->Patient Visit Num
        to find unique visits.  
    3. Return Dataframe
        dataframe.index -> Facility Name
        dataframe.frame -> Statistics on time differences between MSG_DATETIME
                            and ADMIT_DATETIME
    
    Parameters
    ----------
    df : pandas.DataFrame, required
        example:  df = pd.read_csv('some/path/PHESS_OUTPUT_FILE.csv', encoding = 'Cp1252')
    *Timed : bool, optional
        If True, gives completion time in seconds
    
    Returns
    -------
    DataFrame
        A pandas dataframe object is returned as a two dimensional data
        structure with labeled axes.
        
    Requirements
    ------------
    *Libraries*
    -import pandas as pd
    -import numpy as np
    -import datetime
    -import time
    -from pj_funcs import *    

    *Functions*
    - to_days    (found in pj_funcs.py file)

    '''

    start_time = time.time()

    # Cleanup 1:  ADMIT_DATETIME == 'Missing admit datetime'
    df = df[df['ADMIT_DATETIME'] != 'Missing admit datetime']

    # Cleanup 2:  Some datetimes (meaning 1/1000+) have a decimal in them
    #           They cannot be interpreted as datetimes via pd.to_datetime
    #           so we need to convert them.

    # Interperet ADMIT_DATETIME as string
    admit_time = df['ADMIT_DATETIME'].astype(str)

    # Use Pandas str.split function to divide on decimal, expand, and
    #      take the first argument (everything before the decimal).
    admit_time = admit_time.str.split('\.',expand=True)[0]

    # Convert our newly cleaned strings to datetime type. For uniformity, choose UTC
    admit_time = pd.to_datetime(admit_time, utc=True)

    # Do the exact same thing to 'MSG_DATETIME'
    msg_time = df['MSG_DATETIME'].astype(str)
    msg_time = msg_time.str.split('\.',expand=True)[0]
    msg_time = pd.to_datetime(msg_time, utc=True)

    # Update 'ADMIT_DATETIME' and 'MSG_DATETIME' columns to new format
    df['ADMIT_DATETIME'] = admit_time
    df['MSG_DATETIME'] = msg_time
    
    ##################################################################
    
    #  Create TimeDif Column!!

    TimeDif = msg_time - admit_time

    #  Apply my personal to_days function to see datetime differences in days.
    #  Information can be found in pj_funcs.py or by typing 'to_days?' in a cell
    df['TimeDif (days)'] = TimeDif.apply(to_days)
    

    # Only take the important columns in sub-dataframe
    sub_df = df[['ADMIT_DATETIME','MSG_DATETIME','PATIENT_MRN',
                           'PATIENT_VISIT_NUMBER','FACILITY_NAME','TimeDif (days)']]


    ##################################################################
    
    facs = []


    # First sort our data by Facility Name.  Sort=False speeds up runtime
    fac_sort = sub_df.groupby('FACILITY_NAME',sort=False)

    # Label columns we will eventully populate in empty dataframe
    stats_cols = ['Num_Visits','Median','Avg','StdDev','Min','Max']
    empty = pd.DataFrame(columns=stats_cols)

    # Iterate through the groupby object
    for facility, df1 in fac_sort:

            # Create empty list to fill with TimeDif (days) values for visits
            fillme = []

            # Sort by Patient MRN
            MRN_sort = df1.groupby(['PATIENT_MRN'],sort=False)

            # Loop through MRN groupings
            for patient, df2 in MRN_sort:

                # If there is a null value in the MRN group, we have a problem
                if sum(df2['PATIENT_VISIT_NUMBER'].isnull()) > 0:

                    # If there is only one row and its null, its one patient.
                    if len(df2) == 1:
                        fillme.append(df2.iloc[0]['TimeDif (days)'])

                # Cases where all PATIENT_VISIT_NUMBER are non-null!
                else:

                    # Sort further by Patient Visit Number
                    VisNum_sort = df2.groupby(['PATIENT_VISIT_NUMBER'],sort=False)

                    # Loop through Patient Visit Numbers
                    for visit, df3 in VisNum_sort:

                        # Find the row with the newest 
                        index_earliest = df3['ADMIT_DATETIME'].idxmin()

                        # Within our early admit datetime row, pull TimeDif
                        dif_we_take = df3.loc[index_earliest]['TimeDif (days)']

                        # Append correct TimeDif to fillme list
                        fillme.append(dif_we_take)

            # Convert list (that we appended to) into np array and perform stats
            fillme = np.array(fillme)

            stats = [len(fillme),np.median(fillme),np.mean(fillme),np.std(fillme),
                    np.min(fillme),np.max(fillme)]

            # Fill stats into dataframe for that facility.  Rounded to 2 decimals
            empty.loc[facility,:] = np.array(stats).round(2)
        
        
    ###########################################################################
    
    
    
    
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
    
    # Return filled dataframe.
    return empty



##############################################################################################################################


def timeliness_facvisits_hours(df, Timed = False):
    
    '''
    1. Read in Pandas Dataframe straight from PHESS SQL Query-pulled file.
    2. Group events by Facility->Patient MRN->Patient Visit Num
        to find unique visits.  
    3. Return Dataframe
        dataframe.index -> Facility Name
        dataframe.frame -> Statistics on time differences between MSG_DATETIME
                            and ADMIT_DATETIME
    
    Parameters
    ----------
    df : pandas.DataFrame, required
        example:  df = pd.read_csv('some/path/PHESS_OUTPUT_FILE.csv', encoding = 'Cp1252')
    *Timed : bool, optional
        If True, gives completion time in seconds
    
    Returns
    -------
    DataFrame
        A pandas dataframe object is returned as a two dimensional data
        structure with labeled axes.
        
    Requirements
    ------------
    *Libraries*
    -import pandas as pd
    -import numpy as np
    -import datetime
    -import time
    
    *Functions*
    - to_hours    (found in pj_funcs.py file)

    '''

    start_time = time.time()

    # Cleanup 1:  ADMIT_DATETIME == 'Missing admit datetime'
    df = df[df['ADMIT_DATETIME'] != 'Missing admit datetime']

    # Cleanup 2:  Some datetimes (meaning 1/1000+) have a decimal in them
    #           They cannot be interpreted as datetimes via pd.to_datetime
    #           so we need to convert them.

    # Interperet ADMIT_DATETIME as string
    admit_time = df['ADMIT_DATETIME'].astype(str)

    # Use Pandas str.split function to divide on decimal, expand, and
    #      take the first argument (everything before the decimal).
    admit_time = admit_time.str.split('\.',expand=True)[0]

    # Convert our newly cleaned strings to datetime type. For uniformity, choose UTC
    admit_time = pd.to_datetime(admit_time, utc=True)

    # Do the exact same thing to 'MSG_DATETIME'
    msg_time = df['MSG_DATETIME'].astype(str)
    msg_time = msg_time.str.split('\.',expand=True)[0]
    msg_time = pd.to_datetime(msg_time, utc=True)

    # Update 'ADMIT_DATETIME' and 'MSG_DATETIME' columns to new format
    df['ADMIT_DATETIME'] = admit_time
    df['MSG_DATETIME'] = msg_time
    
    ##################################################################
    
    #  Create TimeDif Column!!

    TimeDif = msg_time - admit_time

    #  Apply my personal to_days function to see datetime differences in days.
    #  Information can be found in pj_funcs.py or by typing 'to_days?' in a cell
    df['TimeDif (hrs)'] = TimeDif.apply(to_hours)
    

    # Only take the important columns in sub-dataframe
    sub_df = df[['ADMIT_DATETIME','MSG_DATETIME','PATIENT_MRN',
                           'PATIENT_VISIT_NUMBER','FACILITY_NAME','TimeDif (hrs)']]


    ##################################################################
    
    facs = []


    # First sort our data by Facility Name.  Sort=False speeds up runtime
    fac_sort = sub_df.groupby('FACILITY_NAME',sort=False)

    # Label columns we will eventully populate in empty dataframe
    stats_cols = ['Num_Visits','Avg TimeDif (hrs)','% visits recieved within 24 hours','% visits recieved between 24 and 48 hours ',
                  '% visits recieved after 48 hours']
    empty = pd.DataFrame(columns=stats_cols)

    # Iterate through the groupby object
    for facility, df1 in fac_sort:

            # Create empty list to fill with TimeDif (hrs) values for visits
            fillme = []

            # Sort by Patient MRN
            MRN_sort = df1.groupby(['PATIENT_MRN'],sort=False)

            # Loop through MRN groupings
            for patient, df2 in MRN_sort:

                # If there is a null value in the MRN group, we have a problem
                if sum(df2['PATIENT_VISIT_NUMBER'].isnull()) > 0:

                    # If there is only one row and its null, its one patient.
                    if len(df2) == 1:
                        fillme.append(df2.iloc[0]['TimeDif (hrs)'])

                # Cases where all PATIENT_VISIT_NUMBER are non-null!
                else:

                    # Sort further by Patient Visit Number
                    VisNum_sort = df2.groupby(['PATIENT_VISIT_NUMBER'],sort=False)

                    # Loop through Patient Visit Numbers
                    for visit, df3 in VisNum_sort:

                        # Find the row with the newest 
                        index_earliest = df3['ADMIT_DATETIME'].idxmin()

                        # Within our early admit datetime row, pull TimeDif
                        dif_we_take = df3.loc[index_earliest]['TimeDif (hrs)']

                        # Append correct TimeDif to fillme list
                        fillme.append(dif_we_take)

            # Convert list (that we appended to) into np array and perform stats
            fillme = np.array(fillme)
            
            cond_bottom = (fillme <= 24)
            cond_middle = (fillme > 24)&(fillme < 48)
            cond_top = (fillme >= 48)
            
            percent_bottom = round((sum(cond_bottom)/len(fillme)),3)*100
            percent_middle = round((sum(cond_middle)/len(fillme)),3)*100
            percent_top = round((sum(cond_top)/len(fillme)),3)*100

            stats = [len(fillme),np.mean(fillme),percent_bottom,percent_middle,percent_top]

            # Fill stats into dataframe for that facility.  Rounded to 2 decimals
            empty.loc[facility,:] = np.array(stats).round(2)
        
        
    ###########################################################################
    
    
    
    
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
    
    # Return filled dataframe.
    return empty


##################################################################################

def index_n(m,ind):
    '''
    Indexes some object 'm' by each element in the list 'ind'
    
    Parameters
    ----------
    m: type varies, required
    ind: list, required
    
    Returns
    -------
    m[ind[0]][ind[1]][ind[...]][ind[n]]
     
    Requirements
    ------------
    -Numpy as np
    
    '''
    for i in np.arange(0,len(ind)):
        m = m[ind[i]]
    return m

###################################################################################

def Index_pull(ind,m):
    
    '''
    Locates and returns the element within a message 'm' thats location
        is described by indeces, 'ind'
    
    Parameters
    ----------
    ind: list, required, full index path as list indicating HL7 location.
    m: hl7 type object, required, m = hl7.parse(some_message)
    
    Returns
    -------
    Str
        Element
     
    Requirements
    ------------
    -NoError from pj_funcs.py
    -index_n from pj_funcs.py
    -hl7
    
    '''
    
    output = ''
    
    # Try indexing the message by ind
    if NoError(index_n,m,ind):
        
        #  If the indexing up to the 2nd to last element returns a string, accept it.  Call it 'output'
        if type(index_n(m,ind[:-1])) == str:
            output = index_n(m,ind[:-1])

        # Normally, we will take the exact, full-indexed value.  Call it 'output'
        else:
            output = str(index_n(m,ind))
    
    # Return output.  If none found, return empty string, ''
    return output

######################################################################################


def Index_pull_CONC(field,rest_index,m):
    '''
    Returns a concetated string for elements with repeating fields. Seperated by '|' characters.
    
    Example: consider the case of Ethnicity Code where a patient may have multiple selected ethnicities.
        For our example we will assume this element is always located in PID-22.1.
    
            print(Index_pull_CONC('PID', [22,0,0], m))
                Ethnicity1|Ethnicity2
        
        Note:  Ethnicity1 and Ethnicity2 are pulled from PID|x|-22.1 and PID|y|-22.1 respectively where
            x,y are non-equal integers representing different repetitions of a repeated field.
        
    
    Parameters
    ----------
    field: list (with one element), required, for non-empty return choose valid 3 letter HL7 field
    rest: list, required, integer list indicating where to find it.
    m: hl7 type object, required, m = hl7.parse(some_message)
    
    Returns
    -------
    Str
        Concetation represented by '|'
     
    Requirements
    ------------
    -NoError from pj_funcs.py
    -index_n from pj_funcs.py
    -Numpy as np
    -hl7
    
    '''
    
    # Initialize empty output
    output = ''
    
    # Read in field
    field_str = field[0]
    
    # Check to see if the field exists in our message
    if NoError(index,m,field_str):
        
        # Set the field equal to 'fi'
        fi = m[field_str]
        
        # If the field repeats, it has a non-zero length. Loop through its length 1 by 1
        for u in np.arange(0,len(fi)):
            
            # Identify the total index by summing strings: field, loop_number, rest_index
            tot_index = field+[u]+rest_index
            
            # Make sure message can be indexed by the total index
            if NoError(index_n,m,tot_index):
                
                #  If the indexing up to the 2nd to last element returns a string, accept it.  Call it 'output'
                if type(index_n(m,tot_index[:-1])) == str:
                    full = index_n(m,tot_index[:-1])
                    
                    # If this string, 'full', has non-zero length, add it to our output and end with '|'
                    if len(full)>0:
                        output += full
                        output += '|'
                        
                # Normally, we will take the exact, full-indexed value.  Call it 'output'
                else:
                    full = str(index_n(m,tot_index))
                    
                    # If this string, 'full', has non-zero length, add it to our output and end with '|'
                    if len(full)>0:
                        output += full
                        output += '|'
                        
                # Go back and loop through more repeated fields until no more exist
                
    # if non-zero length output, clean up last trailing '|' character
    if len(output)>0:
        if output[-1] == '|':
            output = output[:-1]
            
    # Return output.  If none found, this will be '' (empty string)
    return output

############################################################################################################

def DI_One(ind,m,df,z,col_name):
    
    '''
    Returns the element value of 'm' indexed by 'ind'.
    Updates the dataframe 'df' cell value indexed by 'z' and 'col_name'
    
    Parameters
    ----------
    ind: list, required, complete index path (as list) to desired element
    m: hl7 type object, required, m = hl7.parse(some_message)
    df:  pandas DataFrame, required
    z:  int, required, valid integer row index of df
    col_name: str, required, valid column in df
    
    Returns
    -------
    Str
        Element
        
    Output
    ------
    Updates dataframe
        df.loc[z,col_name] = Element
     
    Requirements
    ------------
    -Index_pull from pj_funcs.py
    -Pandas
    -hl7
    
    '''
    
    # Call the index on the message.
    obj = Index_pull(ind,m)
    
    # See if the 'obj' is an actual non-zero thing.
    if len(obj)>0:
        
        # If so, append to the row_z, col_colname in Dataframe, df
        df.loc[z,col_name] = obj
        
    # Else:  Do nothing.
    
    # Return the object.  If none found, will return empty str, '' with no df update
    return obj

####################################################################

def DI_One_CONC(field,ind,m,df,z,col_name):
    
    '''
    Returns the CONCETATED element value of 'm' indexed by its respective
        repeating field, 'field', and 'ind'.
    Updates the dataframe 'df' cell value indexed by 'z' and 'col_name'
    
    Parameters
    ----------
    field: list (with one element), required, for non-empty return choose valid 3 letter HL7 field
    ind: list, required, complete index path (as list) to desired element
    m: hl7 type object, required, m = hl7.parse(some_message)
    df:  pandas DataFrame, required
    z:  int, required, valid integer row index of df
    col_name: str, required, valid column in df
    
    Returns
    -------
    Str
        Concetated_Element separated by '|'
        
    Output
    ------
    Updates dataframe
        df.loc[z,col_name] = Concetated_Element
     
    Requirements
    ------------
    -Index_pull_CONC from pj_funcs.py
    -Pandas
    -hl7
    
    '''
    
    # Call the index on the message.
    obj = Index_pull_CONC(field,ind,m)
    
    # See if the 'obj' is an actual non-zero thing.
    if len(obj)>0:
        
        # If so, append to the row_z, col_colname in Dataframe, df
        df.loc[z,col_name] = obj
        
    # Else:  Do nothing.
    
    # Return the object
    return obj

############################################################################################################

def NSSP_Element_Grabber(data,Timed = True, Priority_only=False, outfile='None'):
    '''
    Creates dataframe of important elements from PHESS data.
    
    Parameters
    ----------
    data: pandas DataFrame, required, from PHESS sql pull
    
    Timed:  Default is True.  Prints total runtime at end.
    Priority_only:  Default is False.  
        If True, only gives priority 1 or 2 elements
    outfile:  Default is 'None':
        Replace with file name for dataframe to be wrote to as csv
        Will be located in working directory.
        DO NOT INCLUDE .csv IF YOU CHOOSE TO MAKE ONE
    
    Returns
    -------
    dataframe
        
    Requirements
    ------------
    - import pandas as pd
    - import numpy as np
    - import time
    '''
    # Start our runtime clock.
    start_time = time.time()
    
    
    # Read in reader file as pandas dataframe

    DATA_PATH = pkg_resources.resource_filename('HL7reporting', 'supporting/')
    FILE = pkg_resources.resource_filename('HL7reporting', 'supporting/NSSP_Element_Reader.xlsx')
    reader = pd.read_excel(FILE)
    
    # Create empty dataframe with rows we want interpreted from reader file
    df = pd.DataFrame(columns=reader['Processed Column'])
    
    # Create a few extra columns straight from our data file
    df['MESSAGE'] = data['MESSAGE']
    df['FACILITY_NAME'] = data['FACILITY_NAME']
    df['PATIENT_VISIT_NUMBER'] = data['PATIENT_VISIT_NUMBER']
    df['PATIENT_MRN'] = data['PATIENT_MRN']

    # Create a subset of rows from our reader file.  Only ones to loop through.
    # Order by 'Group_Order' so that some run before others that rely on previous.
    reader_sub = reader[reader.Ignore == 0].sort_values('Group_Order')

    # Loop through all data rows
    for z in np.arange(0,len(data)):
        
        # Locate our message
        message = df['MESSAGE'][z]
        
        # Decipher using hl7 function
        m = hl7.parse(message)
        
        # For each row in our reader file subset
        for j in np.arange(0,len(reader_sub)):
            
            # Initialize object.  Don't want one recycled from last loop
            obj=''
            
            # Choose the row we will use from the reader file
            row = reader_sub.iloc[j]
            
            # Identify element name we're working with.  Also a column name in output dataframe
            col_name = str(row['Processed Column'])
            
            # Identify code from our reader file we use to find the element in the HL7 message
            subcode = row['Code']
            
            # Does executing this code (originally a string) cause an error?
            ### NOTE:  calling locals and globals allows you to access all home-grown functions
            if NoError(exec,subcode,globals(), locals()):
                
                # If no errors, execute the code.
                exec(subcode,globals(), locals())


    # Some values may be empty strings and we do not want to count them as filled
    df = df.replace('',np.nan)
                
    # End time stopwatch
    end_time = time.time()

    # Unless they did not want it, print runtime
    if Timed != False:
        print(end_time-start_time)
    
    # If they only want priority elements:
    if Priority_only==True:
        # left = all columns interpreted from reader file
        left = df.iloc[:,:-4] 
        # right = MESSAGE, FACNAME, PATIENT_VN, PATIENT_MRN
        right = df.iloc[:,-4:]
        # find all cols we want from reader file. Priority cols
        priority_cols = reader['Processed Column'][(reader['Priority'] == 1.0)|(reader['Priority'] == 2.0)]
        # Index our left set by these columns 
        col_cut = left.loc[:,priority_cols]
        # glue left indexed with right again
        df = col_cut.join(right)
        
    # If they want an output file...
    if outfile!='None':
        # Specify output path and add csv bit.
        outpath = outfile+'.csv'
        # No index
        df.to_csv(outpath, index=False)
    
    # return the dataframe!
    return df

############################################################################################################

def priority_cols(df, priority='both', extras=None, drop_cols=None):
    '''
    Spits out priority columns from a dataframe.
    Priority can be 1,2, or both.
    Extras indicate additional columns from the original dataframe you would like the output to contain.
    Drop_Cols indicate columns that you want to NOT include

    Parameters
    ----------
    df: pandas dataframe, required
    *priority: str, optional (default is both)
            'both' - returns priority 1 and priority 2 element columns
            'one' or '1' - returns priority 1 element columns only
            'two' or '2' - returns priority 2 element columns only
    *extras:  list, optional (default is None)
            list must contain valid column values from df.
    *drop_cols:  list, optional (default is None)
            list must contain valid column values from df.

    Returns
    -------
    pandas Dataframe
       
    Requirements
    ------------
    -import pandas as pd
    '''

    DATA_PATH = pkg_resources.resource_filename('HL7reporting', 'supporting/')
    FILE = pkg_resources.resource_filename('HL7reporting', 'supporting/NSSP_Element_Reader.xlsx')
    reader = pd.read_excel(FILE)


    # There is a chance that the user removed priority columns from the input dataframe for their own reasons.
    #     Therefore we only want to look at cases where the input dataframe columns match the reader processed column.
    reader = reader[reader['Processed Column'].isin(df.columns)]

    if priority.upper() == 'BOTH':
        cols = reader['Processed Column'][((reader.Priority == 1.0)|(reader.Priority == 2.0))]
        if extras != None:
            cols = list(cols)
            for item in extras:
                cols.append(item)
        new = df.loc[:,cols]
        if drop_cols != None:
            new = new.drop(list(drop_cols),axis=1)
        return new
        
    elif (priority.upper() == 'ONE')|(priority == '1'):
        cols = reader['Processed Column'][(reader.Priority == 1.0)]
        if extras != None:
            cols = list(cols)
            for item in extras:
                cols.append(item)
        new = df.loc[:,cols]
        if drop_cols != None:
            new = new.drop(list(drop_cols),axis=1)
        return new
    
    elif (priority.upper() == 'TWO')|(priority == '2'):
        cols = reader['Processed Column'][(reader.Priority == 2.0)]
        if extras != None:
            cols = list(cols)
            for item in extras:
                cols.append(item)
        new = df.loc[:,cols]
        if drop_cols != None:
            new = new.drop(list(drop_cols),axis=1)
        return new
    
    else:
        print('Incorrect entry for specify.  Choose one of the following:  [\'both\',\'1\',\'2\']')

############################################################################################################


############################################################################################################

def validity_check(df, Timed=True):
    
    '''
    Checks to see which elements in a dataframe's specific NSSP priority columns meet NSSP validity standards.
    Returns a True/False dataframe with FACILITY_NAME,PATIENT_MRN,PATIENT_VISIT_NUMBER as only string-type columns
    
    Parameters
    ----------
    
    df - required, pandas Dataframe, output from NSSP_Element_Grabber() function    
    Timed - optional, boolean (True/False), default is True.  Returns time in seconds of completion.
    
    Returns
    --------
    validity_report - True/False dataframe with FACILITY_NAME,PATIENT_MRN,PATIENT_VISIT_NUMBER as only string-type columns
    
    Requirements
    -------------
    import numpy as np
    import pandas as pd
    import time
    
    '''
    
    # Initialize Time
    start_time = time.time()
    
    
    # Read in the validity key

    DATA_PATH = pkg_resources.resource_filename('HL7reporting', 'supporting/')
    FILE = pkg_resources.resource_filename('HL7reporting', 'supporting/NSSP_Validity_Reader.xlsx')
    key = pd.read_excel(FILE)

    # There is a chance that the user had decided to get rid of columns we typically check for validity.
    #      Therefore we need to only loop through 'key' rows that match our input dataframe's columns.
    key = key[key['Element'].isin(df.columns)]
    
    # Initialize empty pandas dataframe
    validity_report = pd.DataFrame()
    
    # Make sure we know nan means NaN
    nan = np.nan
    
    # Loop through each row in our validity key file
    for i in np.arange(0,len(key)):
        
        # Locate the row that our loop is on.  Define:
        row = key.iloc[i]
        
        # The element name 
        col_name = row['Element']
        
        #######################################################################################################
        #  All NSSP Priority Elements have validity checks that fall into one of the following 4 categories.
        #######################################################################################################

        # The list that the value may need to be part of to be valid.
        row_list = row['List']
        
        # The list that the value should not be part of to be valid.
        row_notlist = row['NOT_List']
        
        # The upper bound of a numeric value that it needs in order to be valid.
        row_bounds = row['Bounds']
        
        # The string fomat (in RegEx format) that a value needs to be valid.
        row_format = row['Format']

        #######################################################################################################
        # Check to see if this element has a non-null entry for one of the 4 broad criteria.
        #     If it does (which will only work for one of the four):
        #           Execute the code on the Element's who data column.
        #           Append a newly formed True/False array as a column to our output validity report
        #######################################################################################################
        
        if (row_list == row_list):
            listy = row_list.split(',')
            validity_report[col_name] = (df[col_name].str.upper().isin(listy))

        elif (row_notlist == row_notlist):
            nonlist = row_notlist.split(',')
            validity_report[col_name] = (~df[col_name].str.upper().isin(nonlist))  

        elif (row_bounds == row_bounds):
            num = 120
            validity_report[col_name] = pd.to_numeric(df[col_name]) < num

        elif (row_format == row_format):
            search = row_format
            validity_report[col_name] = (df[col_name].str.contains(search,na=False))
            
            
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
            
    return validity_report

############################################################################################################

def Visualize_Facility_DQ(df, fac_name, hide_yticks = False, Timed = True):
    '''
    Returns Visualization of data quality in the form of a heatmap.
    Rows are all individual visits for the inputted facility.
    Columns are NSSP Priority elements that can be checked for validity.
    Color shows valid entries (green), invalid entries (yellow), and absent entries (red)
    
    Parameters
    ----------
    
    df - required, pandas Dataframe, output from NSSP_Element_Grabber() function
    fac_name - required, str, valid name of facility.
        if unsure of valid entry options, use the following code for options:
        df['FACILITY_NAME'].unique()   # may need to change for your df name
    
    Returns
    --------
    out[0] = Pandas dataframe used to create visualization.  2D composed of 0s (red), 1s (yellow), 2s (green)
    out[1] = Pandas dataframe of data behind visit.  Multiple HL7 messages composing 1 visit concatenated by '~' character
    
    Output
    -------
    sns.heatmap visualization
    
    Requirements
    -------------
    import numpy as np
    import seaborn as sns
    import matplotlib.pylab as plt
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    
    '''

    # Initialize Time
    start_time = time.time()
    
    # Create sub-dataframe of only visits within a facility
    hosp_visits = df[df.FACILITY_NAME==fac_name]

    # Read in our validity key
    DATA_PATH = pkg_resources.resource_filename('HL7reporting', 'supporting/')
    FILE = pkg_resources.resource_filename('HL7reporting', 'supporting/NSSP_Validity_Reader.xlsx')
    key = pd.read_excel(FILE)

    # There is a chance that the user had decided to get rid of columns we typically check for validity.
    #      Therefore we need to only loop through 'key' rows that match our input dataframe's columns.
    key = key[key['Element'].isin(df.columns)]

    # Initialize data quality array (0s 1s 2s)
    out0 = pd.DataFrame(columns=key.Element)
    
    # Initialize data represented array (hl7 data concatenated by ~ character)
    out1 = pd.DataFrame(columns=key.Element)

    # Set original index to 0.  Will increase by 1 after every visit has its info captured.
    cur_index = 0

    # Group by MRN
    MRN_group = hosp_visits.groupby('PATIENT_MRN')

    # Loop through our MRN Groups
    for index,frame in MRN_group:

        # Group by Visit Number
        VISIT_group = frame.groupby('PATIENT_VISIT_NUMBER')

        # Loop through VISITS
        for index2,frame2 in VISIT_group:

            # Initialize dataframe
            one_visit = pd.DataFrame()

            # Only look at visit info that can be checked for validity (must be a validity key element)
            impz = frame2.loc[:,key.Element]

            # Create correct format input for validity check.
            #    needs FACILITY_NAME,PATIENT_VISIT_NUMBER,PATIENT_MRN
            impz2 = impz.copy()
            impz2['FACILITY_NAME'] = fac_name
            impz2['PATIENT_MRN'] = index
            impz2['PATIENT_VISIT_NUMBER'] = index2

            # Run a validity check on our visit's important columns
            one_visit = validity_check(impz2,Timed=False).iloc[:,:]

            # Completness returns 1D list of 0s / 1s determining if there is a non-null value in each column
            completeness = ((~impz.isnull()).sum() != 0).astype(int)

            # Validness returns 1D list of 0s / 1s determining if there is a valid value in each column
            validness = (one_visit.sum() != 0).astype(int)

            # Sum completness + validness to get picture for overall data quality
            tots = completeness+validness

            # Save this overall data quality score into our out0 array and save the index (Patient Visit Number)
            out0.loc[cur_index,:] = tots
            out0.loc[cur_index,'PATIENT_VISIT_NUMBER'] = index2

            # Also save our data that has been assessed for quality.  Concatenate by '~' character
            # First replace all NaN with empty character.  Need this to concat strings together.
            impz_no_na = impz.fillna('')
            
            for col in impz_no_na.columns:
                newcol = '~'.join(impz_no_na[col].astype(str))
                out1.loc[cur_index,col] = newcol

        
            # Visit over, onto the next.  Increase current index by +1
            cur_index += 1

    # Reset our arbitrary 0-n index and replace with the patient visit number
    out0.reset_index()
    out0.set_index('PATIENT_VISIT_NUMBER')

    # Look at how many visits we have
    num_visits = len(out0)

    # Create scalar (just made sense in my head) for figure scaling.
    scalar = int(num_visits/20)+1
    
    # Create figure/axes with my respective scaling
    fig, ax = plt.subplots(figsize=(20/scalar,1*num_visits/scalar))

    # Create custom colormap
    my_cmap = colors.ListedColormap(['Red','Yellow','Green'])

    # Make a heatmap of our 0,1,2 array of absent, invalid, valid elements.
    #     specify linewidth, xticks, yticks, linecolor separation to black
    heatmap = sns.heatmap(np.array(out0)[:,:-1].astype(int),cmap=my_cmap,linewidth=0.5,
                          xticklabels=key.Element, linecolor='k',center=1,
                         yticklabels=out0.PATIENT_VISIT_NUMBER)
    
    # Hide yticks if necissary
    if hide_yticks == True:
        plt.yticks([])


    ###################################################
    # Plot customization
    ###################################################

    # Increase size of xticks and x/y axes
    matplotlib.rc('xtick', labelsize=15) 
    matplotlib.rc('ytick', labelsize=15) 
    #matplotlib.rc('axes', labelsize=25) 
    plt.rc('axes', titlesize=25)     
    plt.rc('axes', labelsize=20)
    #matplotlib.rc('title', labelsize=30) 

    # Set colorbar axis
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.33, 1, 1.66])
    cbar.set_ticklabels(['Absent', 'Invalid', 'Valid'])

    # Set Title
    plt.title('NSSP Priority Element\nData Visualization\n'+fac_name)    
    
    # Set and rotate xticks 90 deg
    plt.xticks(rotation=90) 
    plt.xlabel('NSSP Element')

    # Set ylabel
    plt.ylabel('Patient Visit Number')

    # Show your result
    plt.show()
    
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
    
    return out0,out1

####################################################################################################################################


def issues_in_messages(df, Timed=True, combine_issues_on_message = False, split_issue_column = False):
    '''
    Processes dataframe outputted by NSSP_Element_Grabber() function.
    Outputs dataframe describing message errors.  See optional args for output dataframe customation.
    
    Parameters
    ----------
    
    df - required, pandas Dataframe, output from NSSP_Element_Grabber() function
    *Timed - optional, bool, default is True.  Outputs runtime in seconds upon completion.
    *combine_issues_on_message - optional, bool, default is False.  SEE (2) below
    *split_issue_column - optional, bool, default is False.  SEE (3) below
    
    
    NOTE:  only one of 'combine_issues_on_message' or 'split_issue_column' can be True
    
    Returns
    ----------------------------------------------------------------------------
    Pandas dataframe. Columns include:
    
    (1)
    DEFAULT: WHEN split_issue_colum = False , combine_issue_on_message = False
    
    Group_ID -> string concatenation of FACILITY_NAME|PATIENT_MRN|PATIENT_VISIT_NUMBER
    MESSAGE -> full original message
    Issue -> string concatenation of 'error_type|element_name|priority|description|valid_options|message_value|suggestion|comment'
    
    ------
    
    (2)
    WHEN combine_issue_on_message = True, split_issue_colum = False 
    
    Group_ID -> string concatenation of FACILITY_NAME|PATIENT_MRN|PATIENT_VISIT_NUMBER
    MESSAGE -> full original message
    Issue -> string concatenation of 'error_type|element_name|priority|description|valid_options|message_value|suggestion|comment'
             MULTIPLE string concatenations per cell, separated by newline '\n'
    
    Num_Missings -> number of issues that had a type of 'Missing or Null'
    Num_Invalids -> number of issues that had a type of 'Invalid'
    Num_Issues_Total -> number of total issues
    
    ------
    
    (3)
    WHEN combine_issue_on_message = False , split_issue_colum = True
    
    Group_ID -> string concatenation of FACILITY_NAME|PATIENT_MRN|PATIENT_VISIT_NUMBER
    MESSAGE -> full original message
    error_type -> 'Missing or Null' or 'Invalid'
    element_name -> NSSP Priority Element name with issue
    priority -> NSSP Priority '1' or '2'
    description -> Describes location/parameters of element in HL7 message
    valid_options -> IF element can be checked for validity, describes a valid entry.
    message_value -> IF element was determined as invalid, give the invalid element value.
    suggestion -> IF element was determined as invalid, give an educated guess as to what they meant.
    comment -> IF element was determined as invalid, give feedback/advice on the message error.
    
    
    --------------------------------------------------------------------------------
    
    Requirements
    -------------
    from pj_funcs import *
    import numpy as np
    import pandas as pd
    import time
    
    '''

    if (combine_issues_on_message==True)&(split_issue_column==True):
        print('ERROR:  Only 1 of: combine_issues_on_message / split_issue_column can be True ')
        return -1
    
    # Initialize Time
    start_time = time.time()
    ########################################################################################################
    # Create dataframe of 0s, 0.5s, 1s representing missing/null , invalid, and valid values.
    ########################################################################################################


    # we only want to look at priority columns
    new = priority_cols(df)

    #  Create a new column combining all information to group by visit
    new['Grouper_ID'] = df.FACILITY_NAME+'|'+df.PATIENT_MRN+'|'+df.PATIENT_VISIT_NUMBER

    ############

    # Run a validity check on the priority columns
    vc = validity_check(new,Timed=False)

    # Validity check only outputs priority columns so redefine our grouper ID.  We will set this to be our index
    vc['Grouper_ID'] = new['Grouper_ID']
    vc = vc.set_index('Grouper_ID')

    # Create a copy of our dataframes priority cols (new) and call it df1.  Set its index
    df1 = new.copy()
    df1 = df1.set_index('Grouper_ID')

    ############

    # Ones that have a non-empty value we will asign a value of 1 to.  Null-values will be assigned 0
    df_comp = (~df1.isnull()).astype(int) 

    # Validity check will also be interpreted as an integer.  
    df_vc = vc.astype(int)

    ############

    # Invalid entries will now be represented as -0.5 in our validity df
    df_vc = df_vc.replace(0,-0.5)

    # Valid entries will be represented as 0 in our validity df
    df_vc = df_vc.replace(1,0)

    ############

    # Find columns that can be checked for validity
    c = df_comp.columns.intersection(df_vc.columns)

    # For these columns, we want to sum our two dataframes [df_vc + df_comp]
    df_comp[c] =  df_comp[c].add(df_vc[c], fill_value=0)

    # NOTE at this point df_comp is an array of -0.5s, 0s, 1s, 2s. If a value that could be invalid was empty, the sum was 0+(-0.5)

    # Replace -0.5 with 0.  Represents an empty visit regarless of if it's also invalid
    df_comp = df_comp.replace(-0.5,0)

    # Reset the index and make a new, copied MESSAGE column
    df_comp = df_comp.reset_index()
    df_comp['MESSAGE'] = df['MESSAGE']

    ######################################################################

    # set the index of new again
    new  = new.set_index('Grouper_ID')

    # Replace any instance of | to ~ because we will later use pipe characters for an important purpose
    new = new.replace('\|','~', regex=True)
    new = new.reset_index()

    ########################################################################################################
    # Begin our part where we Create the dataframe 
    ########################################################################################################

    # Load our key and set its index
    DATA_PATH = pkg_resources.resource_filename('HL7reporting', 'supporting/')
    FILE = pkg_resources.resource_filename('HL7reporting', 'supporting/Message_Corrector_Key.xlsx')
    key = pd.read_excel(FILE)

    key = key.set_index('Element')

    # Initialize df_out
    df_out = pd.DataFrame(columns=['MESSAGE','Grouper_ID','Issue'])
    cur_index = 0

    # To save time on efficiency, we write our rows directly instead of appending them (which rewrites array)
    #     to write rows directly, you need a rough estimate of how many rows you have.  We just did it exactly
    pointless = np.array([''] * (((df_comp == 0)|(df_comp == 0.5)).sum().sum()))
    df_out['DELETE_L8R'] = pointless


    # Loop through all rows in our dataframe of 0s,0.5s,1s (called df_comp)
    for i in np.arange(0,len(df_comp)):

        # Each row will have a grouper and Message we will eventually store in our output dataframe
        grouperID = df_comp['Grouper_ID'].iloc[i]
        message = df_comp['MESSAGE'].iloc[i]

        # Loop through all of the columns in our row
        for col in df_comp.columns:

            # Initialize empty list.  We will fill with strings and concatenate to fill our 'Issue' column if missing/invalid
            entry = []

            # See if the current cell value is 0 -> representing null/missing value
            if df_comp.loc[i,col] == 0:

                # Append the problem, the element name (col), the element priority, the element description
                entry.append('Missing or Null')
                entry.append(str(col))
                entry.append(str(key.loc[col,'Priority']))
                entry.append(str(key.loc[col,'Description']))

                # If there is a list of valid options in our key, append that, otherwise append empty string
                if (key.loc[col,'Valid_Options']) == (key.loc[col,'Valid_Options']):
                    entry.append(str(key.loc[col,'Valid_Options']))
                else:
                    entry.append('')

                # Append empty strings for Message entry, Comment, and Suggestion
                entry.append('')
                entry.append('')
                entry.append('')

            ########################################################################################################

            # See if the current cell value is 0.5 -> representing an invalid value
            elif df_comp.loc[i,col] == 0.5:

                # Append the problem, the element name (col), the element priority, the element description
                entry.append('Invalid')
                entry.append(str(col))
                entry.append(str(key.loc[col,'Priority']))
                entry.append(str(key.loc[col,'Description']))

                # Our df_comp cell was 0.5, therefore there is a list of valid options in our key, append that
                entry.append(str(key.loc[col,'Valid_Options']))

                # Append the value that was determined to be invalid.  DataFrame called new contains all initial cell values.
                entry.append(str(new.loc[i,col]))

                # Initialize our comment and suggestion.  If comment/suggestion exists, our executed code will replace these
                comment = ''
                suggestion = ''

                # See if we have a non-null code value.  
                if (key.loc[col,'Suggestion_Code']) == (key.loc[col,'Suggestion_Code']):

                    # Nearly all of these will call on our invalid value.  Define that
                    invalid_value = str(new.loc[i,col])

                    # Execute the code within the cell. Exec will append comment/suggestion to entry
                    code_to_run = str(key.loc[col,'Suggestion_Code'])
                    exec(code_to_run,globals(),locals())
                
                # If we don't have any code to exec(ute), append empty comment/suggestion to issue string
                else:
                    entry.append(comment)
                    entry.append(suggestion)




            ########################################################################################################

            # If there was a problem (either missing/invalid) the list called entry will be non-empty
            if len(entry) > 0:

                # Join our entries by a pipe character
                issue_string = '|'.join(entry)

                # Append our 3 column row to our output dataframe at the current index
                df_out.loc[cur_index] = [message,grouperID,issue_string,'']

                # Update the current index
                cur_index += 1


    # Delete the axis we initially made just to set length
    df_out = df_out.drop('DELETE_L8R',axis=1)

    
    ##############################################
    # Optional Args TIME
    ##############################################
    
    if (combine_issues_on_message == True):
    
        # create empty dataframe.  Correct length (column-wise).
        comb_on_issue = pd.DataFrame(columns=['MESSAGE','Grouper_ID','Issue'])

        # create the correct lengthed (row-wise) dataframe by recognizing all unique messages.  
        comb_on_issue.MESSAGE = df_out.MESSAGE.unique()

        # initialize a count
        count = 0

        # Loop through groupby objects when we group by MESSAGE
        for index,frame in df_out.groupby('MESSAGE'):

            # Identify the message (which is the index) and the grouper_ID (same for all parts of frame. Arbitrarily choose first index)
            message = index
            grouperID = frame.Grouper_ID.iloc[0]

            # Drop all duplicate rows.  Some messages may appear more than once in our original dataset
            frame2 = frame.drop_duplicates()

            # Combine all unique Issue values by a newline seperator.
            comb_issue = frame2.Issue.str.cat(sep='\n')   

            # Append our new info to dataframe and update our count.  
            comb_on_issue.iloc[count] = [message,grouperID,comb_issue]  
            count+=1
            
        # Create some extra columns describing number of types of errors
        comb_on_issue['Num_Missings'] = comb_on_issue.Issue.str.count('Missing or Null')
        comb_on_issue['Num_Invalids'] = comb_on_issue.Issue.str.count('Invalid\|')
        comb_on_issue['Num_Issues_Total'] = comb_on_issue['Num_Missings'] + comb_on_issue['Num_Invalids']
        
        # Rename our df_out so that we can only return one thing
        df_out = comb_on_issue

    ######################################################################################################################
    
    if (split_issue_column == True):
        expanded_issue = df_out.Issue.str.split('\|',expand=True)
        expanded_issue.columns = ['Issue_Type','Element_Name','Priority','Description','Valid_Options',
                                  'Message_Value','Suggestion','Comment']
        
        # Rename our df_out so that we only return one thing
        df_out = df_out[['MESSAGE','Grouper_ID']].join(expanded_issue)
    
    ######################################################################################################################

    # Keep track of end time
    end_time = time.time()

    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
        
    return df_out


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################

def validity_and_completeness_report(df,description='long',visit_count=False,outfile=None, Timed=True):
    '''
    
    dataframe1 -> Returns completenesss report by hospital with facility,element,percentmissing,percentinvalid,description
    dataframe2 -> Determines the incompleteness (0), invalid (1), or valid and complete (2) for every element in all messages
    
    
    Parameters
    ----------
    df: pandas DataFrame, required (output from NSSP_Element_Grabber() funciton)
    
    description:  str, optional.  (Either 'long' or 'short')
        if 'short', description of location is shorter and less descriptive
        elif 'long', description is sentence structured and descriptive
    
    visit_count:  bool, optional
        if True, add the number of visits to dataframe2
        
    outfile: string, optional
        if True, send excel file to ../data/processed.  Name defined by outfile
        *DO NOT INCLUDE .xlsx or full path
    
    
    Returns
    -------
    df1
        Dataframe showing issues in messages for each hospitals.  Report structure
    
    df2
        Dataframe assessing all messages for incomlete,invalid,valid elements represented as 0s, 1s, and 2s
        
        
    Requirements
    ------------
    -from pj_funcs import *
    
    '''

    start_time = time.time()
    
    # Get validity and completion reports (0s and 1s)
    v = validity_check(df,Timed=False)
    c = priority_cols(~df.isnull())

    ###########################################################################################

    # Check out completion report for columns that can be assessed for validity
    only_validity = c[v.columns]

    # Look out elements that can ONLY be assessed for completion.  Completes now become 2s
    no_valid_checks = list(set(list(c.columns)) - set(list(v.columns)))
    only_completion = c[no_valid_checks]
    only_completion = only_completion.astype(int).replace(1,2)

    # For elements that can be assessed for validity, make it 0s, 1s, 2s.
    validity_completion = only_validity.astype(int) + v.astype(int)

    # Join together validity and completeness stuffz
    df012s = only_completion.join(validity_completion)

    # Append facility name and visit information to dataframe.
    df012s['FacName'] = df.FACILITY_NAME
    df012s['VisInd'] = df.FACILITY_NAME+'|'+df.PATIENT_MRN+'|'+df.PATIENT_VISIT_NUMBER

    ###########################################################################################


    if visit_count == False:
        empty = pd.DataFrame(columns=['Facility','Element','Percent Missing','Percent Invalid','Description'])

    else:
        empty = pd.DataFrame(columns=['Facility','Visit Count','Element','Percent Missing','Percent Invalid','Description'])



    # Create empty dataframe to append our information to 
    empty['Description'] = [np.nan]*len(df012s)

    # Set initial index
    cur_num = 0

    # Group by Facility 
    g1 = df012s.groupby('FacName')

    for ind1, frame1 in g1:

        # Create empty list (l) to append to
        l = []

        # Group by Visit Index (looks like Facility|MRN|VisitNum)
        g2 = frame1.groupby('VisInd')

        # Loop though visits
        for ind2,frame2 in g2:

            # Only want 0s,1s,2s.  No facility name or visit identifier
            f = frame2.drop(['FacName','VisInd'],axis=1) 

            # Only take the max value for elements considering all rows in a single visit. 
            ##### ex: If Patient_Age is 0,0,0,1,2 in respective rows, only take 2 which represents complete and valid.
            l.append(f.max())

        # Get a summary of all visits
        summary = pd.DataFrame(l)

        # Length of this represents all visits
        total_visits = len(summary)

        # Check to see what percent are incomplete and invalid
        cond_complete =(((summary == 0).sum() / len(summary) * 100))
        cond_invalid = (((summary == 1).sum() / len(summary) * 100))

        # Check out any element that has any percent incomplete or invalid
        reportable_incompletes = cond_complete[cond_complete>0]
        reportable_invalids = cond_invalid[cond_invalid>0]

        # Combine any elements that might be partially incomplete or invalid
        report = pd.concat([reportable_incompletes,reportable_invalids],axis=1)

        if visit_count == False:
            # Loop through all bad entries
            for i in np.arange(0,len(report)):

                # Append important information to the empty dataframe 
                importants = ind1,report.index[i],np.round(report.iloc[i,0],2),np.round(report.iloc[i,1],2),np.nan
                empty.iloc[cur_num] = importants

                # Update the index count
                cur_num += 1

        else:
            for i in np.arange(0,len(report)):

                # Append important information to the empty dataframe 
                importants = ind1,total_visits,report.index[i],np.round(report.iloc[i,0],2),np.round(report.iloc[i,1],2),np.nan
                empty.iloc[cur_num] = importants

                # Update the index count
                cur_num += 1


    empty = empty.dropna(axis=0,how='all')

    #####################################################################################################################

    if description == 'long':

        DATA_PATH = pkg_resources.resource_filename('HL7reporting', 'supporting/')
        FILE = pkg_resources.resource_filename('HL7reporting', 'supporting/Message_Corrector_Key.xlsx')
        key = pd.read_excel(FILE)


        key = key.set_index('Element')
        for i in np.arange(0,len(empty)):
            el = empty.loc[i,'Element']
            empty.loc[i,'Description'] = key.loc[el,'Description']

    elif description == 'short':

        DATA_PATH = pkg_resources.resource_filename('HL7reporting', 'supporting/')
        FILE = pkg_resources.resource_filename('HL7reporting', 'supporting/NSSP_Element_Reader.xlsx')
        key = pd.read_excel(FILE)

        key = key.set_index('Processed Column')
        for i in np.arange(0,len(empty)):
            el = empty.loc[i,'Element']
            empty.loc[i,'Description'] = key.loc[el,'HL7 Segment(s)']


    #####################################

    if outfile != None:

        if visit_count == False:
            empty = empty.set_index(['Facility','Element'])
            empty.to_excel(outfile+'.xlsx')

        else:
            empty = empty.set_index(['Facility','Visit Count','Element'])
            empty.to_excel(outfile+'.xlsx')
            
            
    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')
    
            
    return empty, df012s

###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################

def RegEx_on_Full_DataFrame(df, find, replace):
    '''
    Sometimes you want to apply regex find/replace on an entire dataframe.
    * Reminder * -> group one indicated by '\g<1>' in regex
    
    Input:
    ------
    df - pd.Dataframe, required
    find - str, required (RegEx form)
    replace - str, required (RegEx form)
    
    Output:
    -------
    df - pd.Dataframe
    
    Method:
    -------
    1. Convert the dataframe to a 2D numpy array
    2. Flatten that array making it 1D (think of it as stretching it)
    3. Convert flattened version to pd.Series to apply str.replace() using RegEx
    4. After RegEx application, convert back to array, reshape to same as step 1.
    5. Convert 2D numpy array back to pd.Dataframe with original columns
    
    Requirements:
    -------------
    import pandas as pd
    import numpy as np
    
    '''
    original_cols = df.columns
    
    numpy_2d = np.array(df)
    original_shape = numpy_2d.shape
    
    numpy_1d = numpy_2d.flatten()
    series_1d = pd.Series(numpy_1d)
    
    series_application = series_1d.str.replace(find,replace)
    
    back_to_numpy_1d = np.array(series_application)
    back_to_numpy_2d = back_to_numpy_1d.reshape(original_shape)
    
    
    out_df = pd.DataFrame(back_to_numpy_2d,columns=original_cols)
    
    out_df = out_df.set_index(df.index)
    
    return out_df



###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################



def Visualization_interactive(df_before,df_after,str_date_list,priority='both_combined',outfile=False,show_plot=False,Timed=True):
    
    '''
    Creates an annotated heatmap that is interactive with hoverover.
    Heatmap colors represent data completeness as of the first date
    Annotations show the completion percent change with respect to the second date
        (+ indicates increased completeness)

    Parameters
    ----------
    df_before : pandas.DataFrame, required (output of NSSP_Element_Grabber() Function)
        -must be the dataframe representing EARLIER data
        
    df_after : pandas.DataFrame, required (output of NSSP_Element_Grabber() Function)
        -must be the dataframe representing LATER data
        
    str_date_list:  list of strings, required
        -best form example: ['Feb 1 2020','Aug 31 2020']
        
    *priority: str, optional (default = 'both combined')
        -describes output visualization.  Valid options include 'both_combined','both_individuals','1','2'
            both_combined writes all NSSP Priority 1&2 to one x axis
            both_individual writes two separate figures for Priority 1 and 2 respectively
            
    *outfile: bool, optional (default = False)
        -writes .html file to folder '../figures/'
        -if str_date_list=['Feb 1 2020','Aug 31 2020'] and priority='both combined',
            outfile has name -> Feb12020_to_Aug312020_priority1and2.html
        
    *show_plot: bool, optional (default = False)
        - displays the figure
        
    *Timed : bool, optional (default = True)
        -gives completion time in seconds
    
    Returns
    -------
    nothing
        
    Requirements
    ------------
    from pj_funcs import *
    
    '''
    # Initialize time
    start_time = time.time()

    # Only Check out priority columns of our dataframes and separate by priority 1 and 2
    
    before1 = priority_cols(df_before, priority='1',
                  extras=['FACILITY_NAME','PATIENT_VISIT_NUMBER','PATIENT_MRN'],
                  drop_cols=['Site_ID','C_Facility_ID'])
    
    before2 = priority_cols(df_before, priority='2',
                  extras=['FACILITY_NAME','PATIENT_VISIT_NUMBER','PATIENT_MRN'])
    
    after1 = priority_cols(df_after, priority='1',
                  extras=['FACILITY_NAME','PATIENT_VISIT_NUMBER','PATIENT_MRN'],
                  drop_cols=['Site_ID','C_Facility_ID'])
    
    after2 = priority_cols(df_after, priority='2',
                  extras=['FACILITY_NAME','PATIENT_VISIT_NUMBER','PATIENT_MRN'])


    #######################################################################

    # Check for completeness and then only look at priority columns
    before1_comp = priority_cols(completeness_facvisits(before1))
    before2_comp = priority_cols(completeness_facvisits(before2))
    after1_comp = priority_cols(completeness_facvisits(after1))
    after2_comp = priority_cols(completeness_facvisits(after2))


    #######################################################################

    # Set the index to what we want (Facility) and sort the index
    before1_comp = (before1_comp.reset_index().drop(['Num_Visits'],axis=1).set_index('Facility')).sort_index()
    before2_comp = (before2_comp.reset_index().drop(['Num_Visits'],axis=1).set_index('Facility')).sort_index()
    after1_comp = (after1_comp.reset_index().drop(['Num_Visits'],axis=1).set_index('Facility')).sort_index()
    after2_comp = (after2_comp.reset_index().drop(['Num_Visits'],axis=1).set_index('Facility')).sort_index()
    
    # Create a combined dataset with priority 1 and 2 elements
    before_combined = before1_comp.join(before2_comp)
    after_combined = after1_comp.join(after2_comp)
    
    

    #######################################################################

    # Find out the differences between the two dates
    #     use RegEx to add - to negative values, + to positive values, and remove 0 change to clear clutter
    priority_one = (after1_comp - before1_comp).astype(float).round(0).astype(int)
    diffs_1 = RegEx_on_Full_DataFrame(RegEx_on_Full_DataFrame(priority_one.astype(str),'^([^-0])','+\g<1>'),'^0$','')

    priority_two = (after2_comp - before2_comp).astype(float).round(0).astype(int)
    diffs_2 = RegEx_on_Full_DataFrame(RegEx_on_Full_DataFrame(priority_two.astype(str),'^([^-0])','+\g<1>'),'^0$','')

    # Sort the indices and create a combined version with prioirty 1 and 2 elements
    diffs_1 = diffs_1.sort_index()
    diffs_2 = diffs_2.sort_index()
    
    diffs_combined = diffs_1.join(diffs_2)

    #######################################################################

    # Based on user input for priority, set a few important values we'll use for plotting
    
    if priority == 'both_combined':
        bases = [before_combined]
        afters = [after_combined]
        diffs = [diffs_combined]
        prio = ['1 and 2']
    
    
    elif priority == 'both_individuals':
        bases = [before1_comp,before2_comp]
        afters = [after1_comp,after2_comp]
        diffs = [diffs_1,diffs_2]
        prio = ['1','2']
        
    elif priority == '1':
        bases = [before1_comp]
        afters = [after1_comp]
        diffs = [diffs_1]
        prio = ['1']
        
    elif priority == '2':
        bases = [before2_comp]
        afters = [after2_comp]
        diffs = [diffs_2]
        prio = ['2']
        
    else:
        print('ERROR: Please enter a correct value for priority\nOptions are \'both_combined\',\'both_individuals\',\'1\', or \'2\'')

    #######################################################################
        
        
    # Loop through the number of plots we'll need to make (either 1 or 2)
    for zz in np.arange(0,len(bases)):
        
        # Get important information
        base = bases[zz]
        after = afters[zz]
        diff = diffs[zz]
        pri = prio[zz]

        # Create a list of lists that describes all hoverover text
        colnames = [list(base.columns)]*len(base)
        hover=[]
        for x in range(len(np.array(base))):
            hover.append(['Facility: '+str(base.index[x])+'<br>'+'Element: '+str(k)+'<br>'+'----'+'<br>'+'Completion Before: ' + str(np.round(i,1)) + '%' + '<br>' + 'Completion After: ' + str(np.round(j,1)) + '%'
                              for i, j, k in zip(np.array(base)[x], np.array(after)[x], colnames[x])])

        #######################################################################

        # Make heatmap elements as simple variables
        heat = np.array(base.astype(float))
        z_text = np.array(diff)
        x = np.array(base.columns)
        y = np.array(base.index)

        # Create two seperate text annotations to be written.  One for positive changes, one for negatives.
        #      we do this because + will have green color, - will have reddish black color
        negs = RegEx_on_Full_DataFrame(pd.DataFrame(z_text).astype(str),'\+.*','')
        poss = RegEx_on_Full_DataFrame(pd.DataFrame(z_text).astype(str),'-.*','')

        #######################################################################

        # Create a large heatmap with descriptive title
        layout_heatmap = go.Layout(
            title=('NSSP Priority '+pri+' Element Completeness Report<br>Heatmap:  Completeness as of '+str_date_list[0]+'<br>Annotations:  Completeness change (%) as of '+str_date_list[1]),
            title_x=0.5,
            xaxis=dict(title='NSSP Priority Element'), 
            yaxis=dict(title='Indiana Facility', dtick=1),
            autosize=False,
            width=1500,
            height=1000
        )

        # Make one heatmap with positive value annotations in green (see font_colors arg)
        ff_fig1 = ff.create_annotated_heatmap(heat,x=list(x), y=list(y),annotation_text=np.array(poss), colorscale='rdylgn',hoverinfo='text',
                                         text=hover,font_colors=['rgb(0, 253, 0)','rgb(0, 253, 0)'],showscale = True)

        # Make exact same heatmap with negative value annotations in reddish-black (see font_colors arg)
        ff_fig2 = ff.create_annotated_heatmap(heat,x=list(x), y=list(y),annotation_text=np.array(negs), colorscale='rdylgn',hoverinfo='text',
                                         text=hover,font_colors=['rgb(76, 0, 0)','rgb(76, 0, 0)'],showscale = True)


        # Append out heatmap, its annotations, and a colorbar to our figure
        fig  = go.FigureWidget(ff_fig1)
        fig.layout=layout_heatmap
        fig.layout.annotations = ff_fig1.layout.annotations + ff_fig2.layout.annotations
        fig.data[0].colorbar = dict(title='Percent Complete', titleside = 'right')
        
        # Read in optional user argument if they want to see plot
        if show_plot == True: 
            iplot(fig)
            
        # Read in optional user argument to see if they want to save file.  I customized the name to be as descriptive as possible.
        if outfile == True:
            filename = ('_to_'.join(str_date_list)+'_priorty'+pri+'.html').replace(' ','')
            fig.write_html(filename)

    # Keep track of end time
    end_time = time.time()
    
    # If user requests to see elapsed time, show them it in seconds
    if Timed == True:
        print('Time Elapsed:   '+str(round((end_time-start_time),3))+' seconds')


###########################################################################################################################################################

def heatmap_compNvalid(df, outfilename=None, daterange=None, hospitals='IHA'):
    
    '''
    Create 2 heatmap subplots of elements that:
        (left) can be assessed for completion
        (right) can be assessed for validity
        
    Input
    -----
    df - pd.Dataframe, required
        Output from NSSP_Element_Grabber() function
    
    outfilename - str, optional
        Specify the name of HTML file to be written to ../figures/   
        *** DO NOT INCLUDE .html ***
    
    daterange - str, optional
        Specify the range that the assessment is being taken over.
        Example:  'Sep 7, 2020 - Sep 14, 2020'
    
    hospitals - str, optional
        Specify the name of the hospitals we are working with
    
    
    Output
    ------
    completion_df - the dataframe that makes up the completion heatmap
    validity_df - the dataframe that makes up the validity heatmap
    
    
    Requirements
    ------------
    from pj_funcs import *
    
    '''
    
    ################################################
    # Validity Check Portion
    ################################################

    # Do validity check and add some cols
    v_check = validity_check(df,Timed=False)
    v_check['Facility'] = df['FACILITY_NAME']
    v_check['VisInd'] = df['FACILITY_NAME'].astype(str)+'|'+df['PATIENT_MRN'].astype(str)+'|'+df['PATIENT_VISIT_NUMBER'].astype(str)

    # Still need to filter by visit/hospital.  Create empty dataframe to append to
    empty = pd.DataFrame(columns=v_check.columns)
    
    # Give array some length & get rid of unnecissary column in output
    empty['Race_Code'] = [np.nan]*len(v_check)
    empty = empty.drop('VisInd',axis=1)

    # Set set arbitrary index to 0 and group by facility
    cur_index = 0
    g1 = v_check.groupby('Facility')

    # loop through facilities
    for index1, frame1 in g1:

        # Group by visit indicator and create empty array to append to
        g2 = frame1.groupby('VisInd')
        l = []

        # loop through indicies
        for index2, frame2 in g2:

            # Drop unneccisary cols
            frame2 = frame2.drop(['Facility','VisInd'],axis=1)
            
            # only take the maximum value (0 or 1) from each column to indicate completion
            l.append(frame2.max().astype(int))

        # Convert 2d list to dataframe
        summary = pd.DataFrame(l)
        
        # Find percent completion of each column (field) for each facility
        a = list((summary.sum()/ len(summary))*100)
        
        # add the facility name to that completion list
        a.append(index1)
        
        # make a new row in the empty dataframe for the hospital.  Add to arbitrary index to move to next call
        empty.iloc[cur_index] = a
        cur_index+=1

    # Valids dataframe - drop only na rows and set the index
    valids = empty.dropna(axis=0,how='all')
    valids = valids.set_index('Facility')
    valids = valids.sort_index()
    
    # Create a list of lists that describes all hoverover text
    colnames = [list(valids.columns)]*len(valids)
    valids_hover=[]
    for x in range(len(np.array(valids))):
        valids_hover.append(['Facility: '+str(valids.index[x])+'<br>'+'Element: '+str(j)+'<br>'+'----'+'<br>'+'Percent Valid: ' + str(np.round(i,1)) + '%'
                        for i, j in zip(np.array(valids)[x], colnames[x])])


    ################################################
    # Completeness Report Portion
    ################################################

    # Check for completeness and format the way we want
    comps = completeness_facvisits(df,Timed=False)
    comps = priority_cols(comps,drop_cols=['Site_ID','C_Facility_ID'])
    comps = comps.reset_index().drop('Num_Visits',axis=1).set_index('Facility')
    
    # We only want completion columns that don't exist in the validity check
    comp_cols = list(set(comps.columns) - set(valids.columns))
    comp = comps[comp_cols]
    comp = comp.sort_index()
    
    # Create a list of lists that describes all hoverover text
    colnames = [list(comp.columns)]*len(comp)
    comp_hover=[]
    for x in range(len(np.array(comp))):
        comp_hover.append(['Facility: '+str(comp.index[x])+'<br>'+'Element: '+str(j)+'<br>'+'----'+'<br>'+'Percent Valid: ' + str(np.round(i,1)) + '%'
                        for i, j in zip(np.array(comp)[x], colnames[x])])


    #############################################################################3

    ########################
    # Make heatmaps!
    ########################
    
    # Heatmap for completion
    plt1 = go.Heatmap(z=np.array(comp.astype(float)),
                      x = comp.columns,y=comp.index,
                      colorscale='rdylgn',
                     hoverinfo='text',hovertext=comp_hover)

    # Heatmap for validity
    plt2 = go.Heatmap(z=np.array(valids.astype(float)),
                      x = valids.columns,y=valids.index,
                      colorscale='rdylgn',
                     hoverinfo='text',hovertext=valids_hover)


    # Append out heatmap, its annotations, and a colorbar to our figure
    fig = make_subplots(rows=1, cols=2,
                        column_widths=[0.58, 0.42], shared_yaxes=True,
                       subplot_titles=('Elements Assessed for Completion',
                                      'Elements Assessed for Completion AND Validity'),
                       horizontal_spacing=0.01)
    
    
    ######################################################################

    # Gridlines


    # Horizontal Lines LEFT plot
    for i in np.arange(0,len(comp.index)+1):
        fig.add_shape(type="line", x0=0-0.5, y0=i-0.5, x1=len(comp.columns)-0.5, y1=i-0.5,
                    line=dict(color="black",width=2),row=1,col=1)

    # Vertical lines LEFT plot
    for i in np.arange(0,len(comp.columns)+1):
        fig.add_shape(type="line", x0=i-0.5, y0=0-0.5, x1=i-0.5, y1=len(comp.index)-0.5,
                    line=dict(color="black",width=2), row=1,col=1)


    ################################################

    # Horizontal Lines RIGHT plot
    for i in np.arange(0,len(valids.index)+1):
        fig.add_shape(type="line", x0=0-0.5, y0=i-0.5, x1=len(valids.columns)-0.5, y1=i-0.5,
                    line=dict(color="black",width=2),row=1,col=2)

    # Vertical Lines RIGHT plot
    for i in np.arange(0,len(valids.columns)+1):
        fig.add_shape(type="line", x0=i-0.5, y0=0-0.5, x1=i-0.5, y1=len(valids.index)-0.5,
                    line=dict(color="black",width=2), row=1,col=2)

######################################################################
    
    # Add the subplots to the figure
    fig.add_trace(plt1,row=1,col=1)
    fig.add_trace(plt2,row=1,col=2)
    

    # Annotate the colorbar
    fig.data[1].colorbar = dict(title='Percent Complete/Valid', titleside = 'right')

    # Work the titles
    if hospitals:
        if daterange:
            fig.update_layout(height=600, width=1600, title_text=hospitals+" Hospitals assessed for HL7 Message Completion/Validity<br>"+daterange,
                             title_x = 0.603)
    else:
        fig.update_layout(height=600, width=1600, title_text='HL7 Message Completion/Validity Report',
                             title_x = 0.603)
            
    # Rotate the x tick labels
    fig.update_xaxes(tickangle = 90)

    # Show it
    fig.show()

    # Write to outfile
    if outfilename:
        fig.write_html(outfilename+".html")

        
    return comp,valids



###########################################################################################################################################################
###########################################################################################################################################################





