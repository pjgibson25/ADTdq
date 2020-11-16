# ADTdq

[Github Project](https://github.com/pjgibson25/ADTdq)


## Setup

-----------------------

First off, please make sure you have a Python version >=3.6 .
If you don't have Python, you can get it by downloading [Anaconda](https://docs.anaconda.com/anaconda/install/).

You will likely have to install a few supporting packages.  In your command prompt or terminal (dependent on OS type), please type the following required dependencies.

    pip install hl7
    pip install regex
    pip install plotly
    pip install tqdm
    pip install ipywidgets
    pip install xlrd

and if you didn't already install my package,

    pip install ADTdq


## Background 

-----------------------

#### How it Started

My name is PJ Gibson and I am a data analyst with the Indiana State Department of Health.
My Informatics department arranged a grant with a group who could improve the quality of hospital reporting.
We needed to track the progress of this hospital reporting, which required diving into HL7 Admission/Discharge/Transfer (ADT) messages and assessing for data completion and validity.
Enter me.


#### The Goal

The main purpose of this package is to give data quality analysis functions to workers in public health informatics. 




## Functions
<p>(click on function name for extended description)</p>
-----------------------

<details>
<summary>list_elements</summary>
  
## Documentation    

    list_elements(include_priority=False):
    
    Displays all potential elements we can search for

    Parameters
    ----------
    include_priority: bool, optional (default is False)  
        - returns 2 column pandas dataframe.  Element Name & Priority


    Returns
    -------
    np.array() (list-like) that contains all elements we can search for
    dataframe IF include_priority = True


## Code Examples 
    
```
# import the library and all its functions
from HL7reporting import *

# set pandas setting to disply a max of 100 rows.    
pd.options.display.max_rows = 100   
    
# save elements/priority as 2 column pandas dataframe.
a = list_elements(include_priority=True)
a

```
## Visualization of Output

<img src="https://github.com/pjgibson25/HL7reporting/raw/master/pics/list_elements.png" alt="list_elements">

<br>
</details>




<details>
<summary>NSSP_Element_Grabber</summary>
  
## Documentation    

    NSSP_Element_Grabber(data,explicit_search=None,Priority_only=False,outfile='None',no_FAC=False,no_MRN=False,no_VisNum=False):
    
    Creates dataframe of important elements from PHESS data.
    Timed with cool updating progressbar (tqdm library).

    NOTE: Your input should contain the column titles:
	   MESSAGE , FACILITY_NAME
    

    Parameters
    ----------
    data: pandas DataFrame, required
	- input containing columns MESSAGE, FACILITY_NAME

    explicit_search: list, optional (default is None)
	- list of priority element names you want specifically.
	  Use argument-less list_elements() function to see all options

    Priority_only:  bool, optional (default is False)  
        - If True, only gives priority 1 or 2 elements

    outfile:  str, optional (default is 'None')
        - Replace with file name for dataframe to be wrote to as csv
            Will be located in working directory.
            DO NOT INCLUDE .csv IF YOU CHOOSE TO MAKE ONE

    no_FAC: Bool, optional (default is False)
	- If you don't have a FACILITY_NAME in your input, change to True
	  NOTE: without a FACILITY_NAME, usage of other functions within library can return errors

    no_MRN: Bool, optional (default is False)
	- If you do not want output to contain MRN information, change to True
	  NOTE: without a MRN, usage of other functions within library can return errors

    no_VisNum: Bool, optional (default is False)
	- If you do not want output to contain patient_visit_number information, change to True
	  NOTE: without a VisNum, usage of other functions within library can return errors
    
    Returns
    -------
    dataframe
        

## Code Examples 
    
```
# import the library and all its functions
from HL7reporting import *

# read in data
data1 = pd.read_csv('somefile.csv',engine='python')

# process through NSSP_Element_Grabber() function
parsed_df = NSSP_Element_Grabber(data1, Priority_only=True,outfile='nameofoutputfile')

```

*if you don't have no facility_name

```
data1 = pd.read_csv('somefile.csv',engine='python')

# process through NSSP_Element_Grabber() function
parsed_df = NSSP_Element_Grabber(new_input_dataframe, Priority_only=True,outfile='outfilename', no_FAC=True)
```

## Visualization of Output

<img src="https://github.com/pjgibson25/HL7reporting/raw/master/pics/nssp_element_grabber.png" alt="nssp_element_grabber_visual">

*note personal details are replaced with random ints and NaN values
<br>
</details>



<details>
<summary>priority_cols</summary>
  
## Documentation    

	priority_cols(df, priority='both', extras=None, drop_cols=None)

    Spits out NSSP priority columns from a dataframe.
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

   
## Code Examples 
    

```
# import the library and all its functions
from ADTdq import *

# read in data
data1 = pd.read_csv('somefile.csv',engine='python')

# process through NSSP_Element_Grabber() function
parsed_df = NSSP_Element_Grabber(data1,Timed=False,
                                    Priority_only=True,outfile='None')


# take the priority element columns from our output dataframe
#### remove two columns that are processed backend (always NaN)
only_priority1_df = priority_cols(parsed_df,priority='1',drop_cols=['Site_ID','C_Facility_ID'])

```

## Visualization of Output

<img src="https://github.com/pjgibson25/ADTdq/raw/master/pics/priority_cols.png" alt="priority_cols_Visual">

*note personal details are replaced with random ints and NaN values
*also note the lower number of columns
<br>
</details>





<details>
<summary>Visualization_interactive</summary>
  
## Documentation    

    Visualization_interactive(df_before,df_after,str_date_list,priority='both_combined',grid=True,outfile=False,show_plot=False,Timed=True):


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
    
    *grid: bool, optional (default = True)
	-describes output visualization.  Draws grid lines over all heatmap cells.
	    NOTE: cyan line divides priority 1 and priority 2 elements regardless of argument.
		  only relevant for priority->both combined            

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
        
   
## Code Examples 
    

```
# import the library and all its functions
from HL7reporting import *

# Read in the two datasets (already ran NSSP_Element_Grabber on)
before = pd.read_csv('path_to_parsed_df_file1',engine='python')
after = pd.read_csv('path_to_parsed_df_file2',engine='python')

Visualization_interactive(before,after,['Oct 11 2020','Oct 28 2020'],priority='both_combined',outfile=True,show_plot=False)

```

## Visualization of Output

<img src="https://github.com/pjgibson25/HL7reporting/raw/master/pics/Visualization_interactive.png" alt="Visualization_interactive_Visual">

*note that this image above is simply an image.  In reality the output is an interactive HTML file with hover_over capabilities
*also note that the y axis is marked over and typically contains facility names.
<br>
</details>








<details>
<summary>issues_in_messages</summary>
  
## Documentation    

    issues_in_messages(df, Timed=True, combine_issues_on_message = False, split_issue_column = False):
 
    Description
	----------
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

## Code Examples 
    
Version 1:
```
# import the library and all its functions
from ADTdq import *

# read in data
data1 = pd.read_csv('somefile.csv',engine='python')

# process through NSSP_Element_Grabber() function
parsed_df = NSSP_Element_Grabber(data1,Timed=False,
                                    Priority_only=True,outfile='None')


# Find issues in messages
split_by_issue = issues_in_messages(parsed_df, split_issue_column=True)

# Get the facility name from the grouper ID
split_by_issue['Fac_Name'] = split_by_issue.Grouper_ID.str.split('\|').str[0]

# First sort the values so that all facility rows are next to one another, then by message similarly
split_by_issue = split_by_issue.sort_values(['Fac_Name','Grouper_ID','MESSAGE','Priority'])

# Set the indices so that when we export to excel, the index cells will merge making it look pretty
split_by_issue = split_by_issue.set_index(['Fac_Name','Grouper_ID','MESSAGE','Issue_Type'])

# Send it to an excel file!
split_by_issue.to_excel('split_by_issue_version1.xlsx')
```
Version 2:
```
# import the library and all its functions
from ADTdq import *

# read in data
data1 = pd.read_csv('somefile.csv',engine='python')

# process through NSSP_Element_Grabber() function
parsed_df = NSSP_Element_Grabber(data1,Timed=False,
                                    Priority_only=True,outfile='None')


# Find issues in messages
comb_issues = issues_in_messages(parsed_df, combine_issues_on_message=True)

# Get the facility name
comb_issues['Fac_Name'] = comb_issues.Grouper_ID.str.split('\|').str[0]

# Make first issue start with bullet point
comb_issues['Issue'] = comb_issues['Issue'].str.replace('^(.*)','• \g<1>',regex=True)

# Make each new line have a bullet point.
comb_issues['Issue'] = comb_issues['Issue'].str.replace('\n','\n• ')

# First sort the values so that all facility rows are next to one another, then by message similarly
comb_issues = comb_issues.sort_values(['Fac_Name','Grouper_ID','MESSAGE'])

# Set the indices so that when we export to excel, the index cells will merge making it look pretty
comb_issues = comb_issues.set_index(['Fac_Name','Grouper_ID','MESSAGE','Issue'])

# Send it to an excel file!
comb_issues.to_excel('comb_issue_version2.xlsx')
```

## Visualization of Output

Version 1
<img src="https://github.com/pjgibson25/ADTdq/raw/master/pics/issues_in_messages_v1.png" alt="issues_in_messages_Visual1">
<br>

Version 2 
<img src="https://github.com/pjgibson25/ADTdq/raw/master/pics/issues_in_messages_v2.png" alt="issues_in_messages_Visual2">
<br>
</details>












<details>
<summary>validity_check</summary>
  
## Documentation    

    validity_check(df, Timed=True)
    
    Checks to see which elements in a dataframe's specific NSSP priority columns meet NSSP validity standards.
    Returns a True/False dataframe with FACILITY_NAME,PATIENT_MRN,PATIENT_VISIT_NUMBER as only string-type columns
    
    Parameters
    ----------
    
    df - required, pandas Dataframe, output from NSSP_Element_Grabber() function    
    Timed - optional, boolean (True/False), default is True.  Returns time in seconds of completion.
    
    Returns
    --------
    validity_report - True/False dataframe with FACILITY_NAME,PATIENT_MRN,PATIENT_VISIT_NUMBER as only string-type columns
    
   
## Code Examples 
    

```
# import the library and all its functions
from ADTdq import *

# read in data
data1 = pd.read_csv('somefile.csv',engine='python')

# process through NSSP_Element_Grabber() function
parsed_df = NSSP_Element_Grabber(data1,Timed=False,
                                    Priority_only=True,outfile='None')


# take the priority element columns from our output dataframe
#### remove two columns that are processed backend (always NaN)
only_priority1_df = priority_cols(parsed_df,priority='1',drop_cols=['Site_ID','C_Facility_ID'])

# run the validity check function on it
val = validity_check(only_priority1_df)

```

## Visualization of Output

<img src="https://github.com/pjgibson25/ADTdq/raw/master/pics/validity_check.png" alt="validity_check_Visual">

*note the lower number of columns.  Not all columns able to be assessed for validity
<br>
</details>






<details>
<summary>validity_and_completeness_report</summary>
  
## Documentation    

    validity_and_completeness_report(df,description='long',visit_count=False,outfile=None, Timed=True)
    
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
        if True, send excel file (in current directory).  Name defined by outfile
        *DO NOT INCLUDE .xlsx or full path
    
    
    Returns
    -------
    df1
        Dataframe showing issues in messages for each hospitals.  Report structure
    
    df2
        Dataframe assessing all messages for incomlete,invalid,valid elements represented as 0s, 1s, and 2s
        
   
## Code Examples 
    

```
# import the library and all its functions
from ADTdq import *

# read in data
data1 = pd.read_csv('somefile.csv',engine='python')

# process through NSSP_Element_Grabber() function
parsed_df = NSSP_Element_Grabber(data1,Timed=False,
                                    Priority_only=True,outfile='None')


# run the validity function on it
val = validity_and_completeness_report(parsed_df, description='long')[0] # don't care about array of 0, 1, 2 for now


```

## Visualization of Output[0]

<img src="https://github.com/pjgibson25/ADTdq/raw/master/pics/validity_and_completeness_report.png" alt="validity_and_completeness_report_Visual">

<br>
</details>




<details>
<summary>heatmap_compNvalid</summary>
  
## Documentation    

    heatmap_compNvalid(df, outfilename=None, daterange=None, hospitals='IHA')
    
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
    
   
## Code Examples 
    

```
# import the library and all its functions
from ADTdq import *

# read in data
data1 = pd.read_csv('somefile.csv',engine='python')

# process through NSSP_Element_Grabber() function
parsed_df = NSSP_Element_Grabber(data1,Timed=False,
                                    Priority_only=True,outfile='None')


heatmap_compNvalid(parsed_df, outfilename='heatmap visualization completion and validity')

```

## Visualization of Output

<img src="https://github.com/pjgibson25/ADTdq/raw/master/pics/heatmap_compNvalid.png" alt="heatmap_compNvalid_Visual">

*note that typically the y-axis will show facility names.  Hidden here for confidentiality.
<br>
</details>



<details>
<summary>Visualize_Facility_DQ</summary>
  
## Documentation    

    Visualize_Facility_DQ(df, fac_name, hide_yticks = False, Timed = True)
    
	
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

## Code Examples 
    

```
# import the library and all its functions
from ADTdq import *

# read in data
data1 = pd.read_csv('somefile.csv',engine='python')

# process through NSSP_Element_Grabber() function
parsed_df = NSSP_Element_Grabber(data1,Timed=False,
                                    Priority_only=True,outfile='None')

# produce the visualization
visual = Visualize_Facility_DQ(parsed_df, 'hospital_name')
```

## Visualization of Output

<img src="https://github.com/pjgibson25/ADTdq/raw/master/pics/Visualize_Facility_DQ.png" alt="Visualize_Facility_DQ_Visual">

*note that this only produces the visualization for 1 facility

<br>
</details>







## FAQs
-----------------------

#### Where can I access function documentation outside of this location?

Within a Jupyter Notebook document, you can type:

``FunctionNameHere?`` 

into a jupyter notebook cell and then run it with `SHIFT` + `ENTER`.
The output will show you all the function documentation including a brief description and argument descriptions.


#### Why Python?

I work entirely in Python.
In the field of public health informatics, SASS is the most popular programming language, perhaps followed by R (at least in syndromic surveillance).
I have created this package to run as intuitively as possible with a minimal amount of python knowledge.
I could be wrong, but I believe that one day, public health informatics may become Python-dominant, so this package could help as an introduction to the environment to those unfamiliar.

#### For plottting, what if I want to make small changes such as color changes, formatting, or simple customizing?

Right now I don't have things set up for that sort of work.  My best solution would be for you to dive into my Github reposiory python file linked [here](https://github.com/pjgibson25/ADTdq/blob/master/ADTdq/__init__.py).  You can copy the defined functions into your document and make minor adjustments as you see fit.


#### Why isn't one of my functions working?

The most common problem in this situation is a incorrectly formatted input to a function.  Most of DQ functions stems from an intial NSSP_Element_Grabber() run.  The input to this function should contain the following columns:

`['MESSAGE','FACILITY_NAME'] `

Note that you can pass the argument No_FAC, No_VisNum, or No_MRN.  When these optional arguments are passed, the resulting output could return erros when used in conjunction with other DQ functions.
This is because many of my functions collapse messages into individual visits.  The collapsing process requires Facility, MRN, and Visit Number. Missing elements in these fields throws a wrench into the process.


#### My version is out of date (there has been a more recent release).  How do I update?

Type the following into your command line / terminal

    pip install ADTdq --update


#### My question isn't listed above...what should I do?

feel free to contact me at:

PGibson@isdh.IN.gov 

with any additional questions.

## The Author
PJ Gibson - Data Analyst for Indiana State Department of Health

## Special Thanks
* Harold Gil - Director of Public Health Informatics for Indiana State Department of Health.
Harold assigned me this project, gave me relevant supporting documentation, and helped me along the way with miscellaneous troubleshooting.
* Matthew Simmons - Data Analyst for Indiana State Department of Health.
Matthew helped walk me through some troubleshooting and was a supportive figure throughout the project.
* Ben Sewell, Shuennhau Chang, Logan Downing, Ryan Hastings, Nicholas Hinkley, Rachel Winchell.
Members of my informatics team that also supported me indirectly!
