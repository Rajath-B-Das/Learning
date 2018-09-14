# Configure spark environment
import os
import sys
spark_home = '/usr/local/spark' # point to your spark home dir
os.environ['SPARK_HOME'] = spark_home
sys.path.insert(0, spark_home + "/python")
os.environ['PYSPARK_PYTHON'] = 'python2.7'

# Import packages
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.classification import RandomForestClassificationModel
#from pyspark.ml.regression import RandomForestRegressor as RF1
import csv, threading
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType
#%matplotlib inline
from pyspark.ml.feature import IndexToString

from datetime import datetime, date
import unicodedata

import gc

#### Bootstrapping for Confidence bands
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import cPickle

# Model quality checks
import scipy
from pyspark.mllib.evaluation import BinaryClassificationMetrics as classi_metrics
# Plotting the ROC curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

#sys.path.append(user_path + '/analytics_starter-master/lib/' )
#from python_hive_functions import *
    
username = os.environ.get('USER')
username_hive = username.replace( "-", "_" )

from os.path import expanduser, join
from pyspark.sql import SparkSession
from pyspark.sql import Row

    
def sql_read(path = ""):
    with open (path, "r") as myfile:
        text = myfile.read()
    return text    
    
# warehouse_location points to the default location for managed databases and tables
warehouse_location = 'spark-warehouse'

confList = [
    #["spark.streaming.unpersist", "true"]           # Automated memory management for Streaming apps
    ["spark.driver.cores","1"]
    ,["spark.driver.memory","2g"]
    ,["spark.driver.maxResultSize", "1g"]
    ,["spark.executor.memory","32g"]               # Amount of memory for each executor (shared by all tasks in the executor)
    ,["spark.mesos.coarse", "true"]                 # Accept and hold a fixed amount of resource from Mesos
    #,["spark.dynamicAllocation.enabled","true"]
    #,["spark.dynamicAllocation.initialExecutors","50"]
    #,["spark.dynamicAllocation.minExecutors","50"]
    ,["spark.storage.memoryMapThreshold","200m"]
#    ,["spark.serializer","org.apache.spark.serializer.KryoSerializer"] #Define Kryo as Base Serializer
#    ,["spark.kryoserializer.buffer","128k"]         #Define base memory per node
#    ,["spark.kryoserializer.buffer.max","2047"]  #Define max value that Kryo Serializer can run upto
    ,["spark.rdd.compress", "true"]                 # Enabling RDD compress to optimize storeage space
    ,["spark.cores.max",32]                      # Defines the number of cores
    ,["spark.executor.heartbeatInterval", "10000s"]
    ,["spark.network.timeout", "10000s"]
    ,["spark.shuffle.io.connectionTimeout","10000s"]
    #,["spark.task.cpus", "4"]
    ,["spark.rpc.numRetries", "100"]
    ,["spark.task.maxFailures", "10"]
    ,["spark.executors.max","1"]
    ,["spark.memory.fraction","0.8"]
#    ,["spark.shuffle.io.connectionTimeout","120s"]
    ,["spark.shuffle.file.buffer","128k"]
    ,["spark.shuffle.compress","true"]
    ,["spark.shuffle.spill.compress","true"]
]

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row

SPARK_CONF = SparkConf().setAppName("Schedule_Dotcom_2x0").setAll(confList)

#spark = SparkSession.builder\
#    .config(conf=SPARK_CONF) \
#    .enableHiveSupport().getOrCreate()

#spark.sql("show tables in sbajaj").head()

spark = SparkSession.builder    .config(conf=SPARK_CONF)     .enableHiveSupport().getOrCreate()

# spark.sparkContext.parallelize(range(1000)).map(str).countApproxDistinct()


######################################################################################
# function to read an external script to a string
def read_sql(path = ""):
    with open (path, "r") as myfile:
        text = myfile.read()
    return text

######################################################################################
# function to concatenate multiple files in a folder
def concatenate_files(path = False, file_match = '*.csv'):
    import glob
    import pandas as pd
    
    allFiles = glob.glob(path + "/"+file_match)
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    return frame
    


#######################################################################################
#export MAPR_MAPREDUCE_MODE=yarn

def execute_hql(hostname = False, 
                hive_env_options = "",
                hql_statement = ";", parse_dates=True):
    """
    # https://community.hortonworks.com/questions/24953/solution-for-hive-runtime-error-while-processing-r.html
    
    This function enables you to remotely execute a HiveQL statement as if you were logged into a
    Hadoop edge server, and conveniently get the output back as a dataframe. This is very helpful 
    if, for example, the Hive server doesn't accept JDBC connections.
    This function assumes that you have configured passwordless SSH to the edge server. If you haven't
    done so, then you'll be prompted for a password. If you are invoking this function from a notebook, 
    then the password prompt will appear in the terminal window from which you launched the notebook.
    :param hostname: The hostname of the Hadoop edge server
    :param hql_statement: The HiveQL statement you want to execute
    :param parse_dates: If True, automatically parse any columns with 'date' in the name as a date; also, if there is only one date column, then use that as the index
    :returns: A dataframe containing the output from the HiveQL statement  
    :Example:
    >>> import hql
    >>> hostname = 'bfd-main-client00.sv.walmartlabs.com'
    >>> hql_statement = 'show databases;'
    >>> df = hql.execute_hql(hostname, hql_statement)
    
    Created by: George Roumeliotis
    Last modified by: Sumit Bajaj
    Last modified on: July 02, 2017
    """

    import subprocess
    import StringIO
    import pandas
    import datetime
    import time
    
    
    start_time = datetime.datetime.now()
    
    log_path = './logs/'
    
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
        
    
    log_file_path = log_path + "hive" + datetime.datetime.strftime(datetime.datetime.now(), "-%Y-%m-%d-%H-%M-%S ") + ".log"


    
    #err_file = open("hive.log", "wb")
    err_file = open(log_file_path, "wb")

    
    cli_opts = """set hive.auto.convert.join=True;
                 set hive.exec.parallel=True;
                 set hive.cli.print.header=true;
                 set hive.exec.compress.output=true;
                 set hive.vectorized.execution.enabled = true;
                 set hive.vectorized.execution.reduce.enabled = true;
                 set hive.cbo.enable=true;
                 set hive.compute.query.using.stats=true;
                 set hive.stats.fetch.column.stats=true;
                 set hive.stats.fetch.partition.stats=true;
                 set hive.variable.substitute=true;
                 SET hive.exec.dynamic.partition=true;
                 SET hive.exec.dynamic.partition.mode=nonstrict;
                 SET mapred.map.output.compression.codec=org.apache.hadoop.io.compress.SnappyCodec;
                 SET hive.exec.compress.intermediate=true;
                 SET mapred.output.compression.type=BLOCK;
                 SET mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec;
                 SET hive.enforce.bucketing=true;
                 set mapreduce.reduce.speculative=True;
                 set mapreduce.map.speculative=True;
                 set mapred.tasktracker.expiry.interval=1800000;
                 set mapred.task.timeout= 1800000;
                 """

    # Assemble local operating system command
    #hql_stmt = "hive -S -e " + "\"""" + cli_opts + " " + hql_statement + ';'  + "\"""" 
    hql_stmt = hive_env_options + " /usr/local/bin/hive -e " + "\"""" + cli_opts + " " + hql_statement + ';'  + "\"""" 
    #hql_stmt = 'hive -S -e "{cli_opts} %s"'%hql_statement
    #hql_stmt = hql_stmt.format(cli_opts = cli_opts)
    
    # only ssh if a hostname is specified
    if hostname:
        cmd = ['ssh', hostname]
        cmd.append(hql_stmt)
        # Execute local operating system command
        try:
            output = subprocess.check_output(cmd, stderr = err_file)
        except:
            print "Check log for errors"
    else:
        cmd = [hql_stmt]
        # Execute local operating system command
        output = subprocess.check_output(cmd, stderr = err_file, shell=True)

    
    #print cmd

    # Execute local operating system command
    #output = subprocess.check_output(cmd, shell=True)
    
    # Auto detect date column for use as index
    try:
        df = pandas.read_csv(StringIO.StringIO(output), sep='\t', nrows=1)
    except:
        print 'Finished execution!'
        return
    date_cols = list(df.columns[df.columns.to_series().apply(lambda x: 'date' in x)])

    # If only one date column found, then use as index, otherwise just parse all dates and use integer index
    if (len(date_cols) == 1) and parse_dates:
        df = pandas.read_csv(StringIO.StringIO(output), sep='\t', parse_dates=date_cols, index_col=date_cols[0])
    elif parse_dates:
        df = pandas.read_csv(StringIO.StringIO(output), sep='\t', parse_dates=date_cols)
    else:
        df = pandas.read_csv(StringIO.StringIO(output), sep='\t')
        
    df = pandas.read_csv(StringIO.StringIO(output), sep='\t')
    
    
    header = list(df[:0])

    # strip table/view name from column names
    df.columns = map(lambda x:x.split('.')[len(x.split('.')) - 1], header)

    
    #df.rename(columns=lambda x: str.upper(x), inplace=True)
    
    try:
        #df.rename(columns=lambda x: x.split('.')[1], inplace=True)
        print  "time taken - ", datetime.datetime.now() - start_time # print elapsed time

        return df
    except:
        print  "time taken - ", datetime.datetime.now() - start_time # print elapsed time

        return df
    
    print  "time taken - ", datetime.datetime.now() - start_time # print elapsed time


    # Return dataframe
    return df

print("Spark session initialized")


# ## Please enter the following parameters

# ### Set parameter prefixes

# In[2]:


import getpass
#Getting the schema name
schema_name = getpass.getuser()
print(schema_name)


# In[3]:

max_week_holder="mjoshi2.prop_mod_max_week_holder"
calendar="us_core_dim.calendar_dim"

q="""select min(calendar_date), min(wm_ly_comp_date)
from 
{} cal
inner join
(select max(week_id) as week_id from {}) max_week_holder
on cal.wm_year_wk_nbr=max_week_holder.week_id + 1
""".format(calendar, max_week_holder)
df=spark.sql(q).toPandas()
WEEK_ST_DT= str(df.iloc[0,1])
SCORING_WEEK_ST_DT = str(df.iloc[0,0])
print(WEEK_ST_DT)
print(SCORING_WEEK_ST_DT)


# In[4]:


feature_cols_table='mjoshi2.prop_and_exp_value_features_as_cols'
    
colsnow_cols_table='mjoshi2.prop_and_exp_value_colsnow_as_cols'

  
#For building
wkstdt="\'{}\'".format(WEEK_ST_DT)


#For scoring
score_wkstdt="\'{}\'".format(SCORING_WEEK_ST_DT)
    

# In[6]:


# Preperiod condition to subset customer base
pre_period_conditions="""((traced_flag=1 and f_store_exc_ct_dt>=2 and coalesce(f_og_exc_ct_dt,0)+coalesce(f_dotcom_exc_ct_dt,0)=0))"""

# Target variable
activation_flag='activation_flag_90'


# In[7]:


#year = datetime.datetime.strptime(WEEK_ST_DT, '%Y-%m-%d' ).year
#month = datetime.datetime.strptime(WEEK_ST_DT, '%Y-%m-%d' ).month
#day = datetime.datetime.strptime(WEEK_ST_DT, '%Y-%m-%d' ).day

year = datetime.strptime(WEEK_ST_DT, '%Y-%m-%d' ).year
month = datetime.strptime(WEEK_ST_DT, '%Y-%m-%d' ).month
day = datetime.strptime(WEEK_ST_DT, '%Y-%m-%d' ).day

if month<10:
    month_str='0'+str(month)
else:
    month_str=str(month)
    
if day<10:
    day_str='0'+str(day)
else:
    day_str=str(day)
    
    
week_start_date_suffix=str(year)+'_'+month_str+'_'+day_str


# In[8]:


#year = datetime.datetime.strptime(SCORING_WEEK_ST_DT, '%Y-%m-%d' ).year
#month = datetime.datetime.strptime(SCORING_WEEK_ST_DT, '%Y-%m-%d' ).month
#day = datetime.datetime.strptime(SCORING_WEEK_ST_DT, '%Y-%m-%d' ).day

year = datetime.strptime(SCORING_WEEK_ST_DT, '%Y-%m-%d' ).year
month = datetime.strptime(SCORING_WEEK_ST_DT, '%Y-%m-%d' ).month
day = datetime.strptime(SCORING_WEEK_ST_DT, '%Y-%m-%d' ).day

if month<10:
    month_str='0'+str(month)
else:
    month_str=str(month)

if day<10:
    day_str='0'+str(day)
else:
    day_str=str(day)
    
scoring_week_start_date_suffix=str(year)+'_'+month_str+'_'+day_str


# In[9]:


# Desired dataset (user.table_name)
dataset_schema='mjoshi2'

dataset_name_prefix = '.mu_rfm_metrics_w_pu_bb_flags_added_w_mdse_ogsd'

#Customer Base Definition
cust_base='2x0_dotcom'
post_period_suffix='_90d'

#Table name with which training dataset should be stored
training_dataset_prefix=schema_name + '.mu_prop_mod_'+cust_base+'_training_data'+post_period_suffix

#Table name with which test dataset should be stored
test_dataset_prefix=schema_name + '.mu_prop_mod_'+cust_base+'_test_data'+post_period_suffix

# Table to store final propensity scores and features (test data)
HH_propensity_final_table_test_prefix= schema_name + '.prop_mod_'+cust_base+'_prop_fin_test'+post_period_suffix

# Table to store final propensity scores and features (all HH)
HH_propensity_final_table_all_hh_prefix= schema_name + '.prop_mod_'+cust_base+'_prop_fin_all_hh'+post_period_suffix

# Enter the desired name for the model
model_name_prefix="online_activation_"+cust_base+"_model"+post_period_suffix

# Scoring dataset name
scoring_dataset_name_prefix='aanand2.mu_rfm_metrics_w_pu_bb_flags_added_w_mdse_ogsd'

#Scored dataset name prefix
scored_dataset_name_prefix = schema_name + '.prop_mod_scored_'+cust_base+post_period_suffix


# In[10]:


# Desired dataset (user.table_name)
dataset_name = dataset_schema + dataset_name_prefix + '_sq'+'_' + week_start_date_suffix + '_sbctqnt_w_coded_demogs'

# Table name with which stratified sample should be saved (user.table_name)
strat_table = schema_name + dataset_name_prefix + '_'+cust_base+'_' + week_start_date_suffix + '_strat'

#Table name with which training dataset should be stored
training_dataset= training_dataset_prefix + '_'+week_start_date_suffix

#Table name with which test dataset should be stored
test_dataset= test_dataset_prefix + '_'+week_start_date_suffix

# Table to store final propensity scores and features (test data)
HH_propensity_final_table_test=HH_propensity_final_table_test_prefix + '_' +week_start_date_suffix

# Table to store final propensity scores and features (all HH)
HH_propensity_final_table_all_hh= HH_propensity_final_table_all_hh_prefix + '_'+week_start_date_suffix

# Enter the desired name for the model
model_name= model_name_prefix + '_'+week_start_date_suffix


# Scoring dataset name
scoring_dataset_name = scoring_dataset_name_prefix + '_'+scoring_week_start_date_suffix

#Scored dataset name
scored_dataset_name= scored_dataset_name_prefix + '_'+scoring_week_start_date_suffix


# In[11]:


print("Names Assigned\n")

print strat_table
print dataset_name
print model_name
print HH_propensity_final_table_test


# #### Stratification

# In[ ]:


query0="""drop table if exists {}""".format(strat_table)
spark.sql(query0)


# In[16]:


query="""create table {} stored as ORC as
with base as (
select *, 
if(post_order_cnt_dotcom_inc_ct_dt_15d >0,1,0) as activation_flag_15,
if(post_order_cnt_dotcom_inc_ct_dt_90days >0,1,0) as activation_flag_90,
if(post_order_cnt_dotcom_inc_ct_dt_30d >0,1,0) as activation_flag_30
from {}
where {}
),
base1 as (select *, if(visit_cnt_app_90>0,1,0) as app_stratifier from base),
base2 as (
SELECT *, dense_rank() OVER (PARTITION BY traced_flag, f_store_exc_ct_dt,r_store_exc_ct_dt,
f_dotcom_exc_ct_dt,r_dotcom_exc_ct_dt,app_stratifier  ORDER BY rand() ) as sample_rank
from base1 )
select *
from base2 
where sample_rank <= 1000""".format(strat_table, dataset_name, pre_period_conditions)
#query
execute_hql(hql_statement=query, hive_env_options='export MAPR_MAPREDUCE_MODE=yarn;')


# In[ ]:


print("Stratifiction Completed")


# ## Sample for creating train and test sets

# In[18]:


spark.catalog.clearCache()


# In[19]:


## Get stratified sample in dataframe
fetch_strat_sample_query = "select * from {} where traced_flag=1".format(strat_table)
fetch_strat_sample_query


# In[20]:


## From stratified sample get training sample
hh_train_for_sample = spark.sql(fetch_strat_sample_query)


# In[21]:


count=hh_train_for_sample.count()


# In[22]:


count


# In[23]:


sample_fraction = 2000000.0/count


# In[24]:


if count>2500000:
    hh_train_sample=hh_train_for_sample.sample(False, sample_fraction,0)
else:
    hh_train_sample=hh_train_for_sample


# In[25]:


hh_train_sample.registerTempTable("control_dataset_overall")


# In[3]:


print("Caching hh_train_sample")


# In[26]:


hh_train_sample.cache()
hh_train_sample.count()


# In[27]:


query="""select count(*) from 
          (select household_id, traced_flag from control_dataset_overall group by household_id, traced_flag) inter
          """
spark.sql(query).show(3000,False)


# In[28]:


query="""select traced_flag, count(*) from 
          (select household_id, traced_flag from control_dataset_overall group by household_id, traced_flag) inter
          group by traced_flag"""
spark.sql(query).show(3000,False)


# In[29]:


query="""
select 
cast(Traced_flag as int) as traced_flag,
cast(household_id as bigint) as household_id,
t_weeks_r12_enterprise_visit,
t_weeks_r12_store_visit,
t_weeks_r12_dotcom_visit,
t_weeks_r12_pickup_visit,
sd_store_order_gap,
sd_dotcom_order_gap,
max_store_order_gap,
max_dotcom_order_gap,
avg_store_order_gap,
avg_dotcom_order_gap,
median_store_order_gap,
median_dotcom_order_gap,
m_sng_90,
m_sng_30,
m_sng_5,
f_sng_90,
f_sng_30,
f_sng_5,
m_sc_90,
m_sc_30,
m_sc_5,
f_sc_90,
f_sc_30,
f_sc_5,
m_wp_90,
m_wp_30,
m_wp_5,
f_wp_90,
f_wp_30,
f_wp_5,
visit_cnt_ios_90,
visit_cnt_ios_30,
visit_cnt_ios_5,
homepage_cnt_ios_90,
homepage_cnt_ios_30,
homepage_cnt_ios_5,
search_cnt_ios_90,
search_cnt_ios_30,
search_cnt_ios_5,
atc_cnt_ios_90,
atc_cnt_ios_30,
atc_cnt_ios_5,
visit_cnt_android_90,
visit_cnt_android_30,
visit_cnt_android_5,
homepage_cnt_android_90,
homepage_cnt_android_30,
homepage_cnt_android_5,
search_cnt_android_90,
search_cnt_android_30,
search_cnt_android_5,
atc_cnt_android_90,
atc_cnt_android_30,
atc_cnt_android_5,
visit_cnt_app_90,
visit_cnt_app_30,
visit_cnt_app_5,
r_sng_90,
r_sng_30,
r_sng_5,
r_sc_90,
r_sc_30,
r_sc_5,
r_wp_90,
r_wp_30,
r_wp_5,
app_order_count_t365,
app_order_count_t90,
app_order_count_t30,
app_order_count_t5,
homepage_cnt_app_90,
homepage_cnt_app_30,
homepage_cnt_app_5,
search_cnt_app_90,
search_cnt_app_30,
search_cnt_app_5,
atc_cnt_app_90,
atc_cnt_app_30,
atc_cnt_app_5,
r_app_90,
r_app_30,
r_app_5,
r_homepage_app_90,
r_homepage_app_30,
r_homepage_app_5,
r_search_app_90,
r_search_app_30,
r_search_app_5,
r_atc_app_90,
r_atc_app_30,
r_atc_app_5,
urbanicity_rural,
urbanicity_urban,
age_group_store_seniors,
age_group_store_genx,
age_group_store_millennials,
age_group_store_genz,
gender_cd_f,
gender_cd_b,
gender_cd_m,
marital_status_married,
marital_status_single,
children_flag_0,
children_flag_1,
urbanicity_semi_urban,
income_group_omni_50k_99k,
income_group_omni_25k_49k,
income_group_omni_lesserthan25k,
income_group_omni_more100k,
age_group_store_baby_boomers,
zip_prop_put_s2s_og,
zip_ps_to_prop_put_s2s_og,
zip_ps_pu_prop_put_s2s_og,

max_ins_repeat_rate_365_store,
min_ins_median_2p_gap_365_store,
max_ins_freq_365_store,

max_repeat_rate_t365_dotcom,
min_median_2p_gap_t365_dotcom,
max_freq_t365_dotcom,

max_repeat_rate_t365_og,
min_median_2p_gap_t365_og,
max_freq_t365_og,

f_store_exc_ct_dt AS F_STORE,
f_dotcom_exc_ct_dt AS F_DOTCOM,
f_og_exc_ct_dt AS F_OG,
m_store_exc_ct_dt AS M_STORE,
m_dotcom_exc_ct_dt AS M_DOTCOM,
m_og_exc_ct_dt AS M_OG,
r_store_exc_ct_dt AS R_STORE,
r_dotcom_exc_ct_dt AS R_DOTCOM,
r_og_exc_ct_dt AS R_OG,

priz_accumulated_wealth,
priz_young_accumulators,
priz_mainstream_families,
priz_sustaining_families,
priz_affluent_empty_nests,
priz_conservative_classics,
priz_cautious_couples,
priz_sustaining_seniors,
priz_midlife_success,
priz_young_achievers,
priz_striving_singles,
og_age_weeks,
app_og_order_count_t365,
app_og_order_count_t90,
app_og_order_count_t30,
app_og_order_count_t5,
f_seo_exc_ct_dt,
f_dotcom_seo_exc_ct_dt,
f_og_seo_exc_ct_dt,
r_seo_exc_ct_dt,
r_dotcom_seo_exc_ct_dt,
r_og_seo_exc_ct_dt,
f_organic_pharmacy_exc_ct_dt,
f_dotcom_organic_pharmacy_exc_ct_dt,
f_og_organic_pharmacy_exc_ct_dt,
r_organic_pharmacy_exc_ct_dt,
r_dotcom_organic_pharmacy_exc_ct_dt,
r_og_organic_pharmacy_exc_ct_dt,
f_sem_exc_ct_dt,
f_dotcom_sem_exc_ct_dt,
f_og_sem_exc_ct_dt,
r_sem_exc_ct_dt,
r_dotcom_sem_exc_ct_dt,
r_og_sem_exc_ct_dt,
f_email_exc_ct_dt,
f_dotcom_email_exc_ct_dt,
f_og_email_exc_ct_dt,
r_email_exc_ct_dt,
r_dotcom_email_exc_ct_dt,
r_og_email_exc_ct_dt,
f_organic_app_exc_ct_dt,
f_dotcom_organic_app_exc_ct_dt,
f_og_organic_app_exc_ct_dt,
r_organic_app_exc_ct_dt,
r_dotcom_organic_app_exc_ct_dt,
r_og_organic_app_exc_ct_dt,
f_social_exc_ct_dt,
f_dotcom_social_exc_ct_dt,
f_og_social_exc_ct_dt,
r_social_exc_ct_dt,
r_dotcom_social_exc_ct_dt,
r_og_social_exc_ct_dt,
f_display_exc_ct_dt,
f_dotcom_display_exc_ct_dt,
f_og_display_exc_ct_dt,
r_display_exc_ct_dt,
r_dotcom_display_exc_ct_dt,
r_og_display_exc_ct_dt,
f_affiliates_cse_exc_ct_dt,
f_dotcom_affiliates_cse_exc_ct_dt,
f_og_affiliates_cse_exc_ct_dt,
r_affiliates_cse_exc_ct_dt,
r_dotcom_affiliates_cse_exc_ct_dt,
r_og_affiliates_cse_exc_ct_dt,
f_organic_exc_ct_dt,
f_dotcom_organic_exc_ct_dt,
f_og_organic_exc_ct_dt,
r_organic_exc_ct_dt,
r_dotcom_organic_exc_ct_dt,
r_og_organic_exc_ct_dt,
--- New metrics
F_DOTCOM_pu_exc_ct_dt,
F_DOTCOM_put_exc_ct_dt,
M_DOTCOM_pu_exc_ct_dt,
M_DOTCOM_put_exc_ct_dt,
R_DOTCOM_pu_exc_ct_dt,
R_DOTCOM_put_exc_ct_dt,
F_mdse_cons_exc_ct_dt,
F_mdse_prod_exc_ct_dt,
F_mdse_meat_exc_ct_dt,
F_mdse_bake_exc_ct_dt,
M_mdse_cons_exc_ct_dt,
M_mdse_prod_exc_ct_dt,
M_mdse_meat_exc_ct_dt,
M_mdse_bake_exc_ct_dt,
R_mdse_con_exc_ct_dt,
R_mdseprod_exc_ct_dt,
R_mdse_meat_exc_ct_dt,
R_mdse_bake_exc_ct_dt,
F_sd_groc_exc_ct_dt,
F_sd_hh_exc_ct_dt,
F_sd_per_exc_ct_dt,
F_sd_hbpc_exc_ct_dt,
M_sd_groc_exc_ct_dt,
M_sd_hh_exc_ct_dt,
M_sd_per_exc_ct_dt,
M_sd_hbpc_exc_ct_dt,
R_sd_groc_exc_ct_dt,
R_sd_hh_exc_ct_dt,
R_sd_per_exc_ct_dt,
R_sd_hbpc_exc_ct_dt,
r_last_resp_dt_dc,
r_last_open_dt_dc,
r_last_click_dt_dc,
ctr_dc,
open_rate_dc,
r_last_unnoticed_dc,

--- Post period metrics
post_order_cnt_og_inc_ct_dt,
post_spend_og_inc_ct_dt,
post_spend_dotcom_inc_ct_dt_90days,
post_order_cnt_dotcom_inc_ct_dt_90days,
post_spend_store_inc_ct_dt_90days,
post_order_cnt_store_inc_ct_dt_90days,
post_spend_og_inc_ct_dt_90days,
post_order_cnt_og_inc_ct_dt_90days,
post_spend_dotcom_inc_ct_dt_30d,
post_order_cnt_dotcom_inc_ct_dt_30d,
post_spend_store_inc_ct_dt_30d,
post_order_cnt_store_inc_ct_dt_30d,
post_spend_og_inc_ct_dt_30d,
post_order_cnt_og_inc_ct_dt_30d,
post_spend_og_inc_ct_dt_15d,
post_order_cnt_og_inc_ct_dt_15d,
activation_flag_90,
activation_flag_30,
activation_flag_15,
datediff(purchase_dt_og, {}) as post_days_to_purchase_og
from control_dataset_overall""".format(wkstdt)
hh_train_df = spark.sql(query)


# In[30]:


query="""select *, sum/cnt as prop
from (select count(*) as cnt, sum({}) as sum from control_dataset_overall)inter""".format(activation_flag)
spark.sql(query).show(3000,False)


# In[31]:


hh_train_df=hh_train_df.withColumn("aov_dotcom",hh_train_df.M_DOTCOM/hh_train_df.F_DOTCOM)
hh_train_df=hh_train_df.withColumn("aov_store",hh_train_df.M_STORE/hh_train_df.F_STORE)
hh_train_df=hh_train_df.withColumn("aov_OG",hh_train_df.M_OG/hh_train_df.F_OG)

hh_train_df=hh_train_df.withColumn("post_aov_og_90",hh_train_df.post_spend_og_inc_ct_dt_90days /hh_train_df.post_order_cnt_og_inc_ct_dt_90days )
hh_train_df=hh_train_df.withColumn("post_aov_og_30",hh_train_df.post_spend_og_inc_ct_dt_30d /hh_train_df.post_order_cnt_og_inc_ct_dt_30d )
hh_train_df=hh_train_df.withColumn("post_aov_og_15",hh_train_df.post_spend_og_inc_ct_dt_15d /hh_train_df.post_order_cnt_og_inc_ct_dt_15d )

hh_train_df=hh_train_df.withColumn("rf_dotcom",hh_train_df.R_DOTCOM*hh_train_df.F_DOTCOM)
hh_train_df=hh_train_df.withColumn("rf_store",hh_train_df.R_STORE*hh_train_df.F_STORE)
hh_train_df=hh_train_df.withColumn("rf_OG",hh_train_df.R_OG*hh_train_df.F_OG)


# In[32]:


hh_train_df = hh_train_df.fillna(0)


# ## Creating Pipeline for features

# In[33]:


q="""desc {}""".format(feature_cols_table)
col_names=spark.sql(q).toPandas()
features_raw=col_names['col_name'].tolist()
features = [str(item) for item in features_raw]


# In[34]:


##3 Variables


# In[35]:


q="""desc {}""".format(colsnow_cols_table)
col_names_colsnow=spark.sql(q).toPandas() 
cols_now_raw=col_names_colsnow['col_name'].tolist()
cols_now = [str(item) for item in cols_now_raw]
    
assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
#labelIndexer = StringIndexer(inputCol='label', outputCol="label")
tmp = [assembler_features]
pipeline = Pipeline(stages=tmp)


# In[36]:


feature_df = cols_now
feature_df = pd.DataFrame()
feature_df['feature']=cols_now
feature_df.reset_index(inplace=True)
feature_df['index'] = feature_df.index


# In[37]:


df_class=hh_train_df.select(features)


# In[38]:


allData = pipeline.fit(df_class).transform(df_class)


# ## Test and train dataset

# In[39]:


trainingData, testData = allData.randomSplit([0.7,0.3], seed=0)

#testData.registerTempTable("test_data_set")#testData.write.saveAsTable(test_dataset, format='orc', mode = 'OVERWRITE')#trainingData.write.saveAsTable(training_dataset, format='orc', mode = 'OVERWRITE')
# In[ ]:


print("Caching Test Data\n")


# In[40]:


testData.cache()
testData.count()


# In[ ]:


print("Test Data Cached\n")
print("Caching Training Data\n")


# In[41]:


trainingData.cache()
trainingData.count()


# In[ ]:


print("Training Data Cached\n")


# ## Building Classification Model

# In[33]:


print("Target Variable: "+activation_flag)


# In[ ]:


print("\nBuilding Classifier\n")


# In[43]:


rf = RF(labelCol=activation_flag, featuresCol='features',numTrees=50, maxDepth=10, minInstancesPerNode=100,featureSubsetStrategy="all")
fit_classifier = rf.fit(trainingData)


# In[44]:


feature_imp_df_class= pd.DataFrame()

feature_imp_df_class['importance'] = list(fit_classifier.featureImportances)
feature_imp_df_class.reset_index(inplace=True)
feature_imp_df_class['index'] = feature_imp_df_class.index

class_feature_imp = pd.merge(feature_imp_df_class,feature_df,on='index',how='inner')
class_feature_imp = class_feature_imp.sort_values('importance', ascending=False)
print("\nFeature Importance\n")
print class_feature_imp


# In[12]:


print("\n Clearing Training Data Cache\n")


# In[45]:


del(trainingData)
gc.collect()


# ### Saving Model

# In[ ]:


print("\nSaving Model\n")


# In[47]:


fit_classifier.write().overwrite().save(model_name)


# In[48]:


fit_classifier = RandomForestClassificationModel.load(model_name)


# ### Scoring on Holdout

# In[ ]:


print("\n Scoring on Hold-out data\n")


# In[49]:


retention= fit_classifier.transform(testData)


# In[ ]:


print("\n Clearing Test Data Cache\n")


# In[50]:


del(testData)
gc.collect()


# In[51]:


firstelement=udf(lambda v:float(v[0]),FloatType())
secondelement=udf(lambda v:float(v[1]),FloatType())
retention_final = retention.select(secondelement('probability').alias('pred_activation'),
'household_id', 'traced_flag', activation_flag
 )


# In[34]:


print("\nSaving Scored Hold-out: "+HH_propensity_final_table_test+"\n")


# In[52]:


retention_final.write.saveAsTable(HH_propensity_final_table_test, format='orc', mode = 'OVERWRITE')


# ## Checking model quality

# In[ ]:


print("Model Quality (Hold-out): \n")


# In[54]:


query="""select count(household_id),mean({}),mean(pred_activation) from {}""".format(activation_flag, HH_propensity_final_table_test)
spark.sql(query).show(3000, False)


# In[ ]:


print("Model Quality - Percentile Check (Hold-out): \n")


# In[ ]:


q="""select *, (avg_diff) / avg_pred as pct_pred_fdiff
from
(
select perc_bucket, avg(pred_activation) as avg_pred, avg({}) as  avg_actual, avg(diff) as avg_diff
from
(
select household_id, traced_flag, pred_activation, {}, (pred_activation - {}) as diff,
ntile(100) over (order by pred_activation desc) as perc_bucket
from {}
)interl
group  by perc_bucket
)inter2""".format(activation_flag, activation_flag, activation_flag, HH_propensity_final_table_test)
df_hql=execute_hql(hql_statement = q, hive_env_options = 'export MAPR_MAPREDUCE_MODE=yarn;')
df_spk=spark.createDataFrame(df_hql)
df_spk.show(3000,False)

# ### ROC metrics

# In[57]:


results = retention.select('probability', activation_flag)


# In[58]:


del(retention_final)
gc.collect()


# In[59]:


sc = spark.sparkContext


# In[60]:


# Prepere score-label set
result_collect = results.collect()
result_list = [(float(i[0][0]), 1.0-float(i[1])) for i in result_collect]
score_and_labels=sc.parallelize(result_list)
metrics=classi_metrics(score_and_labels)
print("The ROC score is: ", metrics.areaUnderROC)


# In[61]:


fpr = dict()
tpr = dict()
roc_auc = dict()

y_test = [i[1] for i in result_list]
y_score = [i[0] for i in result_list]

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

#%matplotlib inline
#plt.figure()
#plt.plot(fpr, tpr, label='ROC curve (area=%0.2f)' %roc_auc)
#plt.plot([0,1], [0,1], 'k--')
#plt.xlim([0.0,1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.legend(loc="lower right")
#plt.show()


# ### Precision and recall

# In[62]:


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)
print("\nAverage Precision Score: ", str(average_precision))

# In[63]:


precision, recall, threshold = precision_recall_curve(y_test, y_score)

#plt.step(recall, precision, color='b', alpha=0.2,
#         where='post')
#plt.fill_between(recall, precision, step='post', alpha=0.2,
#                 color='b')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#          average_precision))


# In[64]:


diff_pre_recall=abs(precision-recall)
diff_pre_recall_list=diff_pre_recall.tolist()
min_ind=diff_pre_recall_list.index(min(diff_pre_recall))
print("Optimal threshold: " , threshold[min_ind] , "\nPrecision at this threshold: ", precision[min_ind], "\nRecall at this threshold: ", recall[min_ind])


# ### Testing on all households

# In[ ]:


print("\nTesting on all households: \n")


# In[65]:


train_hh_df_fetch_query_all_hh="""select
cast(Traced_flag as int) as traced_flag,
cast(household_id as bigint) as household_id,
t_weeks_r12_enterprise_visit,
t_weeks_r12_store_visit,
t_weeks_r12_dotcom_visit,
t_weeks_r12_pickup_visit,
sd_store_order_gap,
sd_dotcom_order_gap,
max_store_order_gap,
max_dotcom_order_gap,
avg_store_order_gap,
avg_dotcom_order_gap,
median_store_order_gap,
median_dotcom_order_gap,
m_sng_90,
m_sng_30,
m_sng_5,
f_sng_90,
f_sng_30,
f_sng_5,
m_sc_90,
m_sc_30,
m_sc_5,
f_sc_90,
f_sc_30,
f_sc_5,
m_wp_90,
m_wp_30,
m_wp_5,
f_wp_90,
f_wp_30,
f_wp_5,
visit_cnt_ios_90,
visit_cnt_ios_30,
visit_cnt_ios_5,
homepage_cnt_ios_90,
homepage_cnt_ios_30,
homepage_cnt_ios_5,
search_cnt_ios_90,
search_cnt_ios_30,
search_cnt_ios_5,
atc_cnt_ios_90,
atc_cnt_ios_30,
atc_cnt_ios_5,
visit_cnt_android_90,
visit_cnt_android_30,
visit_cnt_android_5,
homepage_cnt_android_90,
homepage_cnt_android_30,
homepage_cnt_android_5,
search_cnt_android_90,
search_cnt_android_30,
search_cnt_android_5,
atc_cnt_android_90,
atc_cnt_android_30,
atc_cnt_android_5,
visit_cnt_app_90,
visit_cnt_app_30,
visit_cnt_app_5,
r_sng_90,
r_sng_30,
r_sng_5,
r_sc_90,
r_sc_30,
r_sc_5,
r_wp_90,
r_wp_30,
r_wp_5,
app_order_count_t365,
app_order_count_t90,
app_order_count_t30,
app_order_count_t5,
homepage_cnt_app_90,
homepage_cnt_app_30,
homepage_cnt_app_5,
search_cnt_app_90,
search_cnt_app_30,
search_cnt_app_5,
atc_cnt_app_90,
atc_cnt_app_30,
atc_cnt_app_5,
r_app_90,
r_app_30,
r_app_5,
r_homepage_app_90,
r_homepage_app_30,
r_homepage_app_5,
r_search_app_90,
r_search_app_30,
r_search_app_5,
r_atc_app_90,
r_atc_app_30,
r_atc_app_5,
urbanicity_rural,
urbanicity_urban,
age_group_store_seniors,
age_group_store_genx,
age_group_store_millennials,
age_group_store_genz,
gender_cd_f,
gender_cd_b,
gender_cd_m,
marital_status_married,
marital_status_single,
children_flag_0,
children_flag_1,
urbanicity_semi_urban,
income_group_omni_50k_99k,
income_group_omni_25k_49k,
income_group_omni_lesserthan25k,
income_group_omni_more100k,
age_group_store_baby_boomers,
zip_prop_put_s2s_og,
zip_ps_to_prop_put_s2s_og,
zip_ps_pu_prop_put_s2s_og,
max_ins_repeat_rate_365_store,
min_ins_median_2p_gap_365_store,
max_ins_freq_365_store,

max_repeat_rate_t365_dotcom,
min_median_2p_gap_t365_dotcom,
max_freq_t365_dotcom,

max_repeat_rate_t365_og,
min_median_2p_gap_t365_og,
max_freq_t365_og,

f_store_exc_ct_dt AS 
F_STORE,
f_dotcom_exc_ct_dt AS
F_DOTCOM,
f_og_exc_ct_dt AS
F_OG,
m_store_exc_ct_dt AS
M_STORE,
m_dotcom_exc_ct_dt AS
M_DOTCOM,
m_og_exc_ct_dt AS
M_OG,
r_store_exc_ct_dt AS
R_STORE,
r_dotcom_exc_ct_dt AS 
R_DOTCOM,
r_og_exc_ct_dt AS 
R_OG,

priz_accumulated_wealth,
priz_young_accumulators,
priz_mainstream_families,
priz_sustaining_families,
priz_affluent_empty_nests,
priz_conservative_classics,
priz_cautious_couples,
priz_sustaining_seniors,
priz_midlife_success,
priz_young_achievers,
priz_striving_singles,
og_age_weeks,
app_og_order_count_t365,
app_og_order_count_t90,
app_og_order_count_t30,
app_og_order_count_t5,
f_seo_exc_ct_dt,
f_dotcom_seo_exc_ct_dt,
f_og_seo_exc_ct_dt,
r_seo_exc_ct_dt,
r_dotcom_seo_exc_ct_dt,
r_og_seo_exc_ct_dt,
f_organic_pharmacy_exc_ct_dt,
f_dotcom_organic_pharmacy_exc_ct_dt,
f_og_organic_pharmacy_exc_ct_dt,
r_organic_pharmacy_exc_ct_dt,
r_dotcom_organic_pharmacy_exc_ct_dt,
r_og_organic_pharmacy_exc_ct_dt,
f_sem_exc_ct_dt,
f_dotcom_sem_exc_ct_dt,
f_og_sem_exc_ct_dt,
r_sem_exc_ct_dt,
r_dotcom_sem_exc_ct_dt,
r_og_sem_exc_ct_dt,
f_email_exc_ct_dt,
f_dotcom_email_exc_ct_dt,
f_og_email_exc_ct_dt,
r_email_exc_ct_dt,
r_dotcom_email_exc_ct_dt,
r_og_email_exc_ct_dt,
f_organic_app_exc_ct_dt,
f_dotcom_organic_app_exc_ct_dt,
f_og_organic_app_exc_ct_dt,
r_organic_app_exc_ct_dt,
r_dotcom_organic_app_exc_ct_dt,
r_og_organic_app_exc_ct_dt,
f_social_exc_ct_dt,
f_dotcom_social_exc_ct_dt,
f_og_social_exc_ct_dt,
r_social_exc_ct_dt,
r_dotcom_social_exc_ct_dt,
r_og_social_exc_ct_dt,
f_display_exc_ct_dt,
f_dotcom_display_exc_ct_dt,
f_og_display_exc_ct_dt,
r_display_exc_ct_dt,
r_dotcom_display_exc_ct_dt,
r_og_display_exc_ct_dt,
f_affiliates_cse_exc_ct_dt,
f_dotcom_affiliates_cse_exc_ct_dt,
f_og_affiliates_cse_exc_ct_dt,
r_affiliates_cse_exc_ct_dt,
r_dotcom_affiliates_cse_exc_ct_dt,
r_og_affiliates_cse_exc_ct_dt,
f_organic_exc_ct_dt,
f_dotcom_organic_exc_ct_dt,
f_og_organic_exc_ct_dt,
r_organic_exc_ct_dt,
r_dotcom_organic_exc_ct_dt,
r_og_organic_exc_ct_dt,
--- New metrics
F_DOTCOM_pu_exc_ct_dt,
F_DOTCOM_put_exc_ct_dt,
M_DOTCOM_pu_exc_ct_dt,
M_DOTCOM_put_exc_ct_dt,
R_DOTCOM_pu_exc_ct_dt,
R_DOTCOM_put_exc_ct_dt,
F_mdse_cons_exc_ct_dt,
F_mdse_prod_exc_ct_dt,
F_mdse_meat_exc_ct_dt,
F_mdse_bake_exc_ct_dt,
M_mdse_cons_exc_ct_dt,
M_mdse_prod_exc_ct_dt,
M_mdse_meat_exc_ct_dt,
M_mdse_bake_exc_ct_dt,
R_mdse_con_exc_ct_dt,
R_mdseprod_exc_ct_dt,
R_mdse_meat_exc_ct_dt,
R_mdse_bake_exc_ct_dt,
F_sd_groc_exc_ct_dt,
F_sd_hh_exc_ct_dt,
F_sd_per_exc_ct_dt,
F_sd_hbpc_exc_ct_dt,
M_sd_groc_exc_ct_dt,
M_sd_hh_exc_ct_dt,
M_sd_per_exc_ct_dt,
M_sd_hbpc_exc_ct_dt,
R_sd_groc_exc_ct_dt,
R_sd_hh_exc_ct_dt,
R_sd_per_exc_ct_dt,
R_sd_hbpc_exc_ct_dt,

--email metrics
r_last_resp_dt_dc,
r_last_open_dt_dc,
r_last_click_dt_dc,
ctr_dc,
open_rate_dc,
r_last_unnoticed_dc,

--- Post period metrics
post_order_cnt_og_inc_ct_dt,
post_spend_og_inc_ct_dt,
post_spend_dotcom_inc_ct_dt_90days,
post_order_cnt_dotcom_inc_ct_dt_90days,
post_spend_store_inc_ct_dt_90days,
post_order_cnt_store_inc_ct_dt_90days,
post_spend_og_inc_ct_dt_90days,
post_order_cnt_og_inc_ct_dt_90days,
post_spend_dotcom_inc_ct_dt_30d,
post_order_cnt_dotcom_inc_ct_dt_30d,
post_spend_store_inc_ct_dt_30d,
post_order_cnt_store_inc_ct_dt_30d,
post_spend_og_inc_ct_dt_30d,
post_order_cnt_og_inc_ct_dt_30d,
post_spend_og_inc_ct_dt_15d,
post_order_cnt_og_inc_ct_dt_15d,
if(post_order_cnt_dotcom_inc_ct_dt_15d >0,1,0) as activation_flag_15,
if(post_order_cnt_dotcom_inc_ct_dt_90days >0,1,0) as activation_flag_90,
if(post_order_cnt_dotcom_inc_ct_dt_30d >0,1,0) as activation_flag_30,
datediff(purchase_dt_og, {}) as post_days_to_purchase_og
from {}
where 
{}
""".format(wkstdt, dataset_name, pre_period_conditions)


# In[66]:


train_hh_df_all_hh=spark.sql(train_hh_df_fetch_query_all_hh)


# In[67]:


train_hh_df_all_hh=train_hh_df_all_hh.withColumn("aov_dotcom",train_hh_df_all_hh.M_DOTCOM/train_hh_df_all_hh.F_DOTCOM)
train_hh_df_all_hh=train_hh_df_all_hh.withColumn("aov_store",train_hh_df_all_hh.M_STORE/train_hh_df_all_hh.F_STORE)
train_hh_df_all_hh=train_hh_df_all_hh.withColumn("aov_OG",train_hh_df_all_hh.M_OG/train_hh_df_all_hh.F_OG)

train_hh_df_all_hh=train_hh_df_all_hh.withColumn("post_aov_og_90",train_hh_df_all_hh.post_spend_og_inc_ct_dt_90days /train_hh_df_all_hh.post_order_cnt_og_inc_ct_dt_90days )
train_hh_df_all_hh=train_hh_df_all_hh.withColumn("post_aov_og_30",train_hh_df_all_hh.post_spend_og_inc_ct_dt_30d /train_hh_df_all_hh.post_order_cnt_og_inc_ct_dt_30d )
train_hh_df_all_hh=train_hh_df_all_hh.withColumn("post_aov_og_15",train_hh_df_all_hh.post_spend_og_inc_ct_dt_15d /train_hh_df_all_hh.post_order_cnt_og_inc_ct_dt_15d )

train_hh_df_all_hh=train_hh_df_all_hh.withColumn("rf_dotcom",train_hh_df_all_hh.R_DOTCOM*train_hh_df_all_hh.F_DOTCOM)
train_hh_df_all_hh=train_hh_df_all_hh.withColumn("rf_store",train_hh_df_all_hh.R_STORE*train_hh_df_all_hh.F_STORE)
train_hh_df_all_hh=train_hh_df_all_hh.withColumn("rf_OG",train_hh_df_all_hh.R_OG*train_hh_df_all_hh.F_OG)


# In[68]:


df_class=train_hh_df_all_hh.select(features)


# In[ ]:


print("\n Clearing temp variable: train_hh_df_all_hh\n")


# In[69]:


del(train_hh_df_all_hh)
gc.collect()


# In[70]:


df_class = df_class.fillna(0)


# In[71]:


allData = pipeline.fit(df_class).transform(df_class)


# In[ ]:


print("\n Clearing temp variable: df_class\n")


# In[72]:


del(df_class)
gc.collect()


# In[73]:


fit_classifier = RandomForestClassificationModel.load(model_name)


# In[74]:


retention_all_hh= fit_classifier.transform(allData)


# In[ ]:


print("\n Clearing temp variable: allData\n")


# In[75]:


del(allData)
gc.collect()


# In[76]:


firstelement=udf(lambda v:float(v[0]),FloatType())
secondelement=udf(lambda v:float(v[1]),FloatType())
retention_final_all_hh = retention_all_hh.select(secondelement('probability').alias('pred_activation'),
'household_id', 'traced_flag', activation_flag)


# In[ ]:


print("\n Clearing temp variable: retention_all_hh\n")


# In[77]:


del(retention_all_hh)
gc.collect()


# In[78]:


retention_final_all_hh.registerTempTable('retention_final_results_all_hh')


# In[79]:


print("\nSaving Scored All Household data: "+HH_propensity_final_table_all_hh+"\n")


# In[ ]:


query_del="""drop table if exists {}""".format(HH_propensity_final_table_all_hh)
spark.sql(query_del)


# In[80]:


query_save="""create table {} stored as orc as 
select * from {}""".format(HH_propensity_final_table_all_hh, 'retention_final_results_all_hh')
spark.sql(query_save)


# In[ ]:


print("\n Clearing temp variable: retention_final_all_hh\n")


# In[81]:


del(retention_final_all_hh)
gc.collect()


# In[82]:


#retention_final_all_hh.write.saveAsTable(HH_propensity_final_table_all_hh, format='orc',mode = 'OVERWRITE')


# In[ ]:


print("Model Quality (All Household): \n")


# In[83]:


query="""select count(household_id),mean({}),mean(pred_activation) from {} """.format(activation_flag, HH_propensity_final_table_all_hh)
spark.sql(query).show(3000,False)


# In[ ]:


print("Model Quality - Percentile Check (All Household): \n")


# In[ ]:


q="""select *, (avg_diff) / avg_pred as pct_pred_fdiff
from
(
select perc_bucket, avg(pred_activation) as avg_pred, avg({}) as  avg_actual, avg(diff) as avg_diff
from
(
select household_id, traced_flag, pred_activation, {}, (pred_activation - {}) as diff,
ntile(100) over (order by pred_activation desc) as perc_bucket
from {}
)interl
group  by perc_bucket
)inter2""".format(activation_flag, activation_flag, activation_flag, HH_propensity_final_table_all_hh)
df_hql=execute_hql(hql_statement = q, hive_env_options = 'export MAPR_MAPREDUCE_MODE=yarn;')
df_spk=spark.createDataFrame(df_hql)
df_spk.show(3000,False)


# In[ ]:


spark.stop()

