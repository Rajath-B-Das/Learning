
# coding: utf-8

# ### Setting up environment and loading required packages

# In[1]:


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
from pyspark.ml.regression import RandomForestRegressor as RF1
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
#from pyspark.ml.regression import RandomForestRegressor as RF1
import csv, threading
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType
#%matplotlib inline
from pyspark.ml.feature import IndexToString

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
    ,["spark.executor.heartbeatInterval", "7200s"]
    ,["spark.network.timeout", "7200s"]
    ,["spark.shuffle.io.connectionTimeout","7200s"]
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

SPARK_CONF = SparkConf().setAppName("prop_mod_2x0_dotcom").setAll(confList)

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
    hql_stmt = hive_env_options + " hive -e " + "\"""" + cli_opts + " " + hql_statement + ';'  + "\"""" 
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


# In[2]:


#importing python function to run hive queries
sys.path.append('/mnt/projects/analytics_starter/lib/')
from python_hive_functions import *

# execute_hql() used below comes from here


# ## Please enter the following parameters

# In[3]:


from datetime import datetime
import unicodedata


# In[4]:


import getpass
#Getting the schema name
schema_name = getpass.getuser()
schema_name


# In[5]:


import datetime
import time
from datetime import date, timedelta, datetime
import gc


# In[7]:


max_week_holder="mjoshi2.prop_mod_max_week_holder"
calendar="us_core_dim.calendar_dim"

q="""select min(calendar_date), min(wm_ly_comp_date), date_add(min(calendar_date),7), date_add(min(wm_ly_comp_date),7)
from 
{} cal
inner join
(select max(week_id) as week_id from {}) max_week_holder
on cal.wm_year_wk_nbr=max_week_holder.week_id + 1
""".format(calendar, max_week_holder)
df=spark.sql(q).toPandas()
WEEK_ST_DT_D2P= str(df.iloc[0,3])
SCORING_WEEK_ST_DT_D2P= str(df.iloc[0,2])
WEEK_ST_DT= str(df.iloc[0,1])
SCORING_WEEK_ST_DT = str(df.iloc[0,0])
print(WEEK_ST_DT)
print(SCORING_WEEK_ST_DT)


#In[73]:


# Snapshot date
week_start_date="\'{}\'".format(WEEK_ST_DT)
week_start_date_d2p = "\'{}\'".format(WEEK_ST_DT_D2P)

# In[8]:


customer_base='2x0_dotcom'

# Model Building Snapshot
dataset_schema='mjoshi2'

# Preperiod condition to subset customer base
pre_period_conditions="""((traced_flag=1 and f_store_exc_ct_dt>=2 and 
coalesce(f_og_exc_ct_dt,0)+coalesce(f_dotcom_exc_ct_dt,0)=0) and post_order_cnt_dotcom_inc_ct_dt_90days>0)"""

# Regression targets[orders, aov]
regression_targets=['post_order_cnt_dotcom_inc_ct_dt_90days', 'post_aov_90', 'post_days_to_purchase']


# In[9]:


year = datetime.strptime(WEEK_ST_DT, '%Y-%m-%d' ).year
month = datetime.strptime(WEEK_ST_DT, '%Y-%m-%d' ).month
day = datetime.strptime(WEEK_ST_DT, '%Y-%m-%d' ).day

if month<10:
    month_str='0'+str(month)
else:
    month_str=str(month)
if day<10:
    day='0'+str(day)
else:
    day=str(day)

week_start_date_suffix=str(year)+'_'+month_str+'_'+str(day)


# In[10]:


dataset_name_prefix = '.mu_rfm_metrics_w_pu_bb_flags_added_w_mdse_ogsd_sq'

#Table name with which training dataset should be stored
training_dataset_prefix=schema_name + '.orders_aov_d2f_mod_2x0_dotcom_training_data_90d'

#Table name with which test dataset should be stored
test_dataset_prefix=schema_name + '.orders_aov_d2f_mod_2x0_dotcom_test_data_90d'

# Enter the desired name for the models [orders, aov]
regression_models=['orders_2x0_dotcom_reg_model_90days', 'aov_2x0_dotcom_reg_model_90days', 'd2p_2x0_dotcom_reg_model_90days']

# Test Data prediction results 
reg_test_data_prefix=schema_name +'.orders_aov_d2f_mod_regression_2x0_dotcom_test_data_results_90days'

# All HH prediction results - orders
reg_all_hh_prefix=schema_name +'.orders_aov_d2f_mod_regression_2x0_dotcom_all_hh_data_results_90days'


# In[11]:


# Snapshot date
week_start_date="\'{}\'".format(WEEK_ST_DT)

# Desired dataset (user.table_name)
dataset_name = dataset_schema + dataset_name_prefix + '_' + week_start_date_suffix + '_sbctqnt_w_coded_demogs'

# Table name with which stratified sample should be saved (user.table_name)
strat_table = schema_name +  dataset_name_prefix + '_'  + week_start_date_suffix + '_strat'+ '_'  + customer_base

#Table name with which training dataset should be stored
training_data_table= training_dataset_prefix + '_' + week_start_date_suffix

#Table name with which test dataset should be stored
test_data_table= test_dataset_prefix + '_' +week_start_date_suffix

# Table to store final scores and features (test data)
reg_test_data_results=reg_test_data_prefix + '_'  +  week_start_date_suffix

# Table to store final propensity scores and features (all HH)
reg_all_hh_results= reg_all_hh_prefix + '_'  +  week_start_date_suffix

# Enter the desired name for the model
regression_model_names = ['{model}_{week_start_date_suffix}'.format(**locals()) for model in regression_models]


# In[12]:


#week_start_date="\'{}\'".format(WEEK_ST_DT)
week_start_date


# In[13]:


dataset_name


# ### Strat table created while doing dotcom propensity model

# In[13]:


#strat_table = 'vn0d3ov.mu_rfm_metrics_w_pu_bb_flags_added_w_mdse_ogsd_2x0_dotcom_2017_04_01_strat'


# In[17]:


drop="""drop table if exists {}""".format(strat_table)
spark.sql(drop)


# ### Creating stratification table

# In[25]:


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


# In[26]:


query="""select count(*) from {}""".format(strat_table)
spark.sql(query).show(3000,False)


# In[28]:


query="""select traced_flag, count(*)
          from
         (
         select * from {}
         where {}
          ) inter
          group by traced_flag""".format(dataset_name,pre_period_conditions)
spark.sql(query).show(3000,False)


# ## Sample for creating train and test sets

# In[14]:


## Get stratified sample in dataframe
fetch_strat_sample_query = "select * from {} ".format(strat_table)


# In[15]:


## From stratified sample get training sample
hh_train_for_sample = spark.sql(fetch_strat_sample_query)


# In[16]:


count=hh_train_for_sample.count()
count


# In[17]:


sample_fraction = 2000000.0/count


# In[18]:


if count>2500000:
    hh_train_sample=hh_train_for_sample.sample(False, sample_fraction,0)
else:
    hh_train_sample=hh_train_for_sample
hh_train_sample.count()


# In[19]:


del(hh_train_for_sample)
gc.collect()


# In[20]:


hh_train_sample.registerTempTable("control_dataset_overall")


# In[67]:


Query="""
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

min_median_2p_gap_t365_dotcom,
max_repeat_rate_t365_dotcom,
max_freq_t365_dotcom,
min_median_2p_gap_t365_og,
max_repeat_rate_t365_og,
max_freq_t365_og,
max_ins_repeat_rate_365_store,
min_ins_median_2p_gap_365_store,
max_ins_freq_365_store,


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

--- email metrics
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
post_spend_dotcom_inc_ct_dt_15d,
post_order_cnt_dotcom_inc_ct_dt_15d,


activation_flag_90,
activation_flag_30,
activation_flag_15,
datediff(purchase_dt_dotcom, {}) as post_days_to_purchase
from control_dataset_overall""".format(week_start_date_d2p)
hh_train_df = spark.sql(Query)


# In[68]:


hh_train_df=hh_train_df.withColumn("aov_dotcom",hh_train_df.M_DOTCOM/hh_train_df.F_DOTCOM)
hh_train_df=hh_train_df.withColumn("aov_store",hh_train_df.M_STORE/hh_train_df.F_STORE)
hh_train_df=hh_train_df.withColumn("aov_OG",hh_train_df.M_OG/hh_train_df.F_OG)

hh_train_df=hh_train_df.withColumn("post_aov_90",hh_train_df.post_spend_dotcom_inc_ct_dt_90days /hh_train_df.post_order_cnt_dotcom_inc_ct_dt_90days )
hh_train_df=hh_train_df.withColumn("post_aov_30",hh_train_df.post_spend_dotcom_inc_ct_dt_30d /hh_train_df.post_order_cnt_dotcom_inc_ct_dt_30d )
hh_train_df=hh_train_df.withColumn("post_aov_15",hh_train_df.post_spend_dotcom_inc_ct_dt_15d /hh_train_df.post_order_cnt_dotcom_inc_ct_dt_15d )

hh_train_df=hh_train_df.withColumn("rf_dotcom",hh_train_df.R_DOTCOM*hh_train_df.F_DOTCOM)
hh_train_df=hh_train_df.withColumn("rf_store",hh_train_df.R_STORE*hh_train_df.F_STORE)
hh_train_df=hh_train_df.withColumn("rf_OG",hh_train_df.R_OG*hh_train_df.F_OG)


# In[69]:


hh_train_df = hh_train_df.fillna(0)


# ## Creating Pipeline for features

# In[70]:


feature_cols_table='mjoshi2.prop_and_exp_value_features_as_cols_w_d2p_dotcom_models'
    
colsnow_cols_table='mjoshi2.prop_and_exp_value_colsnow_as_cols'

q="""desc {}""".format(feature_cols_table)
col_names=spark.sql(q).toPandas()
features_raw=col_names['col_name'].tolist()
features = [str(item) for item in features_raw]
    
q="""desc {}""".format(colsnow_cols_table)
col_names_colsnow=spark.sql(q).toPandas() 
cols_now_raw=col_names_colsnow['col_name'].tolist()
cols_now = [str(item) for item in cols_now_raw]


# In[71]:


assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
#labelIndexer = StringIndexer(inputCol='label', outputCol="label")
tmp = [assembler_features]
pipeline = Pipeline(stages=tmp)


# In[72]:


feature_df = cols_now
feature_df = pd.DataFrame()
feature_df['feature']=cols_now
feature_df.reset_index(inplace=True)
feature_df['index'] = feature_df.index


# In[73]:


df_class=hh_train_df.select(features)


# In[74]:


del(hh_train_df)
gc.collect()


# In[75]:


allData = pipeline.fit(df_class).transform(df_class)


# In[76]:


del(df_class)
gc.collect()


# ## Test and train dataset

# In[77]:


trainingData, testData = allData.randomSplit([0.7,0.3], seed=0)


# In[79]:


del(allData)
gc.collect()


# ## Regression Model - Orders

# #### Model Building

# In[80]:


regression_targets[0]


# In[81]:


rf_order = RF1(labelCol=regression_targets[0], featuresCol='features',numTrees=50, maxDepth=10, minInstancesPerNode=100,featureSubsetStrategy="all")
fit_reg_order = rf_order.fit(trainingData)


# In[82]:


feature_imp_df_order= pd.DataFrame()

feature_imp_df_order['importance'] = list(fit_reg_order.featureImportances)
feature_imp_df_order.reset_index(inplace=True)
feature_imp_df_order['index'] = feature_imp_df_order.index

order_feature_imp = pd.merge(feature_imp_df_order,feature_df,on='index',how='inner')
print (order_feature_imp.sort_values('importance', ascending=False))


# #### Saving Model

# In[83]:


regression_model_names[0]


# In[84]:


fit_reg_order.write().overwrite().save(regression_model_names[0])


# #### Scoring on Holdout

# In[85]:


transformed_orders = fit_reg_order.transform(testData)


# ##### Saving table and registering temp table with prediction results

# In[86]:


reg_test_data_results


# In[87]:


transformed_orders.write.saveAsTable(reg_test_data_results, format='ORC', mode='OVERWRITE')


# In[88]:


transformed_orders.registerTempTable('transformed_orders_results')


# #### Checking Model Quality

# In[90]:


query="""
with base as
(
select traced_flag,
{},
prediction,
abs(coalesce({},0) - coalesce(prediction , 0 )) as diff
from transformed_orders_results
)
select traced_flag,
mean({}),
mean(prediction),
mean(diff) as mav 
from base
group by traced_flag
""".format(regression_targets[0], regression_targets[0], regression_targets[0])

temp=spark.sql(query).toPandas()

temp


# In[91]:


results = transformed_orders.select('prediction', regression_targets[0])
sc = spark.sparkContext

# Prepere value-prediction set
from pyspark.mllib.evaluation import RegressionMetrics as regress_metrics
result_collect = results.collect()
result_list = [(float(i[0]), float(i[1])) for i in result_collect]
value_and_preds=sc.parallelize(result_list)
metrics=regress_metrics(value_and_preds)


# In[92]:


# Squared Error
print("MSE = %s" % metrics.meanSquaredError)
print("RMSE = %s" % metrics.rootMeanSquaredError)

# R-squared
print("R-squared = %s" % metrics.r2)

# Mean absolute error
print("MAE = %s" % metrics.meanAbsoluteError)

# Explained variance
print("Explained variance = %s" % metrics.explainedVariance)


# ## Regression Model - AOV

# In[93]:


regression_targets[1]


# In[94]:


rf_aov = RF1(labelCol=regression_targets[1], featuresCol='features',numTrees=50, maxDepth=10, minInstancesPerNode=100,featureSubsetStrategy="all")
fit_reg_aov = rf_aov.fit(trainingData)


# In[95]:


feature_imp_df_aov= pd.DataFrame()

feature_imp_df_aov['importance'] = list(fit_reg_aov.featureImportances)
feature_imp_df_aov.reset_index(inplace=True)
feature_imp_df_aov['index'] = feature_imp_df_aov.index

aov_feature_imp = pd.merge(feature_imp_df_aov,feature_df,on='index',how='inner')
print (aov_feature_imp.sort_values('importance', ascending=False))


# In[96]:


fit_reg_aov.write().overwrite().save(regression_model_names[1])


# In[97]:


regression_model_names[1]


# #### Scoring on holdout

# In[98]:


fit_reg_aov= RandomForestRegressionModel.load(regression_model_names[1])


# In[99]:


transformed_aov = fit_reg_aov.transform(testData)


# In[100]:


transformed_aov.registerTempTable('transformed_aov_results')


# ##### Saving table and registering temp table with prediction results

# In[101]:


joined_orders_aov_query="""select
order.*, 
aov.prediction as aov_prediction
from 
transformed_orders_results order
left join 
transformed_aov_results aov
on order.household_id=aov.household_id
and order.traced_flag=aov.traced_flag
"""
joined_orders_aov=spark.sql(joined_orders_aov_query)
joined_orders_aov.registerTempTable('joined_orders_aov_temp')


# In[102]:


joined_orders_aov.write.saveAsTable(reg_test_data_results, format='ORC', mode='OVERWRITE')


# #### Checking model quality

# In[104]:


query="""
with base as
(
select traced_flag,
{},
aov_prediction,
abs(coalesce({},0) - coalesce(aov_prediction , 0 )) as diff
from joined_orders_aov_temp
)
select traced_flag,
mean({}),
mean(aov_prediction),
mean(diff) as mav 
from base
group by traced_flag
""".format(regression_targets[1], regression_targets[1], regression_targets[1])

temp=spark.sql(query).toPandas()

temp


# In[105]:


results = transformed_aov.select('prediction', regression_targets[1])
sc = spark.sparkContext

# Prepere value-prediction set
from pyspark.mllib.evaluation import RegressionMetrics as regress_metrics
result_collect = results.collect()
result_list = [(float(i[0]), float(i[1])) for i in result_collect]
value_and_preds=sc.parallelize(result_list)
metrics=regress_metrics(value_and_preds)


# In[106]:


# Squared Error
print("MSE = %s" % metrics.meanSquaredError)
print("RMSE = %s" % metrics.rootMeanSquaredError)

# R-squared
print("R-squared = %s" % metrics.r2)

# Mean absolute error
print("MAE = %s" % metrics.meanAbsoluteError)

# Explained variance
print("Explained variance = %s" % metrics.explainedVariance)


# ### Regression Model - Days to purchase

# In[107]:


regression_targets[2]


# In[108]:


rf_d2p = RF1(labelCol=regression_targets[2], featuresCol='features',numTrees=50, maxDepth=10, minInstancesPerNode=100,featureSubsetStrategy="all")
fit_reg_d2p = rf_d2p.fit(trainingData)


# In[109]:


feature_imp_df_d2p= pd.DataFrame()

feature_imp_df_d2p['importance'] = list(fit_reg_d2p.featureImportances)
feature_imp_df_d2p.reset_index(inplace=True)
feature_imp_df_d2p['index'] = feature_imp_df_d2p.index

d2p_feature_imp = pd.merge(feature_imp_df_d2p,feature_df,on='index',how='inner')
print (d2p_feature_imp.sort_values('importance', ascending=False))


# In[110]:


regression_model_names[2]


# In[111]:


fit_reg_d2p.write().overwrite().save(regression_model_names[2])


# #### Scoring on Holdout

# In[112]:


fit_reg_d2p= RandomForestRegressionModel.load(regression_model_names[2])


# In[113]:


transformed_d2p = fit_reg_d2p.transform(testData)
transformed_d2p.registerTempTable('transformed_d2p_results')


# ##### Saving table with prediction results

# In[114]:


joined_orders_aov_d2p_query="""select
order_aov.*, 
d2p.prediction as d2p_prediction
from 
joined_orders_aov_temp order_aov
left join 
transformed_d2p_results d2p
on order_aov.household_id=d2p.household_id
and order_aov.traced_flag=d2p.traced_flag
"""
joined_orders_aov_d2p=spark.sql(joined_orders_aov_d2p_query)
joined_orders_aov_d2p.registerTempTable('joined_orders_aov_temp_d2p')


# In[115]:


joined_orders_aov_d2p.write.saveAsTable(reg_test_data_results, format='ORC', mode='OVERWRITE')


# #### Checking model quality

# In[117]:


query="""
with base as
(
select traced_flag,
{},
d2p_prediction,
abs(coalesce({},0) - coalesce(d2p_prediction , 0 )) as diff
from joined_orders_aov_temp_d2p
)
select traced_flag,
mean({}),
mean(d2p_prediction),
mean(diff) as mav 
from base
group by traced_flag
""".format(regression_targets[2], regression_targets[2], regression_targets[2])

temp=spark.sql(query).toPandas()

temp


# In[118]:


results = transformed_d2p.select('prediction', regression_targets[2])
sc = spark.sparkContext

# Prepere value-prediction set
from pyspark.mllib.evaluation import RegressionMetrics as regress_metrics
result_collect = results.collect()
result_list = [(float(i[0]), float(i[1])) for i in result_collect]
value_and_preds=sc.parallelize(result_list)
metrics=regress_metrics(value_and_preds)


# In[119]:


# Squared Error
print("MSE = %s" % metrics.meanSquaredError)
print("RMSE = %s" % metrics.rootMeanSquaredError)

# R-squared
print("R-squared = %s" % metrics.r2)

# Mean absolute error
print("MAE = %s" % metrics.meanAbsoluteError)

# Explained variance
print("Explained variance = %s" % metrics.explainedVariance)


# In[120]:


reg_test_data_results


# In[121]:


del(testData)
del(trainingData)
del(transformed_orders)
del(transformed_aov)
del(transformed_d2p)
gc.collect()


# In[122]:


spark.catalog.clearCache()

spark.stop()
# ### Testing on all households