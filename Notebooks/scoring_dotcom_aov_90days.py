
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
#get_ipython().run_line_magic('matplotlib', 'inline')
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

SPARK_CONF = SparkConf().setAppName("prop_mod_dotcom_90_aov_scored").setAll(confList)

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


# In[3]:


from datetime import datetime
import unicodedata


# In[4]:


import getpass
#Getting the schema name
schema_name = getpass.getuser()
schema_name


# #### Setting up the parameters

# In[5]:


import datetime
import time
from datetime import date, timedelta, datetime
import gc


# In[6]:


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


# In[7]:


week_start_date="\'{}\'".format(WEEK_ST_DT)
scoring_week_start_date="\'{}\'".format(SCORING_WEEK_ST_DT)


# In[55]:


customer_base='dotcom_90days'

# Model Scoring Snapshot
scoring_dataset_schema='mjoshi2'
model_schema='/user/vn05uy5/'

# Scoring Preperiod condition to subset customer base
scoring_pre_period_conditions_2x0="""(traced_flag=1 and f_store_exc_ct_dt>=2 and (coalesce(f_og_exc_ct_dt,0)+coalesce(f_dotcom_exc_ct_dt,0))=0)
"""

scoring_pre_period_conditions_2x1_hh="""(traced_flag=1 and f_store_exc_ct_dt>=2 and (coalesce(f_og_exc_ct_dt,0)+coalesce(f_dotcom_exc_ct_dt,0))>0)"""

scoring_pre_period_conditions_2x1_ugc="""(traced_flag=0 and (coalesce(f_og_exc_ct_dt,0)+coalesce(f_dotcom_exc_ct_dt,0))>0)"""


# In[9]:


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
scoring_week_start_date_suffix=str(year)+'_'+month_str+'_'+str(day_str)


# In[10]:


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
week_start_date_suffix=str(year)+'_'+month_str+'_'+str(day_str)


# In[11]:


# Enter the desired name for the models [orders, aov]
regression_models_2x0=['orders_2x0_dotcom_reg_model_90days', 'aov_2x0_dotcom_reg_model_90days', 'd2p_2x0_dotcom_reg_model_90days']

regression_models_2x1_hh=['orders_2x1_hh_dotcom_reg_model_90days', 'aov_2x1_hh_dotcom_reg_model_90days', 'd2p_2x1_hh_dotcom_reg_model_90days']

regression_models_2x1_ugc=['orders_2x1_ugc_dotcom_reg_model_90days', 'aov_2x1_ugc_dotcom_reg_model_90days', 'd2p_2x1_ugc_dotcom_reg_model_90days']

# Intermediate Scored Datasets 
intermediate_orders_scored_dataset_prefix_2x0=schema_name + '.prop_mod_2x0_dotcom_90_days_orders_inter'
intermediate_aov_scored_dataset_prefix_2x0=schema_name + '.prop_mod_2x0_dotcom_90_days_aov_inter'
intermediate_d2p_scored_dataset_prefix_2x0=schema_name + '.prop_mod_2x0_dotcom_90_days_d2p_inter'

intermediate_orders_scored_dataset_prefix_2x1_hh=schema_name + '.prop_mod_2x1_hh_dotcom_90_days_orders_inter'
intermediate_aov_scored_dataset_prefix_2x1_hh=schema_name + '.prop_mod_2x1_hh_dotcom_90_days_aov_inter'
intermediate_d2p_scored_dataset_prefix_2x1_hh=schema_name + '.prop_mod_2x1_hh_dotcom_90_days_d2p_inter'

intermediate_orders_scored_dataset_prefix_2x1_ugc=schema_name + '.prop_mod_2x1_ugc_dotcom_90_days_orders_inter'
intermediate_aov_scored_dataset_prefix_2x1_ugc=schema_name + '.prop_mod_2x1_ugc_dotcom_90_days_aov_inter'
intermediate_d2p_scored_dataset_prefix_2x1_ugc=schema_name + '.prop_mod_2x1_ugc_dotcom_90_days_d2p_inter'

# Scoring dataset name
scoring_dataset_name_prefix=scoring_dataset_schema+'.mu_rfm_metrics_w_pu_bb_flags_added_w_mdse_ogsd_sq'

#Scored dataset name prefix
scored_dataset_name_prefix_2x0 = schema_name + '.prop_mod_scored_orders_aov_d2p_2x0_dotcom_90d'

scored_dataset_name_prefix_2x1_hh = schema_name + '.prop_mod_scored_orders_aov_d2p_2x1_hh_dotcom_90d'

scored_dataset_name_prefix_2x1_ugc = schema_name + '.prop_mod_scored_orders_aov_d2p_2x1_ugc_dotcom_90d'


# In[12]:


regression_model_names_2x0 = ['{model_schema}{model}_{week_start_date_suffix}'.format(**locals()) for model in regression_models_2x0]

regression_model_names_2x1_hh = ['{model_schema}{model}_{week_start_date_suffix}'.format(**locals()) for model in regression_models_2x1_hh]

regression_model_names_2x1_ugc = ['{model_schema}{model}_{week_start_date_suffix}'.format(**locals()) for model in regression_models_2x1_ugc]


# In[13]:


# Intermediate Scored Datasets 
intermediate_orders_scored_dataset_name_2x0=intermediate_orders_scored_dataset_prefix_2x0+'_'+scoring_week_start_date_suffix
intermediate_aov_scored_dataset_name_2x0=intermediate_aov_scored_dataset_prefix_2x0+'_'+scoring_week_start_date_suffix
intermediate_d2p_scored_dataset_name_2x0=intermediate_d2p_scored_dataset_prefix_2x0+'_'+scoring_week_start_date_suffix

intermediate_orders_scored_dataset_name_2x1_hh=intermediate_orders_scored_dataset_prefix_2x1_hh+'_'+scoring_week_start_date_suffix
intermediate_aov_scored_dataset_name_2x1_hh=intermediate_aov_scored_dataset_prefix_2x1_hh+'_'+scoring_week_start_date_suffix
intermediate_d2p_scored_dataset_name_2x1_hh=intermediate_d2p_scored_dataset_prefix_2x1_hh+'_'+scoring_week_start_date_suffix

intermediate_orders_scored_dataset_name_2x1_ugc=intermediate_orders_scored_dataset_prefix_2x1_ugc+'_'+scoring_week_start_date_suffix
intermediate_aov_scored_dataset_name_2x1_ugc=intermediate_aov_scored_dataset_prefix_2x1_ugc+'_'+scoring_week_start_date_suffix
intermediate_d2p_scored_dataset_name_2x1_ugc=intermediate_d2p_scored_dataset_prefix_2x1_ugc+'_'+scoring_week_start_date_suffix

# Scoring dataset name
scoring_dataset_name = scoring_dataset_name_prefix + '_'  +  scoring_week_start_date_suffix +'_sbctqnt_w_coded_demogs'

#Scored dataset name
scored_dataset_name_2x0= scored_dataset_name_prefix_2x0 + '_'  +  scoring_week_start_date_suffix
scored_dataset_name_2x1_hh= scored_dataset_name_prefix_2x1_hh + '_'  +  scoring_week_start_date_suffix
scored_dataset_name_2x1_ugc= scored_dataset_name_prefix_2x1_ugc + '_'  +  scoring_week_start_date_suffix


# In[14]:


scoring_dataset_name


# ### Loading the features and creating the pipeline

# In[15]:


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


# In[16]:


assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
#labelIndexer = StringIndexer(inputCol='label', outputCol="label")
tmp = [assembler_features]
pipeline = Pipeline(stages=tmp)


# In[17]:


feature_df = cols_now
feature_df = pd.DataFrame()
feature_df['feature']=cols_now
feature_df.reset_index(inplace=True)
feature_df['index'] = feature_df.index


# # Scoring 2x0 Models

# In[105]:


train_hh_df_fetch_query_all_hh_scored="""select
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

if(post_order_cnt_dotcom_inc_ct_dt_15d >0,1,0) as activation_flag_15,
if(post_order_cnt_dotcom_inc_ct_dt_90days >0,1,0) as activation_flag_90,
if(post_order_cnt_dotcom_inc_ct_dt_30d >0,1,0) as activation_flag_30,
datediff(purchase_dt_dotcom, {}) as post_days_to_purchase
from {}
where {}
""".format(scoring_week_start_date,scoring_dataset_name, scoring_pre_period_conditions_2x0)


# In[106]:


train_hh_df_all_hh_scored=spark.sql(train_hh_df_fetch_query_all_hh_scored)


# In[107]:


train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_dotcom",train_hh_df_all_hh_scored.M_DOTCOM/train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_store",train_hh_df_all_hh_scored.M_STORE/train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_OG",train_hh_df_all_hh_scored.M_OG/train_hh_df_all_hh_scored.F_OG)

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_90",train_hh_df_all_hh_scored.post_spend_dotcom_inc_ct_dt_90days /train_hh_df_all_hh_scored.post_order_cnt_dotcom_inc_ct_dt_90days )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_30",train_hh_df_all_hh_scored.post_spend_dotcom_inc_ct_dt_30d /train_hh_df_all_hh_scored.post_order_cnt_dotcom_inc_ct_dt_30d )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_15",train_hh_df_all_hh_scored.post_spend_dotcom_inc_ct_dt_15d /train_hh_df_all_hh_scored.post_order_cnt_dotcom_inc_ct_dt_15d )

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_dotcom",train_hh_df_all_hh_scored.R_DOTCOM*train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_store",train_hh_df_all_hh_scored.R_STORE*train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_OG",train_hh_df_all_hh_scored.R_OG*train_hh_df_all_hh_scored.F_OG)


# In[108]:


#features_1 = list(set(features)-set(['activation_flag_90','activation_flag_30','activation_flag_15',
#                                   'post_aov_og_90',
#'post_aov_og_30', 
#'post_aov_og_15',
#'post_order_cnt_og_inc_ct_dt_90days',
#'post_order_cnt_og_inc_ct_dt_30d',
#'post_order_cnt_og_inc_ct_dt_15d', 
#'post_spend_og_inc_ct_dt_90days',
#'post_spend_og_inc_ct_dt_30d',
#'post_spend_og_inc_ct_dt_15d',
#'post_days_to_purchase']))


# In[109]:


df_class=train_hh_df_all_hh_scored.select(features)
df_class = df_class.fillna(0)
allData = pipeline.fit(df_class).transform(df_class)


# In[110]:


del(train_hh_df_all_hh_scored)
del(df_class)
gc.collect()


# #### Predicting Orders

# In[111]:


fit_reg_order= RandomForestRegressionModel.load(regression_model_names_2x0[0])


# In[112]:


scored_retention_all_hh_orders= fit_reg_order.transform(allData)


# In[113]:


scored_retention_final_all_hh_orders = scored_retention_all_hh_orders.select('prediction','household_id', 'traced_flag')


# In[114]:


del(scored_retention_all_hh_orders)
gc.collect()


# In[115]:


scored_retention_final_all_hh_orders.registerTempTable('scored_retention_final_results_all_hh_orders')


# In[116]:


spark.catalog.clearCache()


# In[117]:


drop="""drop table if exists {}""".format(intermediate_orders_scored_dataset_name_2x0)
spark.sql(drop)


# In[118]:


query="create table {} stored as orc as select * from scored_retention_final_results_all_hh_orders".format(intermediate_orders_scored_dataset_name_2x0)
spark.sql(query)
print 'Done!'


# In[119]:


del(scored_retention_final_all_hh_orders)
gc.collect()


# In[120]:


intermediate_orders_scored_dataset_name_2x0


# In[121]:


q="select traced_flag,count(*), avg(prediction) from {} group by traced_flag".format(intermediate_orders_scored_dataset_name_2x0)
spark.sql(q).show(3000,False)


# #### Predicting AOV

# In[122]:


fit_reg_aov= RandomForestRegressionModel.load(regression_model_names_2x0[1])


# In[123]:


scored_retention_all_hh_aov= fit_reg_aov.transform(allData)


# In[124]:


scored_retention_final_all_hh_aov = scored_retention_all_hh_aov.select('prediction','household_id', 'traced_flag')


# In[125]:


del(scored_retention_all_hh_aov)
gc.collect()


# In[126]:


scored_retention_final_all_hh_aov.registerTempTable('scored_retention_final_results_all_hh_aov')


# In[127]:


spark.catalog.clearCache()


# In[128]:


drop="""drop table if exists {}""".format(intermediate_aov_scored_dataset_name_2x0)
spark.sql(drop)


# In[129]:


query="create table {} stored as orc as select * from scored_retention_final_results_all_hh_aov".format(intermediate_aov_scored_dataset_name_2x0)
spark.sql(query)
print 'Done!'


# In[130]:


del(scored_retention_final_all_hh_aov)
gc.collect()


# In[131]:


q="select count(*), avg(prediction) from {}".format(intermediate_aov_scored_dataset_name_2x0)
spark.sql(q).show(3000,False)


# #### Predicting Days to Purchase

# In[132]:


fit_reg_d2p= RandomForestRegressionModel.load(regression_model_names_2x0[2])


# In[133]:


scored_retention_all_hh_d2p= fit_reg_d2p.transform(allData)


# In[134]:


scored_retention_final_all_hh_d2p = scored_retention_all_hh_d2p.select('prediction','household_id', 'traced_flag' )


# In[135]:


del(scored_retention_all_hh_d2p)
gc.collect()


# In[136]:


scored_retention_final_all_hh_d2p.registerTempTable('scored_retention_final_results_all_hh_d2p')


# In[137]:


drop="""drop table if exists {}""".format(intermediate_d2p_scored_dataset_name_2x0)
spark.sql(drop)


# In[138]:


query="create table {} stored as orc as select * from scored_retention_final_results_all_hh_d2p".format(intermediate_d2p_scored_dataset_name_2x0)
spark.sql(query)
print 'Done!'


# In[139]:


del(scored_retention_final_all_hh_d2p)
gc.collect()


# In[140]:


q="select count(*), avg(prediction) from {}".format(intermediate_d2p_scored_dataset_name_2x0)
spark.sql(q).show(3000,False)


# In[141]:


intermediate_d2p_scored_dataset_name_2x0


# In[142]:


q="select traced_flag,count(*), avg(prediction) from {} group by traced_flag".format(intermediate_d2p_scored_dataset_name_2x0)
spark.sql(q).show(3000,False)


# #### Save a table with all 3 predictions

# In[143]:


spark.catalog.clearCache()


# In[144]:


scored_dataset_name_2x0


# In[145]:


query="""select orders.household_id,orders.traced_flag,
orders.prediction as orders_prediction,
aov.prediction as aov_prediction,
d2p.prediction as d2p_prediction
from
{} orders
left join 
{} aov
on orders.household_id=aov.household_id
and orders.traced_flag=aov.traced_flag
left join 
{} d2p
on orders.household_id=d2p.household_id
and orders.traced_flag=d2p.traced_flag
""".format(intermediate_orders_scored_dataset_name_2x0,intermediate_aov_scored_dataset_name_2x0,intermediate_d2p_scored_dataset_name_2x0)
coll=spark.sql(query)
coll.write.saveAsTable(scored_dataset_name_2x0, format='ORC', mode = 'OVERWRITE')


# In[146]:


del(coll)
gc.collect()


# In[147]:


query="""select traced_flag, 
avg(orders_prediction),
avg(aov_prediction),
avg(d2p_prediction)
from 
{}
group by traced_flag
""".format(scored_dataset_name_2x0)
spark.sql(query).show(3000,False)


# # Scoring 2x1 HH Models

# In[148]:


train_hh_df_fetch_query_all_hh_scored="""select
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

if(post_order_cnt_dotcom_inc_ct_dt_15d >0,1,0) as activation_flag_15,
if(post_order_cnt_dotcom_inc_ct_dt_90days >0,1,0) as activation_flag_90,
if(post_order_cnt_dotcom_inc_ct_dt_30d >0,1,0) as activation_flag_30,
datediff(purchase_dt_dotcom, {}) as post_days_to_purchase
from {}
where {}
""".format(scoring_week_start_date,scoring_dataset_name, scoring_pre_period_conditions_2x1_hh)


# In[149]:


train_hh_df_all_hh_scored=spark.sql(train_hh_df_fetch_query_all_hh_scored)


# In[150]:


train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_dotcom",train_hh_df_all_hh_scored.M_DOTCOM/train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_store",train_hh_df_all_hh_scored.M_STORE/train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_OG",train_hh_df_all_hh_scored.M_OG/train_hh_df_all_hh_scored.F_OG)

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_90",train_hh_df_all_hh_scored.post_spend_dotcom_inc_ct_dt_90days /train_hh_df_all_hh_scored.post_order_cnt_dotcom_inc_ct_dt_90days )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_30",train_hh_df_all_hh_scored.post_spend_dotcom_inc_ct_dt_30d /train_hh_df_all_hh_scored.post_order_cnt_dotcom_inc_ct_dt_30d )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_15",train_hh_df_all_hh_scored.post_spend_dotcom_inc_ct_dt_15d /train_hh_df_all_hh_scored.post_order_cnt_dotcom_inc_ct_dt_15d )

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_dotcom",train_hh_df_all_hh_scored.R_DOTCOM*train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_store",train_hh_df_all_hh_scored.R_STORE*train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_OG",train_hh_df_all_hh_scored.R_OG*train_hh_df_all_hh_scored.F_OG)


# In[151]:


#features_1 = list(set(features)-set(['activation_flag_90','activation_flag_30','activation_flag_15',
#                                   'post_aov_og_90',
#'post_aov_og_30', 
#'post_aov_og_15',
#'post_order_cnt_og_inc_ct_dt_90days',
#'post_order_cnt_og_inc_ct_dt_30d',
#'post_order_cnt_og_inc_ct_dt_15d', 
#'post_spend_og_inc_ct_dt_90days',
#'post_spend_og_inc_ct_dt_30d',
#'post_spend_og_inc_ct_dt_15d',
#'post_days_to_purchase']))


# In[152]:


df_class=train_hh_df_all_hh_scored.select(features)
df_class = df_class.fillna(0)
allData = pipeline.fit(df_class).transform(df_class)


# In[153]:


del(train_hh_df_all_hh_scored)
del(df_class)
gc.collect()


# In[154]:


spark.catalog.clearCache()


# #### Predicting Orders

# In[155]:


fit_reg_order= RandomForestRegressionModel.load(regression_model_names_2x1_hh[0])


# In[156]:


scored_retention_all_hh_orders= fit_reg_order.transform(allData)


# In[157]:


scored_retention_final_all_hh_orders = scored_retention_all_hh_orders.select('prediction','household_id', 'traced_flag')


# In[158]:


del(scored_retention_all_hh_orders)
gc.collect()


# In[159]:


scored_retention_final_all_hh_orders.registerTempTable('scored_retention_final_results_all_hh_orders')


# In[160]:


drop="""drop table if exists {}""".format(intermediate_orders_scored_dataset_name_2x1_hh)
spark.sql(drop)


# In[161]:


query="create table {} stored as orc as select * from scored_retention_final_results_all_hh_orders".format(intermediate_orders_scored_dataset_name_2x1_hh)
spark.sql(query)
print 'Done!'


# In[162]:


del(scored_retention_final_all_hh_orders)
gc.collect()


# In[163]:


intermediate_orders_scored_dataset_name_2x1_hh


# In[164]:


q="select traced_flag,count(*), avg(prediction) from {} group by traced_flag".format(intermediate_orders_scored_dataset_name_2x1_hh)
spark.sql(q).show(3000,False)


# #### Predicting AOV

# In[165]:


fit_reg_aov= RandomForestRegressionModel.load(regression_model_names_2x1_hh[1])


# In[166]:


scored_retention_all_hh_aov= fit_reg_aov.transform(allData)


# In[167]:


scored_retention_final_all_hh_aov = scored_retention_all_hh_aov.select('prediction','household_id', 'traced_flag')


# In[168]:


del(scored_retention_all_hh_aov)
gc.collect()


# In[169]:


scored_retention_final_all_hh_aov.registerTempTable('scored_retention_final_results_all_hh_aov')


# In[170]:


drop="""drop table if exists {}""".format(intermediate_aov_scored_dataset_name_2x1_hh)
spark.sql(drop)


# In[171]:


query="create table {} stored as orc as select * from scored_retention_final_results_all_hh_aov".format(intermediate_aov_scored_dataset_name_2x1_hh)
spark.sql(query)
print 'Done!'


# In[172]:


del(scored_retention_final_all_hh_aov)
gc.collect()


# In[173]:


q="select traced_flag,count(*), avg(prediction) from {} group by traced_flag".format(intermediate_aov_scored_dataset_name_2x1_hh)
spark.sql(q).show(3000,False)


# In[174]:


spark.catalog.clearCache()


# #### Predicting Days to Purchase

# In[63]:


fit_reg_d2p= RandomForestRegressionModel.load(regression_model_names_2x1_hh[2])


# In[64]:


scored_retention_all_hh_d2p= fit_reg_d2p.transform(allData)


# In[65]:


scored_retention_final_all_hh_d2p = scored_retention_all_hh_d2p.select('prediction','household_id', 'traced_flag' )


# In[66]:


del(scored_retention_all_hh_d2p)
gc.collect()


# In[67]:


scored_retention_final_all_hh_d2p.registerTempTable('scored_retention_final_results_all_hh_d2p')


# In[68]:


drop="""drop table if exists {}""".format(intermediate_d2p_scored_dataset_name_2x1_hh)
spark.sql(drop)


# In[69]:


query="create table {} stored as orc as select * from scored_retention_final_results_all_hh_d2p".format(intermediate_d2p_scored_dataset_name_2x1_hh)
spark.sql(query)
print 'Done!'


# In[70]:


del(scored_retention_final_all_hh_d2p)
gc.collect()


# In[71]:


q="select traced_flag,count(*), avg(prediction) from {} group by traced_flag".format(intermediate_d2p_scored_dataset_name_2x1_hh)
spark.sql(q).show(3000,False)


# In[72]:


intermediate_d2p_scored_dataset_name_2x1_hh


# In[73]:


q="select traced_flag,count(*), avg(prediction) from {} group by traced_flag".format(intermediate_d2p_scored_dataset_name_2x1_hh)
spark.sql(q).show(3000,False)


# #### Save a table with all 3 predictions

# In[175]:


spark.catalog.clearCache()


# In[176]:


scored_dataset_name_2x1_hh


# In[177]:


query="""select orders.household_id,orders.traced_flag,
orders.prediction as orders_prediction,
aov.prediction as aov_prediction,
d2p.prediction as d2p_prediction
from
{} orders
left join 
{} aov
on orders.household_id=aov.household_id
and orders.traced_flag=aov.traced_flag
left join 
{} d2p
on orders.household_id=d2p.household_id
and orders.traced_flag=d2p.traced_flag
""".format(intermediate_orders_scored_dataset_name_2x1_hh,intermediate_aov_scored_dataset_name_2x1_hh,intermediate_d2p_scored_dataset_name_2x1_hh)
coll=spark.sql(query)
coll.write.saveAsTable(scored_dataset_name_2x1_hh, format='ORC', mode = 'OVERWRITE')


# In[178]:


del(coll)
gc.collect()


# In[180]:


query="""select traced_flag, 
avg(orders_prediction),
avg(aov_prediction),
avg(d2p_prediction)
from 
{}
group by traced_flag
""".format(scored_dataset_name_2x1_hh)
spark.sql(query).show(3000,False)


# # Scoring 2x1 UGC Models

# In[194]:


train_hh_df_fetch_query_all_hh_scored="""select
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

if(post_order_cnt_dotcom_inc_ct_dt_15d >0,1,0) as activation_flag_15,
if(post_order_cnt_dotcom_inc_ct_dt_90days >0,1,0) as activation_flag_90,
if(post_order_cnt_dotcom_inc_ct_dt_30d >0,1,0) as activation_flag_30,
datediff(purchase_dt_dotcom, {}) as post_days_to_purchase
from {}
where {}
""".format(scoring_week_start_date,scoring_dataset_name, scoring_pre_period_conditions_2x1_ugc)


# In[195]:


train_hh_df_all_hh_scored=spark.sql(train_hh_df_fetch_query_all_hh_scored)


# In[196]:


train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_dotcom",train_hh_df_all_hh_scored.M_DOTCOM/train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_store",train_hh_df_all_hh_scored.M_STORE/train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_OG",train_hh_df_all_hh_scored.M_OG/train_hh_df_all_hh_scored.F_OG)

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_90",train_hh_df_all_hh_scored.post_spend_dotcom_inc_ct_dt_90days /train_hh_df_all_hh_scored.post_order_cnt_dotcom_inc_ct_dt_90days )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_30",train_hh_df_all_hh_scored.post_spend_dotcom_inc_ct_dt_30d /train_hh_df_all_hh_scored.post_order_cnt_dotcom_inc_ct_dt_30d )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_15",train_hh_df_all_hh_scored.post_spend_dotcom_inc_ct_dt_15d /train_hh_df_all_hh_scored.post_order_cnt_dotcom_inc_ct_dt_15d )

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_dotcom",train_hh_df_all_hh_scored.R_DOTCOM*train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_store",train_hh_df_all_hh_scored.R_STORE*train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_OG",train_hh_df_all_hh_scored.R_OG*train_hh_df_all_hh_scored.F_OG)


# In[197]:


#features_1 = list(set(features)-set(['activation_flag_90','activation_flag_30','activation_flag_15',
#                                   'post_aov_og_90',
#'post_aov_og_30', 
#'post_aov_og_15',
#'post_order_cnt_og_inc_ct_dt_90days',
#'post_order_cnt_og_inc_ct_dt_30d',
#'post_order_cnt_og_inc_ct_dt_15d', 
#'post_spend_og_inc_ct_dt_90days',
#'post_spend_og_inc_ct_dt_30d',
#'post_spend_og_inc_ct_dt_15d',
#'post_days_to_purchase']))


# In[198]:


df_class=train_hh_df_all_hh_scored.select(features)
df_class = df_class.fillna(0)
allData = pipeline.fit(df_class).transform(df_class)


# In[199]:


del(train_hh_df_all_hh_scored)
del(df_class)
gc.collect()


# #### Predicting Orders

# In[200]:


fit_reg_order= RandomForestRegressionModel.load(regression_model_names_2x1_ugc[0])


# In[201]:


scored_retention_all_hh_orders= fit_reg_order.transform(allData)


# In[202]:


scored_retention_final_all_hh_orders = scored_retention_all_hh_orders.select('prediction','household_id', 'traced_flag')


# In[203]:


del(scored_retention_all_hh_orders)
gc.collect()


# In[204]:


scored_retention_final_all_hh_orders.registerTempTable('scored_retention_final_results_all_hh_orders')


# In[205]:


drop="""drop table if exists {}""".format(intermediate_orders_scored_dataset_name_2x1_ugc)
spark.sql(drop)


# In[206]:


query="create table {} stored as orc as select * from scored_retention_final_results_all_hh_orders".format(intermediate_orders_scored_dataset_name_2x1_ugc)
spark.sql(query)
print 'Done!'


# In[207]:


del(scored_retention_final_all_hh_orders)
gc.collect()


# In[208]:


intermediate_orders_scored_dataset_name_2x1_ugc


# In[209]:


q="select traced_flag,count(*), avg(prediction) from {} group by traced_flag".format(intermediate_orders_scored_dataset_name_2x1_ugc)
spark.sql(q).show(3000,False)


# #### Predicting AOV

# In[210]:


fit_reg_aov= RandomForestRegressionModel.load(regression_model_names_2x1_ugc[1])


# In[211]:


scored_retention_all_hh_aov= fit_reg_aov.transform(allData)


# In[212]:


scored_retention_final_all_hh_aov = scored_retention_all_hh_aov.select('prediction','household_id', 'traced_flag')


# In[215]:


del(scored_retention_all_hh_aov)
gc.collect()


# In[216]:


scored_retention_final_all_hh_aov.registerTempTable('scored_retention_final_results_all_hh_aov')


# In[217]:


drop="""drop table if exists {}""".format(intermediate_aov_scored_dataset_name_2x1_ugc)
spark.sql(drop)


# In[218]:


query="create table {} stored as orc as select * from scored_retention_final_results_all_hh_aov".format(intermediate_aov_scored_dataset_name_2x1_ugc)
spark.sql(query)
print 'Done!'


# In[219]:


del(scored_retention_final_all_hh_aov)
gc.collect()


# In[220]:


q="select traced_flag,count(*), avg(prediction) from {} group by traced_flag".format(intermediate_aov_scored_dataset_name_2x1_ugc)
spark.sql(q).show(3000,False)


# In[221]:


spark.catalog.clearCache()


# #### Predicting Days to Purchase

# In[80]:


fit_reg_d2p= RandomForestRegressionModel.load(regression_model_names_2x1_ugc[2])


# In[81]:


scored_retention_all_hh_d2p= fit_reg_d2p.transform(allData)


# In[82]:


scored_retention_final_all_hh_d2p = scored_retention_all_hh_d2p.select('prediction','household_id', 'traced_flag' )


# In[83]:


del(scored_retention_all_hh_d2p)
gc.collect()


# In[84]:


scored_retention_final_all_hh_d2p.registerTempTable('scored_retention_final_results_all_hh_d2p')


# In[85]:


drop="""drop table if exists {}""".format(intermediate_d2p_scored_dataset_name_2x1_ugc)
spark.sql(drop)


# In[86]:


query="create table {} stored as orc as select * from scored_retention_final_results_all_hh_d2p".format(intermediate_d2p_scored_dataset_name_2x1_ugc)
spark.sql(query)
print 'Done!'


# In[87]:


del(scored_retention_final_all_hh_d2p)
gc.collect()


# In[88]:


q="select count(*), avg(prediction) from {}".format(intermediate_d2p_scored_dataset_name_2x1_ugc)
spark.sql(q).show(3000,False)


# In[89]:


intermediate_d2p_scored_dataset_name_2x1_ugc


# In[90]:


q="select traced_flag,count(*), avg(prediction) from {} group by traced_flag".format(intermediate_d2p_scored_dataset_name_2x1_ugc)
spark.sql(q).show(3000,False)


# #### Save a table with all 3 predictions

# In[222]:


spark.catalog.clearCache()


# In[223]:


scored_dataset_name_2x1_ugc


# In[224]:


query="""select orders.household_id,orders.traced_flag,
orders.prediction as orders_prediction,
aov.prediction as aov_prediction,
d2p.prediction as d2p_prediction
from
{} orders
left join 
{} aov
on orders.household_id=aov.household_id
and orders.traced_flag=aov.traced_flag
left join 
{} d2p
on orders.household_id=d2p.household_id
and orders.traced_flag=d2p.traced_flag
""".format(intermediate_orders_scored_dataset_name_2x1_ugc,intermediate_aov_scored_dataset_name_2x1_ugc,intermediate_d2p_scored_dataset_name_2x1_ugc)
coll=spark.sql(query)
coll.write.saveAsTable(scored_dataset_name_2x1_ugc, format='ORC', mode = 'OVERWRITE')


# In[225]:


del(coll)
gc.collect()


# In[227]:


query="""select traced_flag, 
avg(orders_prediction),
avg(aov_prediction),
avg(d2p_prediction)
from 
{}
group by traced_flag
""".format(scored_dataset_name_2x1_ugc)
spark.sql(query).show(3000,False)


# In[228]:


spark.stop()

