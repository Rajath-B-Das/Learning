
# coding: utf-8

# In[1]:


import datetime
import time
from datetime import date, timedelta, datetime


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
from pyspark.ml.feature import IndexToString

from datetime import datetime
import unicodedata

   
username = os.environ.get('USER')
username_hive = username.replace( "-", "_" )

from os.path import expanduser, join
from pyspark.sql import SparkSession
from pyspark.sql import Row

    
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

SPARK_CONF = SparkConf().setAppName("Dotcom_Model_Scoring").setAll(confList)

spark = SparkSession.builder.config(conf=SPARK_CONF).enableHiveSupport().getOrCreate()

from datetime import datetime
import unicodedata

import gc


# In[7]:


import getpass
#Getting the schema name
schema_name = getpass.getuser()
schema_name 


# In[8]:


#### Getting week start date 


# In[9]:


#for current date
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


#For scoring
score_wkstdt="\'{}\'".format(SCORING_WEEK_ST_DT)
    


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
    
scoring_week_start_date_suffix=str(year) +'_'+ month_str +'_'+ day_str


# In[15]:


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
    
week_start_date_suffix=str(year)+'_'+month_str+'_'+ day_str


print week_start_date_suffix



# Model schema name
model_schema_name='vn500ij'

# Model type 
activation='90d'

#Scoring schema name
scoring_schema_name = 'mjoshi2'

# Scoring dataset name
scoring_dataset_name=scoring_schema_name+'.mu_rfm_metrics_w_pu_bb_flags_added_w_mdse_ogsd_sq_'+scoring_week_start_date_suffix+'_sbctqnt_w_coded_demogs'


# In[18]:


scoring_select="""
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
"""


# In[19]:


feature_cols_table='mjoshi2.prop_and_exp_value_features_as_cols'
    
colsnow_cols_table='mjoshi2.prop_and_exp_value_colsnow_as_cols'

q="""desc {}""".format(feature_cols_table)
col_names=spark.sql(q).toPandas()
features_raw=col_names['col_name'].tolist()
features = [str(item) for item in features_raw]


# In[20]:


q="""desc {}""".format(colsnow_cols_table)
col_names_colsnow=spark.sql(q).toPandas() 
cols_now_raw=col_names_colsnow['col_name'].tolist()
cols_now = [str(item) for item in cols_now_raw]

assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
#labelIndexer = StringIndexer(inputCol='label', outputCol="label")
tmp = [assembler_features]
pipeline = Pipeline(stages=tmp)


# ### 2x0 Dotcom

# In[21]:


spark.catalog.clearCache()


# In[22]:


#Customer Base
cust_base='2x0_dotcom'

# Enter the desired name for the model
model_name= '/user/' + model_schema_name + '/online_activation_'+cust_base+'_model_'+activation+'_'+ week_start_date_suffix

# Model Type
model_type_prefix='online_activation_'+cust_base+'_model_'+activation
model_type=model_type_prefix+'_'+ week_start_date_suffix

# Preperiod condition to subset customer base
pre_period_conditions="""(traced_flag=1 and f_store_exc_ct_dt>=2 and coalesce(f_dotcom_exc_ct_dt,0)+coalesce(f_og_exc_ct_dt,0)=0) """

#Scored dataset name prefix
scored_dataset_name = schema_name + '.'+model_type_prefix+'_'+scoring_week_start_date_suffix+'_scored_dataset'


# In[23]:


print model_name
print model_type
print scoring_dataset_name
print scored_dataset_name


train_hh_df_fetch_query_all_hh_scored=scoring_select.format(score_wkstdt, scoring_dataset_name, pre_period_conditions)

train_hh_df_all_hh_scored=spark.sql(train_hh_df_fetch_query_all_hh_scored)


# In[25]:


train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_dotcom",train_hh_df_all_hh_scored.M_DOTCOM/train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_store",train_hh_df_all_hh_scored.M_STORE/train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_OG",train_hh_df_all_hh_scored.M_OG/train_hh_df_all_hh_scored.F_OG)

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_90",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_90days/train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_90days )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_30",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_30d /train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_30d )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_15",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_15d /train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_15d )

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_dotcom",train_hh_df_all_hh_scored.R_DOTCOM*train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_store",train_hh_df_all_hh_scored.R_STORE*train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_OG",train_hh_df_all_hh_scored.R_OG*train_hh_df_all_hh_scored.F_OG)
# In[26]:



# In[27]:


df_class=train_hh_df_all_hh_scored.select(features)
 
del(train_hh_df_all_hh_scored)
gc.collect()
 
df_class = df_class.fillna(0)
allData = pipeline.fit(df_class).transform(df_class)
    
del(df_class)
gc.collect()


fit_classifier = RandomForestClassificationModel.load(model_name)

retention_all_hh= fit_classifier.transform(allData)

#firstelement=udf(lambda v:float(v[0]),FloatType())
secondelement=udf(lambda v:float(v[1]),FloatType())
retention_final_all_hh = retention_all_hh.select(secondelement('probability').alias('pred_activation'),
                                                 'household_id', 'traced_flag')

del(retention_all_hh)
gc.collect()


retention_final_all_hh.registerTempTable('retention_final_results_all_hh_scored')


drop = """
drop table if exists {}""".format(scored_dataset_name)
spark.sql(drop)


test= """
create table {} stored as ORC as 
select * from retention_final_results_all_hh_scored""".format(scored_dataset_name)
spark.sql(test)


del(retention_final_all_hh)
gc.collect()

str_op="Completed saving: " + scored_dataset_name
print(str_op)

spark.catalog.clearCache()

# ### 2x1 Dotcom UGC


spark.catalog.clearCache()

#Customer Base
cust_base='2x1_dotcom_ugc'

# Enter the desired name for the model
model_name= '/user/' + model_schema_name + '/online_activation_'+cust_base+'_model_'+activation+'_'+ week_start_date_suffix

# Model Type
model_type_prefix='online_activation_'+cust_base+'_model_'+activation
model_type=model_type_prefix+'_'+ week_start_date_suffix

# Preperiod condition to subset customer base
pre_period_conditions="""(traced_flag=0 and coalesce(f_dotcom_exc_ct_dt,0)+coalesce(f_og_exc_ct_dt,0)>=1)"""

#Scored dataset name prefix
scored_dataset_name = schema_name + '.'+model_type_prefix+'_'+scoring_week_start_date_suffix+'_scored_dataset'


# In[50]:


print model_name
print model_type
print scoring_dataset_name
print scored_dataset_name


train_hh_df_fetch_query_all_hh_scored=scoring_select.format(score_wkstdt, scoring_dataset_name, pre_period_conditions)

train_hh_df_all_hh_scored=spark.sql(train_hh_df_fetch_query_all_hh_scored)


train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_dotcom",train_hh_df_all_hh_scored.M_DOTCOM/train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_store",train_hh_df_all_hh_scored.M_STORE/train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_OG",train_hh_df_all_hh_scored.M_OG/train_hh_df_all_hh_scored.F_OG)

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_90",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_90days/train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_90days )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_30",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_30d /train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_30d )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_15",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_15d /train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_15d )

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_dotcom",train_hh_df_all_hh_scored.R_DOTCOM*train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_store",train_hh_df_all_hh_scored.R_STORE*train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_OG",train_hh_df_all_hh_scored.R_OG*train_hh_df_all_hh_scored.F_OG)


df_class=train_hh_df_all_hh_scored.select(features)
 
del(train_hh_df_all_hh_scored)
gc.collect()
 
df_class = df_class.fillna(0)
allData = pipeline.fit(df_class).transform(df_class)
    
del(df_class)
gc.collect()


fit_classifier = RandomForestClassificationModel.load(model_name)

retention_all_hh= fit_classifier.transform(allData)


#firstelement=udf(lambda v:float(v[0]),FloatType())
secondelement=udf(lambda v:float(v[1]),FloatType())
retention_final_all_hh = retention_all_hh.select(secondelement('probability').alias('pred_activation'),
                                                 'household_id', 'traced_flag')

del(retention_all_hh)
gc.collect()

# In[31]:


retention_final_all_hh.registerTempTable('retention_final_results_all_hh_scored')

drop = """
drop table if exists {}""".format(scored_dataset_name)
spark.sql(drop)


# In[34]:


test= """
create table {} stored as ORC as 
select * from retention_final_results_all_hh_scored""".format(scored_dataset_name)
spark.sql(test)


del(retention_final_all_hh)
gc.collect()

str_op="Completed saving: " + scored_dataset_name
print(str_op)

spark.catalog.clearCache()


#Customer Base
cust_base='2x0_dotcom_app'

# Enter the desired name for the model
model_name= '/user/' + model_schema_name + '/online_activation_'+cust_base+'_model_'+activation+'_'+ week_start_date_suffix

# Model Type
model_type_prefix='online_activation_'+cust_base+'_model_'+activation
model_type=model_type_prefix+'_'+ week_start_date_suffix

# Preperiod condition to subset customer base
pre_period_conditions="""(traced_flag=1 and f_store_exc_ct_dt>=2 and coalesce(f_dotcom_exc_ct_dt,0)+coalesce(f_og_exc_ct_dt,0)=0)"""

#Scored dataset name prefix
scored_dataset_name = schema_name + '.'+model_type_prefix+'_'+scoring_week_start_date_suffix+'_scored_dataset'


print model_name
print model_type
print scoring_dataset_name
print scored_dataset_name


train_hh_df_fetch_query_all_hh_scored=scoring_select.format(score_wkstdt, scoring_dataset_name, pre_period_conditions)

train_hh_df_all_hh_scored=spark.sql(train_hh_df_fetch_query_all_hh_scored)


train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_dotcom",train_hh_df_all_hh_scored.M_DOTCOM/train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_store",train_hh_df_all_hh_scored.M_STORE/train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_OG",train_hh_df_all_hh_scored.M_OG/train_hh_df_all_hh_scored.F_OG)

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_90",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_90days/train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_90days )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_30",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_30d /train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_30d )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_15",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_15d /train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_15d )

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_dotcom",train_hh_df_all_hh_scored.R_DOTCOM*train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_store",train_hh_df_all_hh_scored.R_STORE*train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_OG",train_hh_df_all_hh_scored.R_OG*train_hh_df_all_hh_scored.F_OG)

df_class=train_hh_df_all_hh_scored.select(features)
 
del(train_hh_df_all_hh_scored)
gc.collect()
 
df_class = df_class.fillna(0)
allData = pipeline.fit(df_class).transform(df_class)
    
del(df_class)
gc.collect()



fit_classifier = RandomForestClassificationModel.load(model_name)

retention_all_hh= fit_classifier.transform(allData)


#firstelement=udf(lambda v:float(v[0]),FloatType())
secondelement=udf(lambda v:float(v[1]),FloatType())
retention_final_all_hh = retention_all_hh.select(secondelement('probability').alias('pred_activation'),
                                                 'household_id', 'traced_flag')

del(retention_all_hh)
gc.collect()

retention_final_all_hh.registerTempTable('retention_final_results_all_hh_scored')

drop = """
drop table if exists {}""".format(scored_dataset_name)
spark.sql(drop)

test= """
create table {} stored as ORC as 
select * from retention_final_results_all_hh_scored""".format(scored_dataset_name)
spark.sql(test)


del(retention_final_all_hh)
gc.collect()

str_op="Completed saving: " + scored_dataset_name
print(str_op)

spark.catalog.clearCache()


#Customer Base
cust_base='2x1_dotcom_app'

# Enter the desired name for the model
model_name= '/user/' + model_schema_name + '/online_activation_'+cust_base+'_model_'+activation+'_'+ week_start_date_suffix

# Model Type
model_type_prefix='online_activation_'+cust_base+'_model_'+activation
model_type=model_type_prefix+'_'+ week_start_date_suffix

# Preperiod condition to subset customer base
pre_period_conditions="""(traced_flag=1 and f_store_exc_ct_dt>=2 and coalesce(f_dotcom_exc_ct_dt,0)+coalesce(f_og_exc_ct_dt,0)>=1)
or (traced_flag=0 and coalesce(f_dotcom_exc_ct_dt,0)+coalesce(f_og_exc_ct_dt,0)>=1)"""

#Scored dataset name prefix
scored_dataset_name = schema_name + '.'+model_type_prefix+'_'+scoring_week_start_date_suffix+'_scored_dataset'


# In[86]:


print model_name
print model_type
print scoring_dataset_name
print scored_dataset_name


# In[24]:


train_hh_df_fetch_query_all_hh_scored=scoring_select.format(score_wkstdt, scoring_dataset_name, pre_period_conditions)

train_hh_df_all_hh_scored=spark.sql(train_hh_df_fetch_query_all_hh_scored)


# In[25]:


train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_dotcom",train_hh_df_all_hh_scored.M_DOTCOM/train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_store",train_hh_df_all_hh_scored.M_STORE/train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_OG",train_hh_df_all_hh_scored.M_OG/train_hh_df_all_hh_scored.F_OG)

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_90",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_90days/train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_90days )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_30",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_30d /train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_30d )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_15",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_15d /train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_15d )

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_dotcom",train_hh_df_all_hh_scored.R_DOTCOM*train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_store",train_hh_df_all_hh_scored.R_STORE*train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_OG",train_hh_df_all_hh_scored.R_OG*train_hh_df_all_hh_scored.F_OG)
# In[26]:



# In[27]:


df_class=train_hh_df_all_hh_scored.select(features)
 
del(train_hh_df_all_hh_scored)
gc.collect()
 
df_class = df_class.fillna(0)
allData = pipeline.fit(df_class).transform(df_class)
    
del(df_class)
gc.collect()


fit_classifier = RandomForestClassificationModel.load(model_name)

retention_all_hh= fit_classifier.transform(allData)



#firstelement=udf(lambda v:float(v[0]),FloatType())
secondelement=udf(lambda v:float(v[1]),FloatType())
retention_final_all_hh = retention_all_hh.select(secondelement('probability').alias('pred_activation'),
                                                 'household_id', 'traced_flag')

del(retention_all_hh)
gc.collect()

# In[31]:


retention_final_all_hh.registerTempTable('retention_final_results_all_hh_scored')


# In[32]:


scored_dataset_name


# In[33]:


drop = """
drop table if exists {}""".format(scored_dataset_name)
spark.sql(drop)


# In[34]:


test= """
create table {} stored as ORC as 
select * from retention_final_results_all_hh_scored""".format(scored_dataset_name)
spark.sql(test)


del(retention_final_all_hh)
gc.collect()


str_op="Completed saving: " + scored_dataset_name
print(str_op)


# ### 2x1 Dotcom HH


spark.catalog.clearCache()

#Customer Base
cust_base='2x1_dotcom_hh'

# Enter the desired name for the model
model_name= '/user/' + model_schema_name + '/online_activation_'+cust_base+'_model_'+activation+'_'+ week_start_date_suffix

# Model Type
model_type_prefix='online_activation_'+cust_base+'_model_'+activation
model_type= model_type_prefix+'_'+ week_start_date_suffix

# Preperiod condition to subset customer base
pre_period_conditions="""(traced_flag=1 and f_store_exc_ct_dt>=2 and coalesce(f_dotcom_exc_ct_dt,0)+coalesce(f_og_exc_ct_dt,0)>=1)"""

#Scored dataset name prefix
scored_dataset_name = schema_name + '.'+model_type_prefix+'_'+scoring_week_start_date_suffix+'_scored_dataset'


print model_name
print model_type
print scoring_dataset_name
print scored_dataset_name


# In[24]:


train_hh_df_fetch_query_all_hh_scored=scoring_select.format(score_wkstdt, scoring_dataset_name, pre_period_conditions)

train_hh_df_all_hh_scored=spark.sql(train_hh_df_fetch_query_all_hh_scored)


# In[25]:


train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_dotcom",train_hh_df_all_hh_scored.M_DOTCOM/train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_store",train_hh_df_all_hh_scored.M_STORE/train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("aov_OG",train_hh_df_all_hh_scored.M_OG/train_hh_df_all_hh_scored.F_OG)

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_90",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_90days/train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_90days )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_30",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_30d /train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_30d )
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("post_aov_og_15",train_hh_df_all_hh_scored.post_spend_og_inc_ct_dt_15d /train_hh_df_all_hh_scored.post_order_cnt_og_inc_ct_dt_15d )

train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_dotcom",train_hh_df_all_hh_scored.R_DOTCOM*train_hh_df_all_hh_scored.F_DOTCOM)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_store",train_hh_df_all_hh_scored.R_STORE*train_hh_df_all_hh_scored.F_STORE)
train_hh_df_all_hh_scored=train_hh_df_all_hh_scored.withColumn("rf_OG",train_hh_df_all_hh_scored.R_OG*train_hh_df_all_hh_scored.F_OG)


df_class=train_hh_df_all_hh_scored.select(features)
 
del(train_hh_df_all_hh_scored)
gc.collect()
 
df_class = df_class.fillna(0)
allData = pipeline.fit(df_class).transform(df_class)
    
del(df_class)
gc.collect()


fit_classifier = RandomForestClassificationModel.load(model_name)

retention_all_hh= fit_classifier.transform(allData)


# In[30]:


#firstelement=udf(lambda v:float(v[0]),FloatType())
secondelement=udf(lambda v:float(v[1]),FloatType())
retention_final_all_hh = retention_all_hh.select(secondelement('probability').alias('pred_activation'),
                                                 'household_id', 'traced_flag')

del(retention_all_hh)
gc.collect()

# In[31]:


retention_final_all_hh.registerTempTable('retention_final_results_all_hh_scored')

drop = """
drop table if exists {}""".format(scored_dataset_name)
spark.sql(drop)


test= """
create table {} stored as ORC as 
select * from retention_final_results_all_hh_scored""".format(scored_dataset_name)
spark.sql(test)


del(retention_final_all_hh)
gc.collect()

str_op="Completed saving: " + scored_dataset_name
print(str_op)





spark.stop()