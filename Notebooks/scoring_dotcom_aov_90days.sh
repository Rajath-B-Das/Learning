

counter=0
model_name="scoring_dotcom_aov_90days"
retrigger_mail_subject_prefix="Prop_Mod_Build - Fail - Retriggering - Query Failed for "
failed_mail_subject_prefix="Prop_Mod_Build - Fail - Query Failed for "
succeeded_mail_subject_prefix="Prop_Mod_Build - Success - Query Succeeded for "
log_pick_up_path='/home/vn05uy5/Scheduling/Model_Scoring/Dotcom/Orders_AOV_D2P'
log_pick_up_file_extension=".txt"

retrigger_mail_subject="$retrigger_mail_subject_prefix$model_name"
failed_mail_subject="$failed_mail_subject_prefix$model_name"
succeeded_mail_subject="$succeeded_mail_subject_prefix$model_name"
log_pick_up_file="$log_pick_up_path""/$model_name$log_pick_up_file_extension"




while [ $counter -lt 3 ]; do
runstatus_code=1000
        echo "in loop"
                                echo $counter
                                /usr/local/bin/python2.7 "${log_pick_up_path}/"${model_name}.py &> "${log_pick_up_path}/"${model_name}.txt
        if [ $? -ne 0 ]
        then
                runstatus_code=`expr 1 - 1`
mail -s "${retrigger_mail_subject}" -a "${log_pick_up_file}" ashok.suthar@walmart.com ashok.suthar@mu-sigma.com Rajath.das@mu-sigma.com Mayuresh.Joshi@mu-sigma.com Meet.shah@mu-sigma.com Saheba.vasdev@mu-sigma.com <<eol
Hi All,
Query failed for ${model_name} building. Attempt # ${counter} failed.  Retriggering.
Thank you.
With regards,
Modeling team
eol
counter=`expr $counter + 1`

        else
                counter=`expr 3 + 1 - 1`

        fi
done

if [ $runstatus_code -ne 1000 ]
then
        mail -s "${failed_mail_subject}" -a "${log_pick_up_file}" ashok.suthar@walmart.com ashok.suthar@mu-sigma.com Rajath.das@mu-sigma.com Mayuresh.Joshi@mu-sigma.com Meet.shah@mu-sigma.com Saheba.vasdev@mu-sigma.com <<eol
Hi All,
Query failed for ${model_name} Model building. Apologies for the issue. We will look into it shortly.
Thank you.
With regards,
Modeling team
eol
        exit 1
else
        mail -s "${succeeded_mail_subject}"  -a "${log_pick_up_file}"     ashok.suthar@walmart.com ashok.suthar@mu-sigma.com Rajath.das@mu-sigma.com Mayuresh.Joshi@mu-sigma.com Meet.shah@mu-sigma.com Saheba.vasdev@mu-sigma.com <<eol
Hi All,
Query success for ${model_name} model building.
Thank you.
With regards,
Modelling Team
eol
fi

#mailx -s "${retrigger_mail_subject}" -a "${model_name}".txt ashok.suthar@walmart.com,ashok.suthar@mu-sigma.com,Rajath.das@mu-sigma.com,Mayuresh.Joshi@mu-sigma.com,Meet.shah@mu-sigma.com,Saheba.vasdev@mu-sigma.com

