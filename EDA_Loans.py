#!/usr/bin/env python
# coding: utf-8

# #### Author: Shruti Gupta
# #### File Name: Shruti Gupta - Mini Project - Week 10 Mini Project
# #### Date: 04/05/2019

# ## Data Cleaning and Exploratory Analysis
# In machine learning, you clean up the data and turn raw data into features from which you can derive the pattern. There are methods available to extract features that will be covered in upcoming sessions but it's very important to build the intuition. The process of data cleaning and visualization helps with that. In this assignment, we will try to manually identify the important features in the given dataset.
# 
# ### Dataset: Lending Club data
# 
# Years of data to download: 2007-2011
# 
# Load the Lending Club data into a pandas dataframe. The data contains 42538 rows and 145 columns. Not all these columns contain meaningful (or any) information so they need to be cleaned. The loans are categorized into different grades and sub-grades. It would be interesting to see whether they have any impact on the interest rates or not.
# The process should lead us into default prediction, and finding the columns that directly predict how the loan will behave. These would be our most important features.
# 
# We strongly recommend that you look in to the columns closely to see the relationship between them. This is not a guided assignment and you can use the techniques that you have learnt so far to clean and visualize the data. 
# 
# There is no one right answer but this tests your ability to handle a much larger unknown dataset.
# 
# Here are the broad guidelines:
# 
#     View the data 
#     Find the columns that are useful (may be null columns) and the ones that are not 
#     Delete the columns that are not needed
#     Clean columns values like int_rate and term by removing the string part and convert the column to numeric.
#     Identify the columns containing useful information, they would be the features. 
#     Visualize the important features
# 

# #### Importing the libraries that are required

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Loading Data

# In[2]:


loans = pd.read_csv('C:\\Users\\HP\\Documents\\Courses\\digitalvidya\\datasets\\LoanStats3a.csv', low_memory = False, header = 1,skiprows=0)


# #### Checking of loans info

# In[3]:


loans.shape


# In[4]:


loans.info()


# In[5]:


loans.head(20)


# In this stage checked about the data information and whether or not the data loading happened properly.

# #### Checking of Fields
# seeing which fields are na and cleaning/dropping them accordingly

# In[6]:


loans.isna()


# In[7]:


drop_cols=['id','member_id','url','annual_inc_joint','dti_joint','verification_status_joint','tot_cur_bal','open_acc_6m','open_act_il','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim','inq_fi','total_cu_tl','inq_last_12m',
           'acc_open_past_24mths','avg_cur_bal','bc_open_to_buy','bc_util','mo_sin_old_il_acct','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq','mths_since_recent_revol_delinq','num_accts_ever_120_pd','num_actv_bc_tl',
           'num_actv_rev_tl','num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq','percent_bc_gt_75','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit',
           'revol_bal_joint','sec_app_earliest_cr_line','sec_app_inq_last_6mths','sec_app_mort_acc','sec_app_open_acc','sec_app_revol_util','sec_app_open_act_il','sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med','sec_app_mths_since_last_major_derog','hardship_type','hardship_reason',
          'hardship_status','deferral_term','hardship_amount','hardship_start_date','hardship_end_date','payment_plan_start_date','hardship_length','hardship_dpd','hardship_loan_status','orig_projected_additional_accrued_interest','hardship_payoff_balance_amount','hardship_last_payment_amount','hardship_flag','disbursement_method']


# In[8]:


loans=loans.drop(drop_cols,axis=1)


# In[9]:


loans=loans.drop([42536,42537],axis=0)


# In[10]:


loans.shape


# In[11]:


loans.columns


# The reason to drop 82 columns and 2 rows was due there being no useful data or it being completely empty as seen when checking for null values. While there are still some null values in the columns that are not deleted it can be still be used as there is values on which data caluclation can occur.

# #### Cleaning the values in various fields.
# 
# converting string data to numeric once removing any possible string data

# In[12]:


loans['int_rate']


# In[13]:


loans['int_rate']=loans['int_rate'].str.extract('(\d*\.\d+|\d+)', expand = False)


# In[14]:


loans['int_rate']=loans.int_rate.astype(float)


# In[15]:


loans.int_rate


# In[16]:


loans.term


# In[17]:


loans.term=loans['term'].str.extract('(\d+)', expand = False)
loans['term']=loans.term.apply(pd.to_numeric)


# In[19]:


loans.term


# In[19]:


loans['term']=loans.term.dropna()


# In[20]:


loans.term


# #### Important Information
# 
# Identifying the important fields from the data that can be usefull

# In[21]:


loans.columns


# In[22]:


loans.head(15)


# ##### using of the method describe to help in understanding which of the remaining columns are usefull

# In[23]:


loans.describe()


# In[23]:


cols=loans[['loan_amnt', 'funded_amnt_inv', 'int_rate', 'term','installment','grade','sub_grade','settlement_status','settlement_amount']]
loans2=pd.DataFrame(cols)


# The values that appears shows that columns like loan_amnt,funded_amnt,funded_amnt_inv,term,int_rateminstallment,dti,
# settlement_amount, settlement_percentage can be used to do calcluation and visualisation while others are not as as useful as no main calcuation can happen on it.

# #### Visualization of the important fields/information

# ###### Graph: 1

# In[24]:


loan_grades= loans2.grade.value_counts()
loan_grades


# In[25]:


fig, ax = plt.subplots(figsize=(8,7))
explode = (0.12,0.11,0.10,0.05,0.04,0.03,0.07)
ax.pie(loan_grades, labels = None, autopct='%1.1f%%', startangle=90, shadow = True, explode = explode)
ax.legend(bbox_to_anchor=(1,0.5), labels=loan_grades.index)
plt.suptitle('The Grade Type of Loans')


# ##### Understanding
# <br>
# It can be seen that out of all the grade type of loans presesnt, grade B has the maximum. Grade F and G have minimum of the data for the loan for the interest rate to paid or the loan not being fulfilled.

# ###### Graph: 2

# In[27]:


loans2.groupby(['int_rate']).grade.value_counts().nlargest(20)


# In[28]:


loans2.groupby(['int_rate']).grade.value_counts().nlargest(20).plot(kind = 'bar',title = 'Sub Grade based on Int Rate')


# ##### Understanding
# <br>
# it can be seen that grade B has the maximum loan type where the loan interest type can be change and to be paid. it can be seen interest has an affect by the grade type.

# ###### Graph: 3

# In[29]:


loan_sub_grades= loans2.groupby(['int_rate']).sub_grade.value_counts().nlargest(20)
loan_sub_grades


# In[30]:


loan_sub_grades.plot(kind = 'bar',title = 'Sub Grade based on Int Rate')


# ##### Understanding
# <br>
# It can be seen most of the sub grade is B4 where the it is affected by the interest rate. the interest rate is affected by sub grade to be paid by a person when required. 

# In[78]:


test=loans2.groupby(['grade','sub_grade'],as_index=False).int_rate.mean()
test


# In[80]:


sns.pointplot(x="sub_grade", y="int_rate", hue="grade",data=test, palette="Set3")


# ###### Graph: 4

# In[31]:


settlement_pay=loans2.groupby(['settlement_status','int_rate']).settlement_amount.mean().nlargest(25)
settlement_pay


# In[32]:


settlement_pay.plot.bar()


# ##### Understanding
# <br>
# THis bar grahp shows how the settlement amount is affected by the status of settlement and how much interest rate is affecting the highest settlement amount. Though it can be seen that most of the settlement amount is complete even though some of the interest rate is above 20%.

# ###### Graph: 5

# In[33]:


loans2.corr()


# In[34]:


sns.heatmap(loans2.corr());
plt.suptitle('HeatMap showing Correlation between Informative Fields')


# ##### Understanding
# <br>
# it can be seen that int rate has positive correlation to various fields though by not too much as the range mostly range between 0.2 and 0.4, which is not too much.

# ###### Graph: 6

# In[35]:


loan_int=loans2.groupby('term').int_rate.mean()
loan_int


# In[36]:


loan_int.plot.bar(stacked=True)
plt.suptitle('The Mean of the Interest Rate against Term')


# ##### Understanding
# <br>
# It can be be seen that the mean of interest is more on the as the term increases.

# ###### Graph: 7

# 7(a)

# In[37]:


installments1=loans2[loans2['int_rate'] != 6.03].installment
installments1


# In[38]:


sns.distplot(installments1, bins = 20)
plt.suptitle('Hstogram of Installments bases of Interest Rate !=6.03')


# 7(b)

# In[39]:


installments2=loans2[loans2['int_rate'] != 11.71].installment
installments2


# In[40]:


sns.distplot(installments2, bins = 20)
plt.suptitle('Hstogram of Installments bases of Interest Rate !=11.71')


# ##### Understanding
# <br>
# In the histograms it can be seen that the data is affect not including some of the interst rate though it not be too much. The skewness is towards the left though how wde the displaying to bars may change.

# #### Conclusion
# <br>
# In conclusion, the insterest rate is affected by or has affect on  fields such settlement amount, the grades and sub grades. There are other fields that affected in correlation to interest rate but it may not by too much as see in the heat map.
