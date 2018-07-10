
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[3]:


df.shape


# c. The number of unique users in the dataset.

# In[4]:


number_of_users = df['user_id'].nunique()
print(number_of_users)


# d. The proportion of users converted.

# In[5]:


converted_users = df.query('converted == 1').user_id.nunique()
print(converted_users)

proportion = converted_users / number_of_users
print(proportion)


# e. The number of times the `new_page` and `treatment` don't line up.

# In[6]:


error_treatment = df[(df['group'] == "treatment") & (df['landing_page'] == "old_page")].count()

error_control = df[(df['group'] == "control") & (df['landing_page'] == "new_page")].count()

print(error_treatment)
print(error_control)

#sum is 1965 + 1928 

print (1965 + 1928)


# f. Do any of the rows have missing values?

# In[7]:


df.info()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


df2 = df.drop(df[(df['group'] == "control") & (df['landing_page'] == "new_page")].index)

df2 = df2.drop(df2[(df2['group'] == "treatment") & (df2['landing_page'] == "old_page")].index)



# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[10]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


pd.concat(g for _, g in df2.groupby("user_id") if len(g) > 1)

#Source: https://stackoverflow.com/questions/14657241/how-do-i-get-a-list-of-all-the-duplicate-items-using-pandas-in-python


# c. What is the row information for the repeat **user_id**? 

# In[12]:


#see above


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[13]:


df2 = df2.drop(1899, axis = 0)


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[14]:


total_users = df2['user_id'].nunique()
converted_users = df2[df2['converted'] == 1].count()
conversion_prob = converted_users/total_users
print(conversion_prob)


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[15]:


control_conversion = df2[(df2['group'] == "control") & (df2['converted'] == 1)].count()
total_control = df2[(df2['group'] == "control")].count()
print(control_conversion/total_control)


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[16]:


treatment_conversion = df2[(df2['group'] == "treatment") & (df2['converted'] == 1)].count()
total_treatment = df2[(df2['group'] == "treatment")].count()
print(treatment_conversion/total_treatment)


# d. What is the probability that an individual received the new page?

# In[17]:


new_page_prob = total_treatment / (total_control + total_treatment)
print(new_page_prob)


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# It appears just by looking at the numbers in the first section that there is no difference between the control and treatment (new page and old) in terms of conversion rate, however, it is not possible to determine if there is a difference between these rates without performing statistical analyses. There may not appear to be a difference, but this difference could be significant. 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# Null Hypothesis: There is no difference in conversion rate between the old and new web pages. 
# Alternative Hypothesis: The new web page leads to a higher conversion rate when compared to the old web page. 

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[18]:


#This is the same as that calculated above, where the conversion
#rate was calculated regardless of page
total_users = df2['user_id'].nunique()
converted_users = df2[df2['converted'] == 1].count()
conversion_prob = converted_users/total_users
print(conversion_prob)


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[19]:


#This is the same as that calculated above, where the conversion
#rate was calculated regardless of page (p old = p new)
total_users = df2['user_id'].nunique()
converted_users = df2[df2['converted'] == 1].count()
conversion_prob = converted_users/total_users
print(conversion_prob)


# c. What is $n_{new}$?

# In[20]:


#n new is equal to the total number in csv
total_treatment = df2[(df2['landing_page'] == "new_page")].count()
print(total_treatment)


# d. What is $n_{old}$?

# In[21]:


#n old is equal to the total number in csv
total_control = df2[(df2['landing_page'] == "old_page")].count()
print(total_control)


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[22]:


#n = 145310
new_page_converted = np.random.choice([0,1], size = (145310), p = [0.88, 0.12]) 
p_new = (new_page_converted == 1).mean()


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[23]:


#n = 145274
old_page_converted = np.random.choice([0,1], size = (145274), p = [0.88, 0.12]) 
p_old = (old_page_converted == 1).mean()




# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[24]:


print(((new_page_converted == 1).mean()) - ((old_page_converted == 1).mean()))
print(p_new - p_old)


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[25]:



p_diffs = []
for i in range(10000):
    new_page_converted = np.random.choice([0,1], size = (145310), p = [0.88, 0.12]) 
    p_new = (new_page_converted == 1).mean()
    old_page_converted = np.random.choice([0,1], size = (145310), p = [0.88, 0.12]) 
    p_old = (old_page_converted == 1).mean()
    p_diffs.append(p_new - p_old)
    


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[26]:


plt.hist(p_diffs);


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[27]:


p_diffs = np.array(p_diffs)
#Difference observed in ab_data.csv taken from the above sections, 
# 0.118808 (new) - 0.120386 (old)
actual_difference = 0.118808 - 0.120386
print((p_diffs > actual_difference).mean())
plt.axvline(x=actual_difference, color = 'red')
plt.hist(p_diffs);


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?
# 

# Answer: 90.23% of the values in p_diffs are greater than the actual difference observed in ab_data.csv. 
# We just computed the p value, which is 0.9023. 
# This means that we cannot reject the null hypothesis. In other words, there is a 90% chance that the null hypothesis is true. Rejecting the null hypothesis would accept a 90.23% error rate, which is not acceptable. 

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[28]:


import statsmodels.api as sm

convert_old = df2[(df2['group'] == "treatment") & (df2['converted'] == 1)].count()
convert_new = df2[(df2['group'] == "control") & (df2['converted'] == 1)].count()
n_old = df2[(df2['landing_page'] == "old_page")].count()
n_new = df2[(df2['landing_page'] == "new_page")].count()
print(convert_old, convert_new, n_old, n_new)


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[29]:


import statsmodels.api as sm
z_score, p_value = sm.stats.proportions_ztest([17489, 17264], [145274, 145310], alternative = 'smaller')

print(z_score, p_value)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# Answer: The z-score and the p value mean that we fail to reject the null hypothesis. This is because there's a 90.5% chance (or 0.905 probablility) that our parameter is within the null hypothesis. An acceptable probability for rejecting the null hypothesis is 0.05 or less. 
# In summary, there is no significant difference in conversion rates between the old and new pages. 
# These findings agree with parts j and k, the p values obtained were both 0.9. 

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# Answer: Logistic Regression

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[30]:


df2[['test', 'ab_page']] = pd.get_dummies(df2['group'])
df2['intercept'] = 1
df2 = df2.drop(['test'], axis = 1)
df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[31]:


from sklearn.linear_model import LogisticRegression
#df2['intercept'] = 1
logit_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = logit_mod.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[32]:


results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# The p value associated with ab_page is 0.190. This is different than the value obtained in Part II. The null hypothesis associated with the regression model is that the slope is equal to zero. The alternative is that the slope is not equal to zero. In part two we were comparing conversion rates between the old and new pages, while in this linear regression model, the p value of 0.190 is referring to the null hypothesis around the slope being equal to zero. 

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# Answer: There are good reasons for adding variables, since there are many factors that may influence a change in a variable. Adding more variables can provide more information. 
# The disadvantage when we add variables is that the  correlation coefficient may increase, and the probability of each decreases (thereby providing a false positive to reject the null hypothesis). Also, adding more variables may introduce more errors such as multicollinearity. 

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.

# In[33]:


countries_df = pd.read_csv('./countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new.head()
df_new['country'].unique()


# In[34]:


### Create the necessary dummy variables
df_new[['UK', 'US', 'CA']] = pd.get_dummies(df_new['country'])
df_new = df_new.drop('US', axis = 1)
df_new.head()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[35]:


### Fit Your Linear Model And Obtain the Results
df_new['intercept'] = 1
logit_mod = sm.Logit(df_new['converted'], df_new[['intercept', 'ab_page', 'UK', 'CA']])
results = logit_mod.fit()
results.summary()


# In[76]:


print (np.exp(-0.0099))
print (np.exp(-0.0506))


# <a id='conclusions'></a>
# ## Conclusions
# 
# This model suggests that country also has an impact when all other variables are held constant. That is, if the US is considered baseline, and we exponentiate the coef for CA (-0.0099), we see that there's a 0.99 chance of successful conversion when compared to the US. For the UK, there's a 0.95 chance of successful conversion when compared to the US. 
# 
# ### Gather Submission Materials
# 
# Once you are satisfied with the status of your Notebook, you should save it in a format that will make it easy for others to read. You can use the __File -> Download as -> HTML (.html)__ menu to save your notebook as an .html file. If you are working locally and get an error about "No module name", then open a terminal and try installing the missing module using `pip install <module_name>` (don't include the "<" or ">" or any words following a period in the module name).
# 
# You will submit both your original Notebook and an HTML or PDF copy of the Notebook for review. There is no need for you to include any data files with your submission. If you made reference to other websites, books, and other resources to help you in solving tasks in the project, make sure that you document them. It is recommended that you either add a "Resources" section in a Markdown cell at the end of the Notebook report, or you can include a `readme.txt` file documenting your sources.
# 
# ### Submit the Project
# 
# When you're ready, click on the "Submit Project" button to go to the project submission page. You can submit your files as a .zip archive or you can link to a GitHub repository containing your project files. If you go with GitHub, note that your submission will be a snapshot of the linked repository at time of submission. It is recommended that you keep each project in a separate repository to avoid any potential confusion: if a reviewer gets multiple folders representing multiple projects, there might be confusion regarding what project is to be evaluated.
# 
# It can take us up to a week to grade the project, but in most cases it is much faster. You will get an email once your submission has been reviewed. If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com. In the meantime, you should feel free to continue on with your learning journey by continuing on to the next module in the program.
