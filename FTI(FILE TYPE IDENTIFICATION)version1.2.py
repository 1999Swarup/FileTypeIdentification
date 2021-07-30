#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from glob import glob  
import os
from collections import Counter
import pickle 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score



# ## PreProcessing

# #### Building Data Frame [ 1: kotlin files]

# In[2]:


# set the path to your file location
path=r'C:\Users\This PC\Others\blueOptima\kt'
# create a empty list, where you store the content
list_of_text = []
# create a empty list, where you store the content like values:frequency
freq_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
freq=[]
# create a empty list, where you store the byte values
val=[]
# create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
prob_freq_dist_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
prob_freq=[]
# create a empty list, where you store the byte values
prob_val=[]

"""
Created a helper method, that uses a Counter object and the length of the file contents, 
to adjust every byte value count by the number of bytes in the file. 
Inside of the helper method there is a generator function that loops over a Counter instance and applies the calculation.
As you will see we need to wrap the generator inside of a dict, so that the Counter does not count the frequency 
of value-frequency tuples, which are all going to be unique, 
but instead it will apply a dict to itself and allow us to use all the additional functionality that the Counter class offers.
"""

def probability_distribution(content):
    def _helper():
        absolute_distribution = Counter(content)
        length = len(content)
        for value, frequency in absolute_distribution.items():
            yield int(value), float(frequency) / length
    return Counter(dict(_helper()))

# loop over the files in the folder print("0x{:02x}: {}".format(value, frequency),len(val),"\n",text)
for file in os.listdir(path):
    # create a empty list, where you store the content like values:frequency
    freq_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    freq=[]
    # create a empty list, where you store the byte values
    val=[]
    # create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
    prob_freq_dist_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    prob_freq=[]
    # create a empty list, where you store the byte values
    prob_val=[]
    # open the file
    with open(os.path.join(path, file),"rb") as f:
        text = f.read()
        c = Counter(text)
        prob_dist_c=probability_distribution(text)
    #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
    for prob_value, prob_dist_freq in prob_dist_c.most_common(n=10):
        prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
        prob_val.append("{:02x}".format(prob_value))
        prob_freq.append("{:.0%}".format(prob_dist_freq))
    #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
    for value, frequency in c.most_common(n=10):
        freq_value.append("0x{:02x}: {}".format(value,frequency))
        val.append("{:02x}".format(value))
        freq.append("{}".format(frequency))
    list_of_text.append((c,val,freq,freq_value,prob_val,prob_freq,prob_freq_dist_value,text,file))
df_kt = pd.DataFrame(list_of_text, columns = ['OccurrenceOfByteContents','OnlyValueForAbsoluteDistribution','OnlyFrequency','Value:Frequency','OnlyValueForProbalilityDistribution','OnlyProbalilityDistribution','Value:ProbalilityDistribution','Text', 'Filename'])


print(df_kt.info())
print("*" * 80)
print(df_kt.memory_usage())
print("*" * 80)
df_kt.head(2)


# #### Building Data Frame [ 2: mak files]

# In[3]:


# set the path to your file location
path=r'C:\Users\This PC\Others\blueOptima\mak'
# create a empty list, where you store the content
list_of_text = []
# create a empty list, where you store the content like values:frequency
freq_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
freq=[]
# create a empty list, where you store the byte values
val=[]
# create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
prob_freq_dist_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
prob_freq=[]
# create a empty list, where you store the byte values
prob_val=[]

"""
Created a helper method, that uses a Counter object and the length of the file contents, 
to adjust every byte value count by the number of bytes in the file. 
Inside of the helper method there is a generator function that loops over a Counter instance and applies the calculation.
As you will see we need to wrap the generator inside of a dict, so that the Counter does not count the frequency 
of value-frequency tuples, which are all going to be unique, 
but instead it will apply a dict to itself and allow us to use all the additional functionality that the Counter class offers.
"""

def probability_distribution(content):
    def _helper():
        absolute_distribution = Counter(content)
        length = len(content)
        for value, frequency in absolute_distribution.items():
            yield int(value), float(frequency) / length
    return Counter(dict(_helper()))

# loop over the files in the folder
for file in os.listdir(path):
    # create a empty list, where you store the content like values:frequency
    freq_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    freq=[]
    # create a empty list, where you store the byte values
    val=[]
    # create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
    prob_freq_dist_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    prob_freq=[]
    # create a empty list, where you store the byte values
    prob_val=[]
    # open the file
    with open(os.path.join(path, file),"rb") as f:
        text = f.read()
        c = Counter(text)
        prob_dist_c=probability_distribution(text)
    #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
    for prob_value, prob_dist_freq in prob_dist_c.most_common(n=10):
        prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
        prob_val.append("{:02x}".format(prob_value))
        prob_freq.append("{:.0%}".format(prob_dist_freq))
    #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
    for value, frequency in c.most_common(n=10):
        freq_value.append("0x{:02x}: {}".format(value,frequency))
        val.append("{:02x}".format(value))
        freq.append("{}".format(frequency))
    # append the text and filename
    list_of_text.append((c,freq_value,val,freq,prob_freq_dist_value,prob_val,prob_freq,text, file))
# create a dataframe and save
df_mak = pd.DataFrame(list_of_text, columns = ['OccurrenceOfByteContents','Value:Frequency','OnlyValueForAbsoluteDistribution','OnlyFrequency','Value:ProbalilityDistribution','OnlyValueForProbalilityDistribution','OnlyProbalilityDistribution','Text', 'Filename'])


print(df_mak.info())
print("*" * 80)
print(df_mak.memory_usage())
print("*" * 80)
df_mak.head(2)


# #### Building Data Frame [ 3: ml files]

# In[4]:


# set the path to your file location
path=r'C:\Users\This PC\Others\blueOptima\ml'
# create a empty list, where you store the content
list_of_text = []
# create a empty list, where you store the content like values:frequency
freq_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
freq=[]
# create a empty list, where you store the byte values
val=[]
# create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
prob_freq_dist_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
prob_freq=[]
# create a empty list, where you store the byte values
prob_val=[]

"""
Created a helper method, that uses a Counter object and the length of the file contents, 
to adjust every byte value count by the number of bytes in the file. 
Inside of the helper method there is a generator function that loops over a Counter instance and applies the calculation.
As you will see we need to wrap the generator inside of a dict, so that the Counter does not count the frequency 
of value-frequency tuples, which are all going to be unique, 
but instead it will apply a dict to itself and allow us to use all the additional functionality that the Counter class offers.
"""

def probability_distribution(content):
    def _helper():
        absolute_distribution = Counter(content)
        length = len(content)
        for value, frequency in absolute_distribution.items():
            yield int(value), float(frequency) / length
    return Counter(dict(_helper()))

# loop over the files in the folder
for file in os.listdir(path):
    # create a empty list, where you store the content like values:frequency
    freq_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    freq=[]
    # create a empty list, where you store the byte values
    val=[]
    # create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
    prob_freq_dist_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    prob_freq=[]
    # create a empty list, where you store the byte values
    prob_val=[]
    # open the file
    with open(os.path.join(path, file),"rb") as f:
        text = f.read()
        c = Counter(text)
        prob_dist_c=probability_distribution(text)
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for prob_value, prob_dist_freq in prob_dist_c.most_common(n=10):
            prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
            prob_val.append("{:02x}".format(prob_value))
            prob_freq.append("{:.0%}".format(prob_dist_freq))
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for value, frequency in c.most_common(n=10):
            freq_value.append("0x{:02x}: {}".format(value,frequency))
            val.append("{:02x}".format(value))
            freq.append("{}".format(frequency))
    # append the text and filename
    list_of_text.append((c,freq_value,val,freq,prob_freq_dist_value,prob_val,prob_freq,text, file))
# create a dataframe and save
df_ml = pd.DataFrame(list_of_text, columns = ['OccurrenceOfByteContents','Value:Frequency','OnlyValueForAbsoluteDistribution','OnlyFrequency','Value:ProbalilityDistribution','OnlyValueForProbalilityDistribution','OnlyProbalilityDistribution','Text', 'Filename'])


print(df_ml.info())
print("*" * 80)
print(df_ml.memory_usage())
print("*" * 80)
df_ml.head(2)


# #### Building Data Frame [ 4: rexx files]

# In[5]:


# set the path to your file location
path=r'C:\Users\This PC\Others\blueOptima\rexx'
# create a empty list, where you store the content
list_of_text = []
# create a empty list, where you store the content like values:frequency
freq_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
freq=[]
# create a empty list, where you store the byte values
val=[]
# create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
prob_freq_dist_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
prob_freq=[]
# create a empty list, where you store the byte values
prob_val=[]

"""
Created a helper method, that uses a Counter object and the length of the file contents, 
to adjust every byte value count by the number of bytes in the file. 
Inside of the helper method there is a generator function that loops over a Counter instance and applies the calculation.
As you will see we need to wrap the generator inside of a dict, so that the Counter does not count the frequency 
of value-frequency tuples, which are all going to be unique, 
but instead it will apply a dict to itself and allow us to use all the additional functionality that the Counter class offers.
"""

def probability_distribution(content):
    def _helper():
        absolute_distribution = Counter(content)
        length = len(content)
        for value, frequency in absolute_distribution.items():
            yield int(value), float(frequency) / length
    return Counter(dict(_helper()))

# loop over the files in the folder
for file in os.listdir(path):
    # create a empty list, where you store the content like values:frequency
    freq_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    freq=[]
    # create a empty list, where you store the byte values
    val=[]
    # create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
    prob_freq_dist_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    prob_freq=[]
    # create a empty list, where you store the byte values
    prob_val=[]
    # open the file
    with open(os.path.join(path, file),"rb") as f:
        text = f.read()
        c = Counter(text)
        prob_dist_c=probability_distribution(text)
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for prob_value, prob_dist_freq in prob_dist_c.most_common(n=10):
            prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
            prob_val.append("{:02x}".format(prob_value))
            prob_freq.append("{:.0%}".format(prob_dist_freq))
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for value, frequency in c.most_common(n=10):
            freq_value.append("0x{:02x}: {}".format(value,frequency))
            val.append("{:02x}".format(value))
            freq.append("{}".format(frequency))
    # append the text and filename
    list_of_text.append((c,freq_value,val,freq,prob_freq_dist_value,prob_val,prob_freq,text, file))
# create a dataframe and save
df_rexx = pd.DataFrame(list_of_text, columns = ['OccurrenceOfByteContents','Value:Frequency','OnlyValueForAbsoluteDistribution','OnlyFrequency','Value:ProbalilityDistribution','OnlyValueForProbalilityDistribution','OnlyProbalilityDistribution','Text', 'Filename'])

print(df_rexx.info())
print("*" * 80)
print(df_rexx.memory_usage())
print("*" * 80)
df_rexx.head(2)


# #### Building Data Frame [ 5: csproj files]

# In[6]:


# set the path to your file location
path=r'C:\Users\This PC\Others\blueOptima\csproj'
# create a empty list, where you store the content
list_of_text = []
# create a empty list, where you store the content like values:frequency
freq_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
freq=[]
# create a empty list, where you store the byte values
val=[]
# create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
prob_freq_dist_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
prob_freq=[]
# create a empty list, where you store the byte values
prob_val=[]

"""
Created a helper method, that uses a Counter object and the length of the file contents, 
to adjust every byte value count by the number of bytes in the file. 
Inside of the helper method there is a generator function that loops over a Counter instance and applies the calculation.
As you will see we need to wrap the generator inside of a dict, so that the Counter does not count the frequency 
of value-frequency tuples, which are all going to be unique, 
but instead it will apply a dict to itself and allow us to use all the additional functionality that the Counter class offers.
"""

def probability_distribution(content):
    def _helper():
        absolute_distribution = Counter(content)
        length = len(content)
        for value, frequency in absolute_distribution.items():
            yield int(value), float(frequency) / length
    return Counter(dict(_helper()))

# loop over the files in the folder
for file in os.listdir(path):
    # create a empty list, where you store the content like values:frequency
    freq_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    freq=[]
    # create a empty list, where you store the byte values
    val=[]
    # create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
    prob_freq_dist_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    prob_freq=[]
    # create a empty list, where you store the byte values
    prob_val=[]
    # open the file
    with open(os.path.join(path, file),"rb") as f:
        text = f.read()
        c = Counter(text)
        prob_dist_c=probability_distribution(text)
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for prob_value, prob_dist_freq in prob_dist_c.most_common(n=10):
            prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
            prob_val.append("{:02x}".format(prob_value))
            prob_freq.append("{:.0%}".format(prob_dist_freq))
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for value, frequency in c.most_common(n=10):
            freq_value.append("0x{:02x}: {}".format(value,frequency))
            val.append("{:02x}".format(value))
            freq.append("{}".format(frequency))
    # append the text and filename
    list_of_text.append((c,freq_value,val,freq,prob_freq_dist_value,prob_val,prob_freq,text, file))

# create a dataframe and save
df_csproj = pd.DataFrame(list_of_text, columns = ['OccurrenceOfByteContents','Value:Frequency','OnlyValueForAbsoluteDistribution','OnlyFrequency','Value:ProbalilityDistribution','OnlyValueForProbalilityDistribution','OnlyProbalilityDistribution','Text', 'Filename'])

print(df_csproj.info())
print("*" * 80)
print(df_csproj.memory_usage())
print("*" * 80)
df_csproj.head(2)


# #### Building Data Frame [ 6: jenkinsfile files]

# In[7]:


# set the path to your file location
path=r'C:\Users\This PC\Others\blueOptima\jenkinsfile'
# create a empty list, where you store the content
list_of_text = []
# create a empty list, where you store the content like values:frequency
freq_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
freq=[]
# create a empty list, where you store the byte values
val=[]
# create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
prob_freq_dist_value=[]
# create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
prob_freq=[]
# create a empty list, where you store the byte values
prob_val=[]

"""
Created a helper method, that uses a Counter object and the length of the file contents, 
to adjust every byte value count by the number of bytes in the file. 
Inside of the helper method there is a generator function that loops over a Counter instance and applies the calculation.
As you will see we need to wrap the generator inside of a dict, so that the Counter does not count the frequency 
of value-frequency tuples, which are all going to be unique, 
but instead it will apply a dict to itself and allow us to use all the additional functionality that the Counter class offers.
"""

def probability_distribution(content):
    def _helper():
        absolute_distribution = Counter(content)
        length = len(content)
        for value, frequency in absolute_distribution.items():
            yield int(value), float(frequency) / length
    return Counter(dict(_helper()))

# loop over the files in the folder
for file in os.listdir(path):
    # create a empty list, where you store the content like values:frequency
    freq_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    freq=[]
    # create a empty list, where you store the byte values
    val=[]
    # create a empty list, where you store the content like values:probability distribution - probalility frequency distribution value
    prob_freq_dist_value=[]
    # create a empty list, where you store the frequency of absolute distribution i.e., how many times a certain byte value occurs
    prob_freq=[]
    # create a empty list, where you store the byte values
    prob_val=[]
    # open the file
    with open(os.path.join(path, file),"rb") as f:
        text = f.read()
        c = Counter(text)
        prob_dist_c=probability_distribution(text)
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for prob_value, prob_dist_freq in prob_dist_c.most_common(n=10):
            prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
            prob_val.append("{:02x}".format(prob_value))
            prob_freq.append("{:.0%}".format(prob_dist_freq))
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for value, frequency in c.most_common(n=10):
            freq_value.append("0x{:02x}: {}".format(value,frequency))
            val.append("{:02x}".format(value))
            freq.append("{}".format(frequency))
    # append the text and filename
    list_of_text.append((c,freq_value,val,freq,prob_freq_dist_value,prob_val,prob_freq,text, file))

    
# create a dataframe and save
df_jenkinsfile = pd.DataFrame(list_of_text, columns = ['OccurrenceOfByteContents','Value:Frequency','OnlyValueForAbsoluteDistribution','OnlyFrequency','Value:ProbalilityDistribution','OnlyValueForProbalilityDistribution','OnlyProbalilityDistribution','Text', 'Filename'])

print(df_jenkinsfile.info())
print("*" * 80)
print(df_jenkinsfile.memory_usage())
print("*" * 80)
df_jenkinsfile.head(2)


# ### Feature Scaling & some more preprocessing

# In[8]:


##### For each DataFrame
final_df_jenkinsfile=df_jenkinsfile[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','Filename']].copy()
final_df_csproj=df_csproj[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','Filename']].copy()
final_df_rexx=df_rexx[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','Filename']].copy()
final_df_ml=df_ml[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','Filename']].copy()
final_df_mak=df_mak[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','Filename']].copy()
final_df_kt=df_kt[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','Filename']].copy()


# In[9]:


final_df_csproj.head()


# In[10]:


# reset the index
final_df_jenkinsfile.reset_index(drop=True)


# In[11]:


final_df_csproj.reset_index(drop=True)


# In[12]:


final_df_rexx.reset_index(drop=True)


# In[13]:


final_df_ml.reset_index(drop=True)


# In[14]:


final_df_mak.reset_index(drop=True)


# In[15]:


final_df_kt.reset_index(drop=True)


# In[16]:


#As all columns have the same number of lists, you can call Series.explode on each column
final_df_jenkinsfile=final_df_jenkinsfile.set_index(['Filename']).apply(pd.Series.explode).reset_index()
final_df_csproj=final_df_csproj.set_index(['Filename']).apply(pd.Series.explode).reset_index()
final_df_rexx=final_df_rexx.set_index(['Filename']).apply(pd.Series.explode).reset_index()
final_df_ml=final_df_ml.set_index(['Filename']).apply(pd.Series.explode).reset_index()
final_df_mak=final_df_mak.set_index(['Filename']).apply(pd.Series.explode).reset_index()
final_df_kt=final_df_kt.set_index(['Filename']).apply(pd.Series.explode).reset_index()


# In[17]:


final_df_kt.reset_index(drop=True)


# In[18]:


# getting the target for my model from the filename
final_df_jenkinsfile['ClassName'] = final_df_jenkinsfile['Filename'].str.split('.').str[-1]
final_df_csproj['ClassName'] = final_df_csproj['Filename'].str.split('.').str[-1]
final_df_rexx['ClassName'] = final_df_rexx['Filename'].str.split('.').str[-1]
final_df_ml['ClassName'] = final_df_ml['Filename'].str.split('.').str[-1]
final_df_mak['ClassName'] = final_df_mak['Filename'].str.split('.').str[-1]
final_df_kt['ClassName'] = final_df_kt['Filename'].str.split('.').str[-1]


# In[19]:


final_df_kt.reset_index(drop=True)


# In[20]:


# removing the % sign present near data in the column probabilistic percentage
final_df_jenkinsfile=final_df_jenkinsfile.replace('\%','',regex=True)
final_df_csproj=final_df_csproj.replace('\%','',regex=True)
final_df_rexx=final_df_rexx.replace('\%','',regex=True)
final_df_ml=final_df_ml.replace('\%','',regex=True)
final_df_mak=final_df_mak.replace('\%','',regex=True)
final_df_kt=final_df_kt.replace('\%','',regex=True)


# In[21]:


final_df_kt.head()


# In[22]:


##### For each DataFrame
fdf_jenkinsfile=final_df_jenkinsfile[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','ClassName']].copy()
fdf_csproj=final_df_csproj[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','ClassName']].copy()
fdf_rexx=final_df_rexx[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','ClassName']].copy()
fdf_ml=final_df_ml[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','ClassName']].copy()
fdf_mak=final_df_mak[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','ClassName']].copy()
fdf_kt=final_df_kt[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','ClassName']].copy()


# In[23]:


fdf_kt.head()


# In[24]:


#replacing all infinite values with nan and than nan with 0
fdf_jenkinsfile.replace([np.inf, -np.inf], np.nan, inplace=True)
fdf_jenkinsfile['OnlyFrequency'].fillna(fdf_jenkinsfile['OnlyFrequency'].mode()[0], inplace=True)
fdf_jenkinsfile['OnlyProbalilityDistribution'].fillna(fdf_jenkinsfile['OnlyProbalilityDistribution'].mode()[0], inplace=True)
fdf_jenkinsfile.fillna(0, inplace=True)

fdf_csproj.replace([np.inf, -np.inf], np.nan, inplace=True)
fdf_csproj['OnlyFrequency'].fillna(fdf_csproj['OnlyFrequency'].mode()[0], inplace=True)
fdf_csproj['OnlyProbalilityDistribution'].fillna(fdf_csproj['OnlyProbalilityDistribution'].mode()[0], inplace=True)
fdf_csproj.fillna(0, inplace=True)

fdf_rexx.replace([np.inf, -np.inf], np.nan, inplace=True)
fdf_rexx['OnlyFrequency'].fillna(fdf_rexx['OnlyFrequency'].mode()[0], inplace=True)
fdf_rexx['OnlyProbalilityDistribution'].fillna(fdf_rexx['OnlyProbalilityDistribution'].mode()[0], inplace=True)
fdf_rexx.fillna(0, inplace=True)

fdf_ml.replace([np.inf, -np.inf], np.nan, inplace=True)
fdf_ml['OnlyFrequency'].fillna(fdf_ml['OnlyFrequency'].mode()[0], inplace=True)
fdf_ml['OnlyProbalilityDistribution'].fillna(fdf_ml['OnlyProbalilityDistribution'].mode()[0], inplace=True)
fdf_ml.fillna(0, inplace=True)

fdf_mak.replace([np.inf, -np.inf], np.nan, inplace=True)
fdf_mak['OnlyFrequency'].fillna(fdf_mak['OnlyFrequency'].mode()[0], inplace=True)
fdf_mak['OnlyProbalilityDistribution'].fillna(fdf_mak['OnlyProbalilityDistribution'].mode()[0], inplace=True)
fdf_mak.fillna(0, inplace=True)

fdf_kt.replace([np.inf, -np.inf], np.nan, inplace=True)
fdf_kt['OnlyFrequency'].fillna(fdf_kt['OnlyFrequency'].mode()[0], inplace=True)
fdf_kt['OnlyProbalilityDistribution'].fillna(fdf_kt['OnlyProbalilityDistribution'].mode()[0], inplace=True)
fdf_kt.fillna(0, inplace=True)


# In[25]:


fdf_kt.head()


# In[26]:


fdf_kt.head()


# In[27]:


#converting all non-numeric data in column OnlyValueForAbsoluteDistribution to numeric type


# In[28]:


fdf_jenkinsfile = fdf_jenkinsfile[pd.to_numeric(fdf_jenkinsfile['OnlyValueForAbsoluteDistribution'], errors='coerce').notnull()]
fdf_jenkinsfile['OnlyValueForAbsoluteDistribution'] = fdf_jenkinsfile['OnlyValueForAbsoluteDistribution'].astype(int)


# In[29]:


fdf_csproj = fdf_csproj[pd.to_numeric(fdf_csproj['OnlyValueForAbsoluteDistribution'], errors='coerce').notnull()]
fdf_csproj['OnlyValueForAbsoluteDistribution'] = fdf_csproj['OnlyValueForAbsoluteDistribution'].astype(int)


# In[30]:


fdf_rexx = fdf_rexx[pd.to_numeric(fdf_rexx['OnlyValueForAbsoluteDistribution'], errors='coerce').notnull()]
fdf_rexx['OnlyValueForAbsoluteDistribution'] = fdf_rexx['OnlyValueForAbsoluteDistribution'].astype(int)


# In[31]:


fdf_ml = fdf_ml[pd.to_numeric(fdf_ml['OnlyValueForAbsoluteDistribution'], errors='coerce').notnull()]
fdf_ml['OnlyValueForAbsoluteDistribution'] = fdf_ml['OnlyValueForAbsoluteDistribution'].astype(int)


# In[32]:


fdf_kt = fdf_kt[pd.to_numeric(fdf_kt['OnlyValueForAbsoluteDistribution'], errors='coerce').notnull()]
fdf_kt['OnlyValueForAbsoluteDistribution'] = fdf_kt['OnlyValueForAbsoluteDistribution'].astype(int)


# In[33]:


fdf_mak = fdf_mak[pd.to_numeric(fdf_mak['OnlyValueForAbsoluteDistribution'], errors='coerce').notnull()]
fdf_mak['OnlyValueForAbsoluteDistribution'] = fdf_mak['OnlyValueForAbsoluteDistribution'].astype(int)


# In[34]:


#changing data types of columns for each file type
fdf_jenkinsfile['OnlyValueForAbsoluteDistribution']=fdf_jenkinsfile.OnlyValueForAbsoluteDistribution.astype(int)
fdf_jenkinsfile['OnlyFrequency']=fdf_jenkinsfile.OnlyFrequency.astype(int)
fdf_jenkinsfile['OnlyProbalilityDistribution']=fdf_jenkinsfile.OnlyProbalilityDistribution.astype(int)

fdf_csproj['OnlyValueForAbsoluteDistribution']=fdf_csproj.OnlyValueForAbsoluteDistribution.astype(int)
fdf_csproj['OnlyFrequency']=fdf_csproj.OnlyFrequency.astype(int)
fdf_csproj['OnlyProbalilityDistribution']=fdf_csproj.OnlyProbalilityDistribution.astype(int)

fdf_rexx['OnlyValueForAbsoluteDistribution']=fdf_rexx.OnlyValueForAbsoluteDistribution.astype(int)
fdf_rexx['OnlyFrequency']=fdf_rexx.OnlyFrequency.astype(int)
fdf_rexx['OnlyProbalilityDistribution']=fdf_rexx.OnlyProbalilityDistribution.astype(int)

fdf_ml['OnlyValueForAbsoluteDistribution']=fdf_ml.OnlyValueForAbsoluteDistribution.astype(int)
fdf_ml['OnlyFrequency']=fdf_ml.OnlyFrequency.astype(int)
fdf_ml['OnlyProbalilityDistribution']=fdf_ml.OnlyProbalilityDistribution.astype(int)

fdf_mak['OnlyValueForAbsoluteDistribution']=fdf_mak.OnlyValueForAbsoluteDistribution.astype(int)
fdf_mak['OnlyFrequency']=fdf_mak.OnlyFrequency.astype(int)
fdf_mak['OnlyProbalilityDistribution']=fdf_mak.OnlyProbalilityDistribution.astype(int)

fdf_kt['OnlyValueForAbsoluteDistribution']=fdf_kt.OnlyValueForAbsoluteDistribution.astype(int)
fdf_kt['OnlyFrequency']=fdf_kt.OnlyFrequency.astype(int)
fdf_kt['OnlyProbalilityDistribution']=fdf_kt.OnlyProbalilityDistribution.astype(int)


# ##  Splitting above data frames to form train, validation and test data frame

# In[35]:


print((fdf_kt.shape,fdf_mak.shape,fdf_ml.shape,fdf_rexx.shape,fdf_csproj.shape,fdf_jenkinsfile.shape))


# ##### Train and Test dataframe for kt files

# In[36]:


df_kt_train, df_kt_test = train_test_split(fdf_kt, test_size=0.3)
df_kt_train.reset_index(drop=True).head(2)


# In[37]:


df_kt_test.reset_index(drop=True).head(2)


# ##### Train and Test dataframe for mak files

# In[38]:


df_mak_train, df_mak_test = train_test_split(fdf_mak, test_size=0.3)
df_mak_train.reset_index(drop=True).head(2)


# In[39]:


df_mak_test.reset_index(drop=True).head(2)


# ##### Train and Test dataframe for ml files

# In[40]:


df_ml_train, df_ml_test = train_test_split(fdf_ml, test_size=0.3)
df_ml_train.reset_index(drop=True).head(2)


# In[41]:


df_ml_test.reset_index(drop=True).head(2)


# ##### Train and Test dataframe for rexx files

# In[42]:


df_rexx_train, df_rexx_test = train_test_split(fdf_rexx, test_size=0.3)
df_rexx_train.reset_index(drop=True).head(2)


# In[43]:


df_rexx_test.reset_index(drop=True).head(2)


# ##### Train and Test dataframe for csproj files

# In[44]:


df_csproj_train, df_csproj_test = train_test_split(fdf_csproj, test_size=0.3)
df_csproj_train.reset_index(drop=True).head(2)


# In[45]:


df_csproj_test.reset_index(drop=True).head(2)


# ##### Train and Test dataframe for jenkinsfile files

# In[46]:


df_jenkinsfile_train, df_jenkinsfile_test = train_test_split(fdf_jenkinsfile, test_size=0.2)
df_jenkinsfile_train.reset_index(drop=True).head(2)


# In[47]:


df_jenkinsfile_test.reset_index(drop=True).head(2)

# In[48]:


print((df_kt_train.shape,df_mak_train.shape,df_ml_train.shape,df_rexx_train.shape,df_csproj_train.shape,df_jenkinsfile_train.shape))


# In[49]:


print((df_kt_test.shape,df_mak_test.shape,df_ml_test.shape,df_rexx_test.shape,df_csproj_test.shape,df_jenkinsfile_test.shape))


# ### Building the train data frame

# In[50]:


frames = [df_jenkinsfile_train,df_csproj_train,df_rexx_train,df_ml_train,df_mak_train,df_kt_train]

train_df = pd.concat(frames)
print(train_df.shape)
train_df.reset_index(drop=True).head(2)


# In[51]:


train_df.reset_index(drop=True).tail(2)


# ### Validation data frame and Test data frame

# ##### Validation and Test dataframe for jenkinsfile files

# In[52]:


df_jenkinsfile_val, df_jenkinsfile_test = train_test_split(df_jenkinsfile_test, test_size=0.1)
df_jenkinsfile_val.reset_index(drop=True).head(2)


# In[53]:


df_jenkinsfile_test.reset_index(drop=True).head(2)


# ##### Validation and Test dataframe for csproj files

# In[54]:


df_csproj_val, df_csproj_test = train_test_split(df_csproj_test, test_size=0.1)
df_csproj_val.reset_index(drop=True).head(2)


# In[55]:


df_csproj_test.reset_index(drop=True).head(2)


# ##### Validation and Test dataframe for rexx files

# In[56]:


df_rexx_val, df_rexx_test = train_test_split(df_rexx_test, test_size=0.1)
df_rexx_val.reset_index(drop=True).head(2)


# In[57]:


df_rexx_test.reset_index(drop=True).head(2)


# ##### Validation and Test dataframe for ml files

# In[58]:


df_ml_val, df_ml_test = train_test_split(df_ml_test, test_size=0.1)
df_ml_val.reset_index(drop=True).head(2)


# In[59]:


df_ml_test.reset_index(drop=True).head(2)


# ##### Validation and Test dataframe for mak files

# In[60]:


df_mak_val, df_mak_test = train_test_split(df_mak_test, test_size=0.1)
df_mak_val.reset_index(drop=True).head(2)


# In[61]:


df_mak_test.reset_index(drop=True).head(2)


# ##### Validation and Test dataframe for kt files

# In[62]:


df_kt_val, df_kt_test = train_test_split(df_kt_test, test_size=0.1)
df_kt_val.reset_index(drop=True).head(2)


# In[63]:


df_kt_test.reset_index(drop=True).head(2)


# ### Building Validation data frame and Test data frame

# In[64]:


frames = [df_jenkinsfile_val,df_csproj_val,df_rexx_val,df_ml_val,df_mak_val,df_kt_val]

val_df = pd.concat(frames)
print(val_df.shape)
val_df.reset_index(drop=True).head(2)


# In[65]:


frames = [df_jenkinsfile_test,df_csproj_test,df_rexx_test,df_ml_test,df_mak_test,df_kt_test]

test_df = pd.concat(frames)
print(test_df.shape)
test_df.reset_index(drop=True).head(2)


# In[66]:


print(train_df.shape)
train_df.reset_index(drop=True).head(2)


# ## Feature Scaling 

# Normalization is scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1. It is also known as Min-Max scaling.
# [ X'=(X-Xmin)/(Xmax-Xmin) ] Xmax and Xmin are the maximum and the minimum values of the feature respectively.
# When the value of X is the minimum value in the column, the numerator will be 0, and hence X’ is 0.On the other hand, when the value of X is the maximum value in the column, the numerator is equal to the denominator and thus the value of X’ is 1
# If the value of X is between the minimum and the maximum value, then the value of X’ is between 0 and 1
# 
# 
# Standardization is another scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.
# Mu is the mean of the feature values and Sigma is the standard deviation of the feature values. Note that in this case, the values are not restricted to a particular range. [X' = (X-Mu)/Sigma] Unlike normalization, standardization does not have a bounding range. So, even if you have outliers in your data, they will not be affected by standardization.

# ##### Standardization 

# In[67]:


from sklearn.preprocessing import StandardScaler

# decision_function_shape='ovr',kernel='poly',C=20,degree=3

# numerical features
num_cols = ['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution']

# apply standardization on numerical features
for i in num_cols:  
    # fit on training data column
    scale = StandardScaler().fit(train_df[[i]])
    
    # transform the training data column
    train_df[i] = scale.transform(train_df[[i]])
    
    # transform the validation data column
    val_df[i] = scale.transform(val_df[[i]])
    
    # transform the testing data column
    test_df[i] = scale.transform(test_df[[i]])


# In[68]:


train_df.reset_index(drop=True).head(2)


# In[69]:


val_df.reset_index(drop=True).head(2)


# In[70]:


test_df.reset_index(drop=True).head(2)


# In[71]:


y_test=test_df['ClassName']
y_val=val_df['ClassName']
y_train=train_df['ClassName']


# In[72]:


ftrain_df=train_df[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution']].copy()
fval_df=val_df[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution']].copy()
ftest_df=test_df[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution']].copy()


# In[73]:


ftrain_df.reset_index(drop=True).head()


# In[74]:


fval_df.reset_index(drop=True).head()


# In[75]:


ftest_df.reset_index(drop=True).head()


# ## EDA

# In[76]:


#analyse the train set using a Multivariate Analysis techniques i.e. Correlation matrix 
cormat=ftrain_df.corr()
plt.figure(figsize=(8,5))
g= sns.heatmap(cormat,annot=True,cmap='viridis',linewidths=.5)


# In[77]:


final_train_set=ftrain_df.join(y_train)

plt.figure(figsize=(8,5))
sns.scatterplot(y='OnlyFrequency', x='ClassName', data=final_train_set, hue='ClassName')


# In[78]:


plt.figure(figsize=(8,5))
sns.scatterplot(y='OnlyProbalilityDistribution', x='ClassName', data=final_train_set, hue='ClassName')


# In[79]:


plt.figure(figsize=(20,15))
sns.boxplot(x=final_train_set['OnlyProbalilityDistribution'],y=final_train_set['ClassName'])


# In[80]:


plt.figure(figsize=(30,15))
sns.boxplot(x=final_train_set['OnlyFrequency'],y=final_train_set['ClassName'])


# ## Feature Engineering

# In[81]:


final_val_set=fval_df.join(y_val)
final_test_set=ftest_df.join(y_test)

final_train_set['encoded_ClassName']= preprocessing.LabelEncoder().fit_transform(final_train_set['ClassName'])
final_y_train=final_train_set['encoded_ClassName']

final_val_set['encoded_ClassName']= preprocessing.LabelEncoder().fit_transform(final_val_set['ClassName'])
final_y_val=final_val_set['encoded_ClassName']

final_test_set['encoded_ClassName']= preprocessing.LabelEncoder().fit_transform(final_test_set['ClassName'])
final_y_test=final_test_set['encoded_ClassName']


# In[82]:


final_train_set.reset_index(drop=True).head(100)


# ## Feature Selection

# In[83]:


cormat=final_train_set.corr()
plt.figure(figsize=(20,10))
g= sns.heatmap(cormat,annot=True,cmap='viridis',linewidths=.5)


# ### Building Model

# ##### SVM

# In[84]:


#Fitting Support Vector Classifer to the Training set
svmclassifier = SVC(C=10,kernel='rbf')
svmclassifier.fit(ftrain_df, y_train)
# Predicting the Test set results
y_val_pred = svmclassifier.predict(fval_df)
y_train_pred = svmclassifier.predict(ftrain_df)
y_test_pred = svmclassifier.predict(ftest_df)
# Accuracy on the Train set results                              
print('\n'+'-'*20+'Accuracy Score on the Train set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_train,y_train_pred)))
# Accuracy on the Validation set results                              
print('\n'+'-'*20+'Accuracy Score on the Validation set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_val,y_val_pred)))
# Accuracy on the Test set results                              
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_test_pred)))


# In[85]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))

"""
Regularization parameter (C): 0.1,1,10,100
    The C parameter in SVM is mainly used for the Penalty parameter of the error term.
Gamma Parameter: 1,10,100,1000 
    Gamma is used when we use the Gaussian RBF kernel.
    If you use linear or polynomial kernel then you do not need gamma only you need C hypermeter.
    It decides that how much curvature we want in a decision boundary.
Kernel:In this parameter, it’s very simple, in this parameter few option’s are there like if you want to 
    model it in a linear manner, we go for ‘linear’ or if your model did not have proper accuracy then you go for 
    non-linear SVM like ‘rbf’, ‘poly’ and ‘sigmoid’ for better accuracy.
Degree:
    It controls the flexibility of the decision boundary.
    Higher degrees yield more flexible decision boundaries.
    Highly recommended for polynomial kernel
decision_function_shape : {'ovo', 'ovr'}    
SVM advantages:
    More effective for high dimensional space
    Handles non-linear data efficiently by using the kernel trick
    A small change to the data does not greatly affect the hyperplane and hence the SVM. So the SVM model is stable

"""
# #### Logistic Regression

# In[86]:


# Fitting Logistic Regression to the Training set
logclassifier = LogisticRegression(max_iter=90000)
print(logclassifier)
logclassifier.fit(ftrain_df, y_train)

y_val_pred = logclassifier.predict(fval_df)
y_train_pred = logclassifier.predict(ftrain_df)
y_test_pred = logclassifier.predict(ftest_df)
# Accuracy on the Train set results                              
print('\n'+'-'*20+'Accuracy Score on the Train set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_train,y_train_pred)))
# Accuracy on the Validation set results                              
print('\n'+'-'*20+'Accuracy Score on the Validation set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_val,y_val_pred)))
# Accuracy on the Test set results                              
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_test_pred)))


# #### KNN

# In[87]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=100)

# Train the model using the training sets
model.fit(ftrain_df,y_train)

#Predict Output
y_val_pred = model.predict(fval_df)
y_train_pred = model.predict(ftrain_df)
y_test_pred = model.predict(ftest_df)
# Accuracy on the Train set results                              
print('\n'+'-'*20+'Accuracy Score on the Train set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_train,y_train_pred)))
# Accuracy on the Validation set results                              
print('\n'+'-'*20+'Accuracy Score on the Validation set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_val,y_val_pred)))
# Accuracy on the Test set results                              
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_test_pred)))


# 
# #### Random Forest Classifier

# In[88]:


# Fitting Random Forest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier()
print(rfclassifier)
rfclassifier.fit(ftrain_df, y_train)

y_val_pred = rfclassifier.predict(fval_df)
y_train_pred = rfclassifier.predict(ftrain_df)
y_test_pred = rfclassifier.predict(ftest_df)
# Accuracy on the Train set results                              
print('\n'+'-'*20+'Accuracy Score on the Train set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_train,y_train_pred)))
# Accuracy on the Validation set results                              
print('\n'+'-'*20+'Accuracy Score on the Validation set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_val,y_val_pred)))
# Accuracy on the Test set results                              
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_test_pred)))


# ##### Checking of a self-made file and its data

# In[89]:


data = [[20,45,5]]
new_df = pd.DataFrame(data, columns=['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution'])


# In[90]:


new_df


# In[91]:


print(svmclassifier.predict(new_df))


# ### Save the model built using SVM

# In[92]:


filename='modelfinal.sav'


# In[93]:


pickle.dump(svmclassifier, open(filename, 'wb'))


# In[94]:


#trying to load the model back again and test on the above self made data
load_model = pickle.load(open(filename,'rb'))


# In[95]:


print(load_model.score(ftest_df,y_test))


# In[ ]:





# In[ ]:





# In[ ]:




