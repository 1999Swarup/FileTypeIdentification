#!/usr/bin/env python
# coding: utf-8

# In[112]:


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

# In[113]:


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
    for prob_value, prob_dist_freq in prob_dist_c.most_common(n=5):
        prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
        prob_val.append("{:02x}".format(prob_value))
        prob_freq.append("{:.0%}".format(prob_dist_freq))
    #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
    for value, frequency in c.most_common(n=5):
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

# In[114]:


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
    for prob_value, prob_dist_freq in prob_dist_c.most_common(n=5):
        prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
        prob_val.append("{:02x}".format(prob_value))
        prob_freq.append("{:.0%}".format(prob_dist_freq))
    #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
    for value, frequency in c.most_common(n=5):
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

# In[115]:


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
        for prob_value, prob_dist_freq in prob_dist_c.most_common(n=5):
            prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
            prob_val.append("{:02x}".format(prob_value))
            prob_freq.append("{:.0%}".format(prob_dist_freq))
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for value, frequency in c.most_common(n=5):
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

# In[116]:


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
        for prob_value, prob_dist_freq in prob_dist_c.most_common(n=5):
            prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
            prob_val.append("{:02x}".format(prob_value))
            prob_freq.append("{:.0%}".format(prob_dist_freq))
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for value, frequency in c.most_common(n=5):
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

# In[117]:


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
        for prob_value, prob_dist_freq in prob_dist_c.most_common(n=5):
            prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
            prob_val.append("{:02x}".format(prob_value))
            prob_freq.append("{:.0%}".format(prob_dist_freq))
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for value, frequency in c.most_common(n=5):
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

# In[118]:


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
        for prob_value, prob_dist_freq in prob_dist_c.most_common(n=5):
            prob_freq_dist_value.append("0x{:02x}: {:.04f}".format(prob_value, prob_dist_freq))
            prob_val.append("{:02x}".format(prob_value))
            prob_freq.append("{:.0%}".format(prob_dist_freq))
        #List the n most common elements and their counts from the most common to the least.  If n is None, then list all element counts.
        for value, frequency in c.most_common(n=5):
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


# ##  Splitting above data frames to form train, validation and test data frame

# In[119]:


print((df_kt.shape,df_mak.shape,df_ml.shape,df_rexx.shape,df_csproj.shape,df_jenkinsfile.shape))


# ##### Train and Test dataframe for kt files

# In[120]:


df_kt_train, df_kt_test = train_test_split(df_kt, test_size=0.3)
df_kt_train.reset_index(drop=True).head(2)


# In[121]:


df_kt_test.reset_index(drop=True).head(2)


# ##### Train and Test dataframe for mak files

# In[122]:


df_mak_train, df_mak_test = train_test_split(df_mak, test_size=0.3)
df_mak_train.reset_index(drop=True).head(2)


# In[123]:


df_mak_test.reset_index(drop=True).head(2)


# ##### Train and Test dataframe for ml files

# In[124]:


df_ml_train, df_ml_test = train_test_split(df_ml, test_size=0.3)
df_ml_train.reset_index(drop=True).head(2)


# In[125]:


df_ml_test.reset_index(drop=True).head(2)


# ##### Train and Test dataframe for rexx files

# In[126]:


df_rexx_train, df_rexx_test = train_test_split(df_rexx, test_size=0.3)
df_rexx_train.reset_index(drop=True).head(2)


# In[127]:


df_rexx_test.reset_index(drop=True).head(2)


# ##### Train and Test dataframe for csproj files

# In[128]:


df_csproj_train, df_csproj_test = train_test_split(df_csproj, test_size=0.3)
df_csproj_train.reset_index(drop=True).head(2)


# In[129]:


df_csproj_test.reset_index(drop=True).head(2)


# ##### Train and Test dataframe for jenkinsfile files

# In[130]:


df_jenkinsfile_train, df_jenkinsfile_test = train_test_split(df_jenkinsfile, test_size=0.2)
df_jenkinsfile_train.reset_index(drop=True).head(2)


# In[131]:


df_jenkinsfile_test.reset_index(drop=True).head(2)


# In[132]:


# Train and Test Data Frame for each type of files has been created above
#df_kt
#df_mak
#df_ml
#df_rexx
#df_csproj
#df_jenkinsfile


# In[133]:


print((df_kt_train.shape,df_mak_train.shape,df_ml_train.shape,df_rexx_train.shape,df_csproj_train.shape,df_jenkinsfile_train.shape))


# In[134]:


print((df_kt_test.shape,df_mak_test.shape,df_ml_test.shape,df_rexx_test.shape,df_csproj_test.shape,df_jenkinsfile_test.shape))


# ### Building the train data frame

# In[135]:


frames = [df_jenkinsfile_train,df_csproj_train,df_rexx_train,df_ml_train,df_mak_train,df_kt_train]

train_df = pd.concat(frames)
print(train_df.shape)
train_df.reset_index(drop=True).head(2)


# In[136]:


train_df.reset_index(drop=True).tail(2)


# ### Validation data frame and Test data frame

# ##### Validation and Test dataframe for jenkinsfile files

# In[137]:


df_jenkinsfile_val, df_jenkinsfile_test = train_test_split(df_jenkinsfile_test, test_size=0.1)
df_jenkinsfile_val.reset_index(drop=True).head(2)


# In[138]:


df_jenkinsfile_test.reset_index(drop=True).head(2)


# ##### Validation and Test dataframe for csproj files

# In[139]:


df_csproj_val, df_csproj_test = train_test_split(df_csproj_test, test_size=0.1)
df_csproj_val.reset_index(drop=True).head(2)


# In[140]:


df_csproj_test.reset_index(drop=True).head(2)


# ##### Validation and Test dataframe for rexx files

# In[141]:


df_rexx_val, df_rexx_test = train_test_split(df_rexx_test, test_size=0.1)
df_rexx_val.reset_index(drop=True).head(2)


# In[142]:


df_rexx_test.reset_index(drop=True).head(2)


# ##### Validation and Test dataframe for ml files

# In[143]:


df_ml_val, df_ml_test = train_test_split(df_ml_test, test_size=0.1)
df_ml_val.reset_index(drop=True).head(2)


# In[144]:


df_ml_test.reset_index(drop=True).head(2)


# ##### Validation and Test dataframe for mak files

# In[145]:


df_mak_val, df_mak_test = train_test_split(df_mak_test, test_size=0.1)
df_mak_val.reset_index(drop=True).head(2)


# In[146]:


df_mak_test.reset_index(drop=True).head(2)


# ##### Validation and Test dataframe for kt files

# In[147]:


df_kt_val, df_kt_test = train_test_split(df_kt_test, test_size=0.1)
df_kt_val.reset_index(drop=True).head(2)


# In[148]:


df_kt_test.reset_index(drop=True).head(2)


# ### Building Validation data frame and Test data frame

# In[149]:


frames = [df_jenkinsfile_val,df_csproj_val,df_rexx_val,df_ml_val,df_mak_val,df_kt_val]

val_df = pd.concat(frames)
print(val_df.shape)
val_df.reset_index(drop=True).head(2)


# In[150]:


frames = [df_jenkinsfile_test,df_csproj_test,df_rexx_test,df_ml_test,df_mak_test,df_kt_test]

test_df = pd.concat(frames)
print(test_df.shape)
test_df.reset_index(drop=True).head(2)


# In[151]:


print(train_df.shape)
train_df.reset_index(drop=True).head(2)


# ## Feature Scaling & some more preprocessing

# ##### For Train DataFrame

# In[152]:


train_final_df=train_df[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','Filename']]


# In[153]:


train_final_df.reset_index(drop=True).head(2)


# In[154]:


#As all columns have the same number of lists, you can call Series.explode on each column
train_final_df=train_final_df.set_index(['Filename']).apply(pd.Series.explode).reset_index()


# In[155]:


train_final_df


# In[156]:


train_final_df['ClassName'] = train_final_df['Filename'].str.split('.').str[-1]


# In[157]:


train_final_df=train_final_df.replace('\%','',regex=True)


# In[158]:


final_train_df = train_final_df[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution']]
final_train_df.info()


# In[159]:


#replacing all infinite values with nan and than nan with 0
final_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_train_df['OnlyFrequency'].fillna(final_train_df['OnlyFrequency'].mode()[0], inplace=True)
final_train_df['OnlyProbalilityDistribution'].fillna(final_train_df['OnlyProbalilityDistribution'].mode()[0], inplace=True)
final_train_df.fillna(0, inplace=True)


# In[160]:


final_train_df['OnlyValueForAbsoluteDistribution']=final_train_df.OnlyValueForAbsoluteDistribution.astype(str)
final_train_df['OnlyFrequency']=final_train_df.OnlyFrequency.astype(int)
final_train_df['OnlyProbalilityDistribution']=final_train_df.OnlyProbalilityDistribution.astype(int)
final_train_df.info()


# In[161]:


final_train_df.head(2)


# In[162]:


y_train=train_final_df['ClassName']
y_train


# ##### For Validation DataFrame

# In[163]:


val_final_df=val_df[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','Filename']]


# In[164]:


val_final_df.reset_index(drop=True).head(2)


# In[165]:


#As all columns have the same number of lists, you can call Series.explode on each column
val_final_df=val_final_df.set_index(['Filename']).apply(pd.Series.explode).reset_index()


# In[166]:


val_final_df


# In[167]:


val_final_df['ClassName'] = val_final_df['Filename'].str.split('.').str[-1]
val_final_df=val_final_df.replace('\%','',regex=True)


# In[168]:


val_final_df


# In[169]:


final_val_df = val_final_df[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution']]
final_val_df.info()


# In[170]:


#replacing all infinite values with nan and than nan with 0
final_val_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_val_df['OnlyFrequency'].fillna(final_val_df['OnlyFrequency'].mode()[0], inplace=True)
final_val_df['OnlyProbalilityDistribution'].fillna(final_val_df['OnlyProbalilityDistribution'].mode()[0], inplace=True)
final_val_df.fillna(0, inplace=True)


# In[171]:


final_val_df['OnlyValueForAbsoluteDistribution']=final_val_df.OnlyValueForAbsoluteDistribution.astype(str)
final_val_df['OnlyFrequency']=final_val_df.OnlyFrequency.astype(int)
final_val_df['OnlyProbalilityDistribution']=final_val_df.OnlyProbalilityDistribution.astype(int)
final_val_df.info()


# In[172]:


y_val=val_final_df['ClassName']
y_val


# ##### For Test DataFrame

# In[173]:


test_final_df=test_df[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution','Filename']]


# In[174]:


test_final_df.reset_index(drop=True).head(2)


# In[175]:


#As all columns have the same number of lists, you can call Series.explode on each column
test_final_df=test_final_df.set_index(['Filename']).apply(pd.Series.explode).reset_index()


# In[176]:


test_final_df


# In[177]:


test_final_df['ClassName'] = test_final_df['Filename'].str.split('.').str[-1]
test_final_df=test_final_df.replace('\%','',regex=True)


# In[178]:


test_final_df


# In[179]:


final_test_df = test_final_df[['OnlyValueForAbsoluteDistribution','OnlyFrequency','OnlyProbalilityDistribution']]
final_test_df.info()


# In[180]:


#replacing all infinite values with nan and than nan with 0
final_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_test_df['OnlyFrequency'].fillna(final_test_df['OnlyFrequency'].mode()[0], inplace=True)
final_test_df['OnlyProbalilityDistribution'].fillna(final_test_df['OnlyProbalilityDistribution'].mode()[0], inplace=True)
final_test_df.fillna(0, inplace=True)


# In[181]:


final_test_df['OnlyValueForAbsoluteDistribution']=final_test_df.OnlyValueForAbsoluteDistribution.astype(str)
final_test_df['OnlyFrequency']=final_test_df.OnlyFrequency.astype(int)
final_test_df['OnlyProbalilityDistribution']=final_test_df.OnlyProbalilityDistribution.astype(int)
final_test_df.info()


# Normalization is scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1. It is also known as Min-Max scaling.
# [ X'=(X-Xmin)/(Xmax-Xmin) ] Xmax and Xmin are the maximum and the minimum values of the feature respectively.
# When the value of X is the minimum value in the column, the numerator will be 0, and hence X’ is 0.On the other hand, when the value of X is the maximum value in the column, the numerator is equal to the denominator and thus the value of X’ is 1
# If the value of X is between the minimum and the maximum value, then the value of X’ is between 0 and 1
# 
# 
# Standardization is another scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.
# Mu is the mean of the feature values and Sigma is the standard deviation of the feature values. Note that in this case, the values are not restricted to a particular range. [X' = (X-Mu)/Sigma] Unlike normalization, standardization does not have a bounding range. So, even if you have outliers in your data, they will not be affected by standardization.

# ##### Standardization 

# In[182]:


from sklearn.preprocessing import StandardScaler

# decision_function_shape='ovr',kernel='poly',C=20,degree=3

# numerical features
num_cols = ['OnlyFrequency','OnlyProbalilityDistribution']

# apply standardization on numerical features
for i in num_cols:  
    # fit on training data column
    scale = StandardScaler().fit(final_train_df[[i]])
    
    # transform the training data column
    final_train_df[i] = scale.transform(final_train_df[[i]])
    
    # transform the validation data column
    final_val_df[i] = scale.transform(final_val_df[[i]])
    
    # transform the testing data column
    final_test_df[i] = scale.transform(final_test_df[[i]])


# 

# In[183]:


final_train_df.head(2)


# In[184]:


final_val_df.head(2)


# In[185]:


final_test_df.head(2)


# In[186]:


y_test=test_final_df['ClassName']
y_test


# #### Check for null and ununique

# ###### On Train DataFrame

# In[187]:


#replacing all infinite values with nan and than nan with 0
final_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_train_df.fillna(0, inplace=True)
# using isnull() function  
final_train_df.isnull().sum()


# In[188]:


final_train_df.info()


# In[189]:


final_train_df


# ###### On Validation DataFrame

# In[190]:


#replacing all infinite values with nan and than nan with 0
final_val_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_val_df.fillna(0, inplace=True)
# using isnull() function  
final_val_df.isnull().sum()


# In[191]:


final_val_df.info()


# In[192]:


final_val_df


# ###### On Test DataFrame

# In[193]:


#replacing all infinite values with nan and than nan with 0
final_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_test_df.fillna(0, inplace=True)
# using isnull() function  
final_test_df.isnull().sum()


# In[194]:


final_test_df.info()


# In[195]:


final_test_df


# ## EDA

# In[196]:


#analyse the train set using a Multivariate Analysis techniques i.e. Correlation matrix 
cormat=final_train_df.corr()
plt.figure(figsize=(8,5))
g= sns.heatmap(cormat,annot=True,cmap='viridis',linewidths=.5)


# In[197]:


final_train_set=final_train_df.join(y_train)

plt.figure(figsize=(8,5))
sns.scatterplot(y='OnlyFrequency', x='ClassName', data=final_train_set, hue='ClassName')


# In[198]:


plt.figure(figsize=(8,5))
sns.scatterplot(y='OnlyProbalilityDistribution', x='ClassName', data=final_train_set, hue='ClassName')


# In[199]:


plt.figure(figsize=(20,5))
sns.scatterplot(x='OnlyValueForAbsoluteDistribution', y='ClassName', data=final_train_set, hue='ClassName')


# In[200]:


sns.boxplot(x=final_train_set['OnlyProbalilityDistribution'],y=final_train_set['ClassName'])


# In[201]:


plt.figure(figsize=(20,10))
sns.boxplot(x=final_train_set['OnlyFrequency'],y=final_train_set['ClassName'])


# ## Feature Engineering

# One-Hot Encoding of "OnlyValueForAbsoluteDistribution" label will help as it is the column which has both str and int type data.
# It simply creates additional features based on the number of unique values in the categorical feature - "OnlyValueForAbsoluteDistribution". Every unique value in the category will be added as a feature.

# ##### One-Hot encoding the categorical parameters using get_dummies()

# In[202]:


final_train_df=pd.get_dummies(final_train_df, columns = ['OnlyValueForAbsoluteDistribution'])
final_val_df=pd.get_dummies(final_val_df, columns = ['OnlyValueForAbsoluteDistribution'])
final_test_df=pd.get_dummies(final_test_df,columns = ['OnlyValueForAbsoluteDistribution'])


# In[203]:


final_train_df.info()


# In[204]:


final_val_df.info()


# In[205]:


final_test_df.info()


# final_train_df.drop('OnlyValueForAbsoluteDistribution', axis=1, inplace=True)
# final_val_df.drop('OnlyValueForAbsoluteDistribution', axis=1, inplace=True)
# final_test_df.drop('OnlyValueForAbsoluteDistribution', axis=1, inplace=True)

# In[206]:


final_val_set=final_val_df.join(y_val)
final_test_set=final_test_df.join(y_test)

final_train_set['encoded_ClassName']= preprocessing.LabelEncoder().fit_transform(final_train_set['ClassName'])
final_y_train=final_train_set['encoded_ClassName']

final_val_set['encoded_ClassName']= preprocessing.LabelEncoder().fit_transform(final_val_set['ClassName'])
final_y_val=final_val_set['encoded_ClassName']

final_test_set['encoded_ClassName']= preprocessing.LabelEncoder().fit_transform(final_test_set['ClassName'])
final_y_test=final_test_set['encoded_ClassName']


# ## Feature Selection

# In[207]:


final_train_set=final_train_df.join(final_y_train)

cormat=final_train_set.corr()
plt.figure(figsize=(60,50))
g= sns.heatmap(cormat,annot=True,cmap='viridis',linewidths=.5)


# In[222]:


correlated_features = set()
correlation_matrix=final_train_set.corr().abs()

for i in range(len(correlation_matrix .columns)):
    if abs(correlation_matrix.iloc[i, 80]) > 0.04:
        colname = correlation_matrix.columns[i]
        print("-"*50)   
        print(colname,"\n",correlation_matrix.iloc[i, 80])
        print("-"*50)


# In[223]:


#As found above these features were having a good correlation with target hence we can keep them and remove the rest.
#I am keeping only 5 additional features as from each file I also extract only 5 maximum frequent bytes
"""
--------------------------------------------------
OnlyFrequency 
 0.04203620668260215
--------------------------------------------------
--------------------------------------------------
OnlyProbalilityDistribution 
 0.09387349130506747
--------------------------------------------------
--------------------------------------------------
OnlyValueForAbsoluteDistribution_80 
 0.14846575492645617
--------------------------------------------------
--------------------------------------------------
OnlyValueForAbsoluteDistribution_94 
 0.14846575492645617
--------------------------------------------------
--------------------------------------------------
OnlyValueForAbsoluteDistribution_e2 
 0.14846575492645617
--------------------------------------------------
--------------------------------------------------
OnlyValueForAbsoluteDistribution_2a 
 0.08567683058967268
--------------------------------------------------
--------------------------------------------------
OnlyValueForAbsoluteDistribution_72 
 0.09888809591864221
"""
final_train_df = final_train_df[['OnlyValueForAbsoluteDistribution_80','OnlyValueForAbsoluteDistribution_94','OnlyValueForAbsoluteDistribution_e2','OnlyValueForAbsoluteDistribution_2a','OnlyValueForAbsoluteDistribution_72','OnlyFrequency','OnlyProbalilityDistribution']]
final_val_df = final_val_df[['OnlyValueForAbsoluteDistribution_80','OnlyValueForAbsoluteDistribution_94','OnlyValueForAbsoluteDistribution_e2','OnlyValueForAbsoluteDistribution_2a','OnlyValueForAbsoluteDistribution_72','OnlyFrequency','OnlyProbalilityDistribution']]
final_test_df = final_test_df[['OnlyValueForAbsoluteDistribution_80','OnlyValueForAbsoluteDistribution_94','OnlyValueForAbsoluteDistribution_e2','OnlyValueForAbsoluteDistribution_2a','OnlyValueForAbsoluteDistribution_72','OnlyFrequency','OnlyProbalilityDistribution']]

final_train_df.info(),final_val_df.info(),final_test_df.info()


# ### Building Model

# ##### SVM

# In[224]:


#Fitting Support Vector Classifer to the Training set
svmclassifier = SVC(C=10,kernel='rbf',gamma=100,decision_function_shape='ovo')
svmclassifier.fit(final_train_df, y_train)
# Predicting the Test set results
y_val_pred = svmclassifier.predict(final_val_df)
y_train_pred = svmclassifier.predict(final_train_df)
y_test_pred = svmclassifier.predict(final_test_df)
# Accuracy on the Train set results                              
print('\n'+'-'*20+'Accuracy Score on the Train set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_train,y_train_pred)))
# Accuracy on the Validation set results                              
print('\n'+'-'*20+'Accuracy Score on the Validation set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_val,y_val_pred)))
# Accuracy on the Test set results                              
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_test_pred)))


# In[225]:


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

# In[226]:


# Fitting Logistic Regression to the Training set
logclassifier = LogisticRegression(max_iter=90000)
print(logclassifier)
logclassifier.fit(final_train_df, y_train)

y_val_pred = logclassifier.predict(final_val_df)
y_train_pred = logclassifier.predict(final_train_df)
y_test_pred = logclassifier.predict(final_test_df)
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

# In[227]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=100)

# Train the model using the training sets
model.fit(final_train_df,y_train)

#Predict Output
y_val_pred = model.predict(final_val_df)
y_train_pred = model.predict(final_train_df)
y_test_pred = model.predict(final_test_df)
# Accuracy on the Train set results                              
print('\n'+'-'*20+'Accuracy Score on the Train set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_train,y_train_pred)))
# Accuracy on the Validation set results                              
print('\n'+'-'*20+'Accuracy Score on the Validation set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_val,y_val_pred)))
# Accuracy on the Test set results                              
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,y_test_pred)))


# #### Random Forest Classifier

# In[228]:


# Fitting Random Forest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier()
print(rfclassifier)
rfclassifier.fit(final_train_df, y_train)

y_val_pred = rfclassifier.predict(final_val_df)
y_train_pred = rfclassifier.predict(final_train_df)
y_test_pred = rfclassifier.predict(final_test_df)
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
# 
# ##### Checking of a self-made file and its data

# In[229]:


data = [[0,1,0,1,0,45,5]]
new_df = pd.DataFrame(data, columns=['OnlyValueForAbsoluteDistribution_80','OnlyValueForAbsoluteDistribution_94','OnlyValueForAbsoluteDistribution_e2','OnlyValueForAbsoluteDistribution_2a','OnlyValueForAbsoluteDistribution_72','OnlyFrequency','OnlyProbalilityDistribution'])


# In[230]:


new_df


# In[231]:


print(svmclassifier.predict(new_df))


# ### Save the model built using SVM

# In[232]:


filename='finalftimodel.sav'


# In[233]:


pickle.dump(svmclassifier, open(filename, 'wb'))


# In[234]:


#trying to load the model back again and test on the above self made data
load_model = pickle.load(open(filename,'rb'))


# In[235]:


print(load_model.score(final_test_df,y_test))


# In[ ]:





# In[ ]:





# In[ ]:




