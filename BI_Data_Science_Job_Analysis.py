# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:26:18 2019

@author: Gokul
"""

import pandas as pd
import re
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set((stopwords.words('english')))
from nltk.tokenize import word_tokenize

##### Change the path here to run
#################################
#####>>>>>>>>>>>>>>>>>
df1 = pd.read_csv("C:/Users/Gokul/Desktop/Bi_Final_Project_Data_Science/alldata.csv")
df1['reviews'] = df1['reviews'].fillna(0) #replace null values with 0.
df1.head()
df1 = df1.dropna(axis=0, how = 'any') #remove rows with all null values
sum(df1.isnull().any(axis=1)) # check if the removal is successful

df=df1
df['description'] = df['description'].apply(lambda x: re.sub('[^\w\s]','', x))
df.to_csv('1_cleandata.csv')

# Put all string together
all_job_description = ""
for job in df['description']:
    all_job_description += " " + job
  
    
# Remove stop words
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

#from nltk.corpus import stopwords
#stop_words = set((stopwords.words('english')))


#from nltk.tokenize import word_tokenize
all_job_description_l = all_job_description.lower()
tokens = word_tokenize(str(all_job_description_l))
len(tokens)


filtered_description = []
for word in tokens:
    if word not in list(stop_words):
        filtered_description += [word]
        
filtered_description_string = ""
for token in filtered_description:
    filtered_description_string += ' ' + token

from collections import Counter
qq=Counter(filtered_description).most_common()
qq

degree_vector = []
for string in df['description']:
    string = str(string).lower()
    if 'degree' in string:
        index = string.index('degree')
        if index < 100:
            degree_vector.append(string[0:index+100])
        else:
            degree_vector.append(string[index-100:index+100])
    else:
        degree_vector.append('No degree requirement')


def countkeywords(keyword,text):
    num = [str(row).count(keyword) for row in text]
    return keyword,sum(num)


tool_list = ['python','sql','java','scala','excel',
               'aws','hadoop','spark','linux','tableau','sas','gis','sap',
               'unix','hive','oracle','perl','powerpoint','javascript','nosql',
               'mysql','matlab','html','tensorflow','spss','mongodb']

keywords = []
tool_word_count = []
for keyword in tool_list:
    keywords.append(countkeywords(keyword,filtered_description)[0])
    tool_word_count.append(countkeywords(keyword,filtered_description)[1])
    
for keyword in ['r','bi','c','redshift','scikit']:
    keywords.append(keyword)
    tool_word_count.append(filtered_description.count(keyword))
    
tool = pd.DataFrame()
tool['keyword'] = keywords
tool['word_count'] = tool_word_count
tool.to_excel('2_tool.xlsx',sheet_name = 'sheet1')


skills_list = ['machine learning','data analysis','data visualization',
            'modeling','statistical analysis','research','deep learning',
            'optimization','decision tree','logistic','random forest',
            'ab testing','web scraping','neural network','interpersonal skills',
            'communication skills','artificial intelligence']

skill_word_count = []
for skill_keyword in skills_list:
    skill_word_count.append(filtered_description_string.count(skill_keyword))
    
skill = pd.DataFrame()
skill['keyword'] = skills_list
skill['word_count'] = skill_word_count

skill_word_count2 = []
for skill_keyword in ['ai','nlp','ml','bi']:
    skill_word_count2.append(filtered_description.count(skill_keyword))
additional_skill_keyword = ['ai','nlp','ml','bi']
additional = [[word,filtered_description.count(word)] for word in additional_skill_keyword]


filtered_description.count('spark')
skill2 = pd.DataFrame(additional,columns = ['keyword','word_count'])
skill = skill.append(skill2)
skill.to_excel('3_skill.xlsx',sheet_name = 'sheet1')


degree_list = ['master',"masters",'phd','bachelor',"bachelors",
             'high school','No degree requirement']

degree_word_count = []
for degree in degree_list:
    degree_word_count.append(countkeywords(degree,degree_vector)[1])
    
# Tokenize degree_vector to count abbrevations of degrees.
#from nltk.tokenize import word_tokenize
tokenized_degree_vector = [word_tokenize(degree) for degree in degree_vector]
for degree in ['ba','bs','ma','ms','undergraduate','graduate']:
    degree_list.append(degree)
    degree_word_count.append(countkeywords(degree,tokenized_degree_vector)[1])
    
degree = pd.DataFrame()
degree['keyword'] = degree_list
degree['word_count'] = degree_word_count
degree.to_excel('4_degree.xlsx',sheet_name = 'sheet1')


major_list = ['computer science','data science','statistics','quantitative',
            'mathematics','economics','liberal arts','science','buisness',
            'engineer','engineering','psychology','biology','medicine',
            'marketing','business analysis','chemical engineering','chemistry',
            'public health','public relationship']

major_word_count = []
for major in major_list:
    major_word_count.append(countkeywords(major,degree_vector)[1])

major = pd.DataFrame()
major['keyword'] = major_list
major['word_count'] = major_word_count
major.to_excel('5_major.xlsx',sheet_name = 'sheet1')


# Compare by title
lower_position = [position.lower() for position in df['position']]
temp_counter = Counter(lower_position).most_common()

category = []
for position in lower_position:
    if "data scientist" in position or 'data science' in position or "machine learning" in position:
        category.append('data scientist')
    elif "analyst" in position or "analytics" in position:
        category.append('data analyst')
    elif "engineer" in position or "artificial intelligence" in position or "ai" in position:
        category.append('engineer')
    else:
        category.append('other')
              
df['title_category'] = category
Counter(category).most_common()

# Compare by company size
size_category = []
for review in df['reviews']:
    if review <= 50:
        size_category.append('small')
    elif review <= 700:
        size_category.append('medium')
    else:
        size_category.append('large')
        
df['size'] = size_category
Counter(size_category).most_common()

# Generate a new dataframe to put all keywords and frequencies in it.
all_keywords = tool_list + skills_list + degree_list + major_list

len(all_keywords)

def word_frequency(keyword):
    temp = []
    for single in df['description']:
        single = single.lower()
        temp.append(single.count(keyword))
    return temp

def add_column(word):
    df[word] = word_frequency(word)

for word in all_keywords:
    add_column(word)

df.to_csv("6_final.csv")
df.to_csv("6_final.xlsx")

# Tokenize all string to count the abbreviation words.
tokenized_description= [word_tokenize(str(single).lower()) for single in df['description']] 
len(tokenized_description)

addtional_keywords = ['ai','ml','bi','r','ba','bs','ma','ms']


def addtional_word_frequency(keyword):
    temp = []
    for single in tokenized_description:
        temp.append(single.count(keyword))
    return temp

def add_additional_column(word):
    df[word] = addtional_word_frequency(word)
    
for word in addtional_keywords:
    add_additional_column(word)

copy = df

list(copy)
copy['artificial intelligence'] += copy['ai']
copy['sql'] += copy['mysql']
copy['machine learning'] += copy['ml']
copy['master'] += (copy['master'] + copy['ma'] + copy['ms'])
copy['bachelor'] += (copy['bachelors'] + copy['ba'] + copy['bs'])
copy['engineer'] += copy['engineering']

copy = copy.drop(columns = ['ai','mysql','ml','master','ma','ms','bachelors','ba','bs','engineering','ba','bs'])

copy.to_csv('7_finalclean.csv')
copy.to_csv('7_finalclean.xlsx')

copy2 = copy.drop(columns = ['unix','powerpoint','oracle','javascript','nosql','linux','perl','html','mongodb',
                             'buisness','interpersonal skills','communication skills',
                            'liberal arts','buisness','psychology',
                            'biology','medicine','marketing',
                            'chemical engineering','chemistry','public health','public relationship'])

list(copy2)[25:60]

copy2['data analysis'] += copy2['statistical analysis']
len(list(copy2))
list(copy2)

copy2.to_csv('8_finalfinalclean.csv')
copy2.to_csv('8_finalfinalclean.xlsx')

dfd = pd.read_csv("8_finalfinalclean.csv")
dfd.drop(columns=['Unnamed: 0'],inplace=True)

df_engineer=dfd[dfd['title_category']=='engineer']
df_engineer.reset_index(inplace=True)
df_engineer.to_csv('Engineer.csv')

df_other=dfd[dfd['title_category']=='other']
df_other.reset_index(inplace=True)
df_other.to_csv('other.csv')

df_Data_scientist=dfd[dfd['title_category']=='data scientist']
df_Data_scientist.reset_index(inplace=True)
df_Data_scientist.to_csv('Data Scientist.csv')

df_Data_analyst=dfd[dfd['title_category']=='data analyst']
df_Data_analyst.reset_index(inplace=True)
df_Data_analyst.to_csv('Data Analyst.csv')

labels = ('Data Scientist','Data Analyst','Engineer','Other')
sizes = len(df_Data_scientist),len(df_Data_analyst),len(df_engineer),len(df_other)
plt.pie(sizes, labels=labels,autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

############################################################################

df_Data_scientist
ds_all_sum=df_Data_scientist.sum(axis=0)
ds_all_sum.to_csv('Data Scientist all sum.csv')

ds_tools=ds_all_sum[['python','r','sql','java','scala','excel','aws','hadoop','spark',
 'tableau','sas','gis','sap','hive','matlab','tensorflow','spss',]]
labels = ('python','r','sql','java','scala','excel','aws','hadoop','spark',
 'tableau','sas','gis','sap','hive','matlab','tensorflow','spss')
sizes = list(ds_tools)
plt.pie(sizes, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

ds_skills=ds_all_sum[['machine learning','data analysis','data visualization','modeling',
 'statistical analysis','research','deep learning','optimization','decision tree',
 'logistic','random forest','ab testing','web scraping','neural network','artificial intelligence','bi']]
labels = ('machine learning','data analysis','data visualization','modeling',
 'statistical analysis','research','deep learning','optimization','decision tree',
 'logistic','random forest','ab testing','web scraping','neural network','artificial intelligence','bi')
sizes = list(ds_skills)
plt.pie(sizes, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

ds_degree=ds_all_sum[['masters','phd','bachelor','high school','No degree requirement','undergraduate','graduate']]
labels = ('masters','phd','bachelor','high school','No degree requirement','undergraduate','graduate')
sizes = list(ds_degree)
plt.pie(sizes, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

ds_major=ds_all_sum[['computer science','data science','statistics','quantitative','mathematics',
 'economics','science','engineer','business analysis']]
labels = ('computer science','data science','statistics','quantitative','mathematics',
 'economics','science','engineer','business analysis')
sizes = list(ds_major)
plt.pie(sizes, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


###################################################


df_Data_analyst
da_all_sum=df_Data_analyst.sum(axis= 0)
da_all_sum.to_csv('Data Analyst all sum.csv')

da_tools=da_all_sum[['python','r','sql','java','scala','excel','aws','hadoop','spark',
 'tableau','sas','gis','sap','hive','matlab','tensorflow','spss']]
labels = ('python','r','sql','java','scala','excel','aws','hadoop','spark',
 'tableau','sas','gis','sap','hive','matlab','tensorflow','spss')
sizes = list(da_tools)
plt.pie(sizes, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

da_skills=da_all_sum[['machine learning','data analysis','data visualization','modeling',
 'statistical analysis','research','deep learning','optimization','decision tree',
 'logistic','random forest','ab testing','web scraping','neural network','artificial intelligence','bi']]
labels = ('machine learning','data analysis','data visualization','modeling',
 'statistical analysis','research','deep learning','optimization','decision tree',
 'logistic','random forest','ab testing','web scraping','neural network','artificial intelligence','bi')
sizes = list(da_skills)
plt.pie(sizes, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

da_degree=da_all_sum[['masters','phd','bachelor','high school','No degree requirement','undergraduate','graduate']]
labels = ('masters','phd','bachelor','high school','No degree requirement','undergraduate','graduate')
sizes = list(da_degree)
plt.pie(sizes, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

da_major=da_all_sum[['computer science','data science','statistics','quantitative','mathematics',
 'economics','science','engineer','business analysis']]
labels = ('computer science','data science','statistics','quantitative','mathematics',
 'economics','science','engineer','business analysis')
sizes = list(da_major)
plt.pie(sizes, labels=labels,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

############################################################################################################

Data_wording=['python','r','sql','java','scala','excel','aws','hadoop','spark','tableau','sas','gis','sap','hive','matlab','tensorflow','spss',
  'machine learning','data analysis','data visualization','modeling','statistical analysis','research','deep learning','optimization','decision tree','logistic','random forest','ab testing','web scraping','neural network','artificial intelligence','bi',
  'masters','phd','bachelor','high school','No degree requirement','undergraduate','graduate',
  'computer science','data science','statistics','quantitative','mathematics',
 'economics','science','engineer','business analysis'
 ]

duping=['Python','R','SQL','Java','Scala','Excel','AWS','Hadoop','Spark','Tableau','SAS','GIS','SAP','Hive','MatLab','Tensorflow','SPSS',
  'Machine-Learning','Data-Analysis','Data-Visualization','Modeling','Statistical-Analysis','Research','Deep-Learning','Optimization','Decision-Tree','Logistic','Random-Forest','ab-Testing','Web-Scraping','Neural-Network','AI','BI',
  'Masters','phd','Bachelor','High-School','No-Degree','Undergraduate','Graduate',
  'Computer-Science','Data-Science','Statistics','Quantitative','Mathematics',
 'Economics','Science','Engineering','Business-Analysis'
 ]

# Data Scientist
all_words_ds=''
k=0
for i in Data_wording:
    word_sp=duping[k]+'  '
    k=k+1
    word=word_sp*ds_all_sum[i]
    all_words_ds=all_words_ds + word

len(all_words_ds)

text_file = open("Data scientist word cloud.txt", "w")
text_file.write("%s"% all_words_ds)
text_file.close()

# Data Analyst
all_words_da=''
k=0
for i in Data_wording:
    word_sp=duping[k]+'  '
    k=k+1
    word=word_sp*da_all_sum[i]
    all_words_da=all_words_da + word
    
len(all_words_da)

text_file = open("Data Analyst word cloud.txt", "w")
text_file.write("%s"% all_words_da)
text_file.close()

###############################################################################################


################################################################################################