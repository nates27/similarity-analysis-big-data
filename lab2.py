"""LAB 2
Name: NATHANIEL NARTEA CASANOVA
Student Number: A0262708B

"""

#Packages to import
import sys
import re
import math
import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import pyspark.sql.functions as f
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import SparkSession

# Import nltk for lemmatization.
# Download nltk models and framework for WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

# Import pandas, matplotlib, and seaborn, and seaborn for last step visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up Spark session configuration
conf = SparkConf()\
    .setMaster("local[*]")\
    .set("spark.executor.memory", "4g")\
    .set("spark.driver.memory", "2g")

sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

"""Input Files"""
arxiv = sys.argv[1] #text file
stopwords = sys.argv[2] #stopwords

with open(stopwords, 'r') as file:
    stopwords_list=file.read().split('\n')

"""Output Files"""
processed = sys.argv[3] #output for processed titles and abstracts
accuracy  = sys.argv[4] #output for accuracy results

results = sys.argv[5] #output for cosine value calculation results
heatmap = sys.argv[6] #output for task 2 heatmap, must be a string

papers =  spark.read.json(arxiv)

"""Data Preprocessing"""
# Proceses titles and abstracts in the input file
# Makes all words lower case and replace special characters with a whitespace
process_column = ['abstract', 'title']

for column in process_column:
    papers = papers.withColumn(column,f.lower(f.col(column)))
    papers = papers.withColumn(column,
                               f.regexp_replace(f.col(column), "(\\d|\\W)+", ' '))

# Tokenizes each word in the title and abstracts
# Removes stop words in the title and abstracts
for column in process_column:
    regexTokenizer = RegexTokenizer(inputCol=column, outputCol='words_'+column,\
                                     pattern="\\W")
    tokenized = regexTokenizer.transform(papers)
    stopwordsRemover = StopWordsRemover(inputCol='words_'+column,
                                         outputCol='filtered_'+column,
                                           stopWords=stopwords_list)
    papers = stopwordsRemover.transform(tokenized)
    papers = papers.withColumn(column,f.concat_ws(' ', f.col('filtered_'+column)))

# Processes categories in the input file
# Makes each category lower case and removes trailing whitespaces
papers = papers.withColumn('categories',f.lower(f.col('categories')))
papers = papers.withColumn('categories',
                               f.regexp_replace(f.col('categories'), "\s+$", ""))

"""Following code block performs word lemmatization"""

"""Function: get_wordnet_pos()

Return WORDNET part of speech (POS) compliance to WORDNET lemmatization (a,n,r,v)
Uses treebank tagset from average_perceptron_tagger to avoid any errors calling default
WODNET POS tagger

"""
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
            return 'a'
    elif treebank_tag.startswith('V'):
            return 'v'
    elif treebank_tag.startswith('N'):
            return 'n'
    elif treebank_tag.startswith('R'):
            return 'r'
    else:
    # Return noun as default POS
        return 'n'

"""Function: lemmatize()

Lemmatizes a list of words using WordNetLemmatizer
"""
def lemmatize(data_str):
    list_pos = 0
    cleaned_str = ''
    lmtzr = WordNetLemmatizer()
    #
    tagged_words = nltk.pos_tag(data_str)
    for word in tagged_words:
        lemma = lmtzr.lemmatize(word[0], get_wordnet_pos(word[1]))
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str

sparkLemmer = f.udf(lambda x: lemmatize(x), f.StringType())

papers = papers.select('id','categories','filtered_title',
                        sparkLemmer('filtered_abstract').alias('abstract'))
papers = papers.select('id','categories','abstract'
                       ,sparkLemmer('filtered_title').alias('title'))

# Output processed titles and abstracts as parquet file for analysis
# papers.repartition(1).write.parquet(processed)

"""Task 1"""
papers_rdd = papers.rdd
n = papers_rdd.count()
abstracts_rdd = papers_rdd.map(lambda x: (x['id'], x['abstract']))\
    .mapValues(lambda l: re.split(r'[^\w]+',l))\
    .flatMapValues(lambda w: w)\
    .map(lambda w: ((w[0], w[1]),1))

tf_rdd = abstracts_rdd.reduceByKey(lambda a, b: a + b)\
    .map(lambda x: (x[0][1], (x[0][0],x[1])))\


df_rdd = abstracts_rdd.map(lambda x: (x[0][1], x[0][0]))\
    .distinct()\
    .map(lambda x: (x[0], 1))\
    .reduceByKey(lambda x, y: x + y)

tf_df = tf_rdd.join(df_rdd)

tf_idf = tf_df.map(lambda x: ((x[1][0][0],x[0]),(x[1][0][1],x[1][1])))\
                   .mapValues(lambda a: (1+math.log10(a[0]))\
                                           *(math.log10((n+1)/(a[1]+1))+1))\
                    .map(lambda x: (x[0][0], (x[0][1], x[1])))\
                    .groupByKey() \
                    .mapValues(lambda x: dict(x)) \
                    .mapValues(lambda x: {k: v / math.sqrt\
                                          (sum([i**2 for i in x.values()])) for k, v in x.items()})


title_rdd = papers_rdd.map(lambda x: (x['id'], x['title']))\
    .mapValues(lambda l: re.split(r'[^\w]+',l))\
    .flatMapValues(lambda w: w)\
    .map(lambda w: ((w[0], w[1]),1))

tf_t_rdd = title_rdd.reduceByKey(lambda a, b: a + b)\
    .map(lambda x: (x[0][1], (x[0][0],x[1])))

tf_t_df = tf_t_rdd.leftOuterJoin(df_rdd)\
    .mapValues(lambda x: (x[0], 0) if x[1] is None else x)

tf_t_idf = tf_t_df.map(lambda x: ((x[1][0][0],x[0]),(x[1][0][1],x[1][1])))\
                   .mapValues(lambda a: (1+math.log10(a[0]))\
                                           *(math.log10((n+1)/(a[1]+1))+1))\
                    .map(lambda x: (x[0][0], (x[0][1], x[1])))\
                    .groupByKey() \
                    .mapValues(lambda x: dict(x)) \
                    .mapValues(lambda x: {k: v / math.sqrt\
                                          (sum([i**2 for i in x.values()]))\
                                              for k, v in x.items()})

def pairwise_dot_product(d1, d2):
    return sum(d1.get(key, 0) * d2.get(key, 0) for key in set(d1) & set(d2))\
    

combinations_rdd = tf_t_idf.cartesian(tf_idf)
cosine_rdd = combinations_rdd.map(lambda x: (x[0][0],\
                                             (x[1][0], pairwise_dot_product(x[0][1], x[1][1]))))

rel_sort = cosine_rdd.reduceByKey(lambda a, b: [a, b][a[1] < b[1]])\
    .map(lambda x: (x[0], x[1][0], x[1][1]))

top_results = rel_sort.map(lambda x: ('accuracy', (x[0], x[1])))\
    .mapValues(lambda a: 1 if a[0] == a[1] else 0)\
    .reduceByKey(lambda u, w: u + w)\
    .mapValues(lambda f: f/n)


"""Task 2"""

cat_rdd = papers_rdd.map(lambda x: ((x['id'], x['categories']), x['abstract']))\
    .mapValues(lambda l: re.split(r'[^\w]+',l))\
    .flatMapValues(lambda w: w)\
    .mapValues(lambda w: (w,1))\
    .map(lambda x: ((x[0][0],x[1][0]),(x[0][1],x[1][1])))\
    .reduceByKey(lambda a, b: (a[0], a[1]+b[1]))

cattf_rdd = cat_rdd.map(lambda x: ((x[1][0], x[0][1]),x[1][1]))\
    .reduceByKey(lambda a, b: a + b)

cat_vec = cattf_rdd.map(lambda x: (x[0][0], (x[0][1], x[1])))\
                .groupByKey()\
                .mapValues(lambda x: dict(x)) \
                .mapValues(lambda x: {k: v / math.sqrt\
                                         (sum([i**2 for i in x.values()]))\
                                              for k, v in x.items()})

cat_combi = cat_vec.cartesian(cat_vec)

cat_cosine = cat_combi.map(lambda x: (x[0][0],x[1][0],\
                                      pairwise_dot_product(x[0][1], x[1][1])))

correl = spark.createDataFrame(cat_cosine, ["Row", "Column", "Cosine"])
correl_pd = correl.toPandas()
correl_pivot = correl_pd.pivot(index='Row',columns='Column',values='Cosine')

"""Output for Task 1"""
top_results.coalesce(1, shuffle = True).saveAsTextFile(accuracy)
columns = ['title_id', 'abstract_id', 'cosine']
sort_result = spark.createDataFrame(rel_sort, schema=columns)
sort_result.repartition(1).write.parquet(results)

samples = rel_sort.filter(lambda x: x[0]!= x[1]).take(5)

sample_dict = []
for sample in samples:
    i, j, k = sample
    title = papers.filter(f.col('id') == i).select('title').first().title
    abstract =  papers.filter(f.col('id') == j).select('abstract').first().abstract
    dict = {'title_id': i, 'title': title, 'abstract_id': j,'abstract': abstract,
                                   'similarity':k}
    sample_dict.append(dict)

samples_pd = pd.DataFrame(sample_dict)
samples_pd.to_csv(processed)

"""Output for Task 2"""

sns.set_theme(style = 'ticks', rc={'figure.dpi': 300})
fig, ax = plt.subplots()
sns.heatmap(correl_pivot, cmap='afmhot_r', ax=ax)
ax.set_title('Categories: Cosine Similarity Matrix', fontweight='bold', fontsize=14)
ax.set_ylabel(None)
ax.set_xlabel(None)
fig.savefig(heatmap, bbox_inches="tight")

sc.stop()





