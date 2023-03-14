"""LAB 2
Name: NATHANIEL NARTEA CASANOVA
Student Number: A0262708B

Python Version: 3.10.9
PySpark: 3.2.1

Packages needed:
matplotlib==3.7.0
nltk==3.7
numpy==1.23.5
pandas==1.5.3
seaborn==0.12.2

"""

#Packages to import
import sys
import re
import math
import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1" #remove warnings for pyarrow

import pyspark.sql.functions as f
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import SparkSession

# Import nltk for lemmatization.
# Download nltk models and frameworks for WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

# Import pandas, matplotlib, and seaborn for visualization
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
arxiv = sys.argv[1] #text data json file
stopwords = sys.argv[2] #stopwords

with open(stopwords, 'r') as file:
    stopwords_list=file.read().split('\n')

"""Output Files Arguments"""

accuracy  = sys.argv[3] #output for accuracy results
results = sys.argv[4] #output for cosine value calculation results
processed = sys.argv[5] #output for processed titles and abstracts
heatmap = sys.argv[6] #output for task 2 heatmap image. must be a .png file

"""DATA PREPROCESSING"""
#Reads json file as a Spark DataFrame
papers =  spark.read.json(arxiv)

# Proceses titles and abstracts in the input file
# Makes all words lower case and replace special characters with a whitespace
process_column = ['abstract', 'title']

for column in process_column:
    papers = papers.withColumn(column,f.lower(f.col(column)))
    papers = papers.withColumn(column,
                               f.regexp_replace(f.col(column), "(\\d|\\W)+", ' '))

# Tokenizes each word in the titles and abstracts
# Removes stop words in the titles and abstracts
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

Return WORDNET part of speech (POS) for compliance to WORDNET lemmatization (a,n,r,v)
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
    #Retrieves POS tag in the list of strings per document
    tagged_words = nltk.pos_tag(data_str)
    #lemmatization of each word in the list of strings
    for word in tagged_words:
        lemma = lmtzr.lemmatize(word[0], get_wordnet_pos(word[1]))
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str

# Wrap lemmatize() into a spark user-defined function (udf)
sparkLemmer = f.udf(lambda x: lemmatize(x), f.StringType())

#Invokes sparkLemmer() to lemmatize abstracts and titles
papers = papers.select('id','categories','filtered_title',
                        sparkLemmer('filtered_abstract').alias('abstract'))
papers = papers.select('id','categories','abstract'
                       ,sparkLemmer('filtered_title').alias('title'))

"""END OF PREPROCESSING"""

# where the real fun starts

"""TASK 1"""
# Convert papers dataframe into an RDD
papers_rdd = papers.rdd

# Count number of documents in the RDD
n = papers_rdd.count()

# Calculates term frequency for each word per abstract
abstracts_rdd = papers_rdd.map(lambda x: (x['id'], x['abstract']))\
    .mapValues(lambda l: re.split(r'[^\w]+',l))\
    .flatMapValues(lambda w: w)\
    .map(lambda w: ((w[0], w[1]),1))

tf_rdd = abstracts_rdd.reduceByKey(lambda a, b: a + b)\
    .map(lambda x: (x[0][1], (x[0][0],x[1])))\

# Calculates documebnt frequency of each word
df_rdd = abstracts_rdd.map(lambda x: (x[0][1], x[0][0]))\
    .distinct()\
    .map(lambda x: (x[0], 1))\
    .reduceByKey(lambda x, y: x + y)

# Join term frequencies and document frequencies RDDs based on document ID
tf_df = tf_rdd.join(df_rdd)

""" Abstract Words TF-IDF Calculation and Normalization

- Calculates TF-IDF of each word w.r.t to each abstract in a document
- Uses scikit-learn's formula for calculating TF-IDF which adds 1 to 
each document frequency to prevent zero divisions
- After calculation, maps tuples to (id, (word, tf-idf))
- Groups each (word, tf-idf) pair to its respective document id to create a
vector of words and their corresponding tf-idf for each abstract
- Converts (word, tf-idf) into a dictionary {word: tf-idf}
- Normalizes tf-idf values w.r.t to each vector of tf-idf values per abstract
- Final tuple structure (id, {abstract_word_1: tf-idf_1...abstract_word_n: tf-idf_n})
"""
tf_idf = tf_df.map(lambda x: ((x[1][0][0],x[0]),(x[1][0][1],x[1][1])))\
                   .mapValues(lambda a: (1+math.log10(a[0]))\
                                           *(math.log10((n+1)/(a[1]+1))+1))\
                    .map(lambda x: (x[0][0], (x[0][1], x[1])))\
                    .groupByKey() \
                    .mapValues(lambda x: dict(x)) \
                    .mapValues(lambda x: {k: v / math.sqrt\
                                          (sum([i**2 for i in x.values()])) for k, v in x.items()})

# Calculates term-frequency for each word in a document title
title_rdd = papers_rdd.map(lambda x: (x['id'], x['title']))\
    .mapValues(lambda l: re.split(r'[^\w]+',l))\
    .flatMapValues(lambda w: w)\
    .map(lambda w: ((w[0], w[1]),1))

tf_t_rdd = title_rdd.reduceByKey(lambda a, b: a + b)\
    .map(lambda x: (x[0][1], (x[0][0],x[1])))

# Join title term frequencies with abstract document frequencies.
# Each title word is the key. 
# For title words not appearing in abstracts corpus, document frequency is mapped to 0
tf_t_df = tf_t_rdd.leftOuterJoin(df_rdd)\
    .mapValues(lambda x: (x[0], 0) if x[1] is None else x)

# Implements TF-IDF calculation and normalization for each document title
# TF-IDF formula takes cares of 0 document frequencies
# Same calculation and mapping as with abstract words
# Final tuple structure (id, {title_word_1: tf-idf_1...title_word_n: tf-idf_n})
tf_t_idf = tf_t_df.map(lambda x: ((x[1][0][0],x[0]),(x[1][0][1],x[1][1])))\
                   .mapValues(lambda a: (1+math.log10(a[0]))\
                                           *(math.log10((n+1)/(a[1]+1))+1))\
                    .map(lambda x: (x[0][0], (x[0][1], x[1])))\
                    .groupByKey() \
                    .mapValues(lambda x: dict(x)) \
                    .mapValues(lambda x: {k: v / math.sqrt\
                                          (sum([i**2 for i in x.values()]))\
                                              for k, v in x.items()})

# Creates all possible combinations of titles vectors and abstract vectors
# Tuple structure will be ((title_id, {title_id: td-idfs}), (abstract_id, {words: td-idfs}))
combinations_rdd = tf_t_idf.cartesian(tf_idf)


""" Cosine Similarity Calculation Function: pairwise_dot_product()
- Function to do pairwise dot product of the dictionary values in a tuple
- Matches each tf-idf in the title and abstract dictionaries using the word as a key
- If a word is not present in either in the title or abstract vector, 0 value is assigned to the word
- Returns dot product
- Magnitude of two vectors not calculated anymore since due to normalization step, the 
product of the magnitude of the two vectors will be 1
"""

def pairwise_dot_product(d1, d2):
    return sum(d1.get(key, 0) * d2.get(key, 0) for key in set(d1) & set(d2))


# Calculates cosine similarity between each title vector and abstract vector pair
# Maps combinations into tuple structure  (title_id, (abstract_id, similarity))
cosine_rdd = combinations_rdd.map(lambda x: (x[0][0],\
                                             (x[1][0], pairwise_dot_product(x[0][1], x[1][1]))))

# For each title_id, retrieves the abstract_id and similarity with the highest similarity value
# Maps resulting tuples into (title_id, abstract_id, similarity) for output
rel_sort = cosine_rdd.reduceByKey(lambda a, b: [a, b][a[1] < b[1]])\
    .map(lambda x: (x[0], x[1][0], x[1][1]))

# Accuracy calculation 
# Set "accuracy" word as key, and title_id and abstract_id pairs as values
# Set tuple structure to (accuracy, value) where value = 1 if title_id = abstract_id, else 0
# Sums all values and divides by total number of documents to get accuracy rate
top_results = rel_sort.map(lambda x: ('accuracy', (x[0], x[1])))\
    .mapValues(lambda a: 1 if a[0] == a[1] else 0)\
    .reduceByKey(lambda u, w: u + w)\
    .mapValues(lambda f: f/n)


"""TASK 2"""

#Calcuates term frequency in each abstract
cat_rdd = papers_rdd.map(lambda x: ((x['id'], x['categories']), x['abstract']))\
    .mapValues(lambda l: re.split(r'[^\w]+',l))\
    .flatMapValues(lambda w: w)\
    .mapValues(lambda w: (w,1))\
    .map(lambda x: ((x[0][0],x[1][0]),(x[0][1],x[1][1])))\
    .reduceByKey(lambda a, b: (a[0], a[1]+b[1]))

# Maps values to key-value pair ((category, word), term frequency)
# Sums all term frequencies for each category-word pair
cattf_rdd = cat_rdd.map(lambda x: ((x[1][0], x[0][1]),x[1][1]))\
    .reduceByKey(lambda a, b: a + b)

# Maps tuple structure to (category, (word, term-frequency sum))
# Then groups all words to its respective cateogry
# Maps the word and term frequency pair to a dictionary
# Normalizes term frequency values in preparation for cosine similarity calcuation
# Maps tuples to structure (category, {category_word_1: tf-idf_1...category_word_n: tf-idf_n})
cat_vec = cattf_rdd.map(lambda x: (x[0][0], (x[0][1], x[1])))\
                .groupByKey()\
                .mapValues(lambda x: dict(x)) \
                .mapValues(lambda x: {k: v / math.sqrt\
                                         (sum([i**2 for i in x.values()]))\
                                              for k, v in x.items()})

#Creates all possible combinations of categories and their corresponding word vectors
cat_combi = cat_vec.cartesian(cat_vec)

# Calculates cosine similarity between each pair of categories by reusing pairwise_dot_product function
# Tuple structure mapped as (category, category, cosine similarity)
cat_cosine = cat_combi.map(lambda x: (x[0][0],x[1][0],\
                                      pairwise_dot_product(x[0][1], x[1][1])))

# Converts cat_cosine RDD to spark dataframe
# Converts correl spark dataframe to pandas dataframe
# Pivots pandas dataframe to create a similariy matrix between categories
correl = spark.createDataFrame(cat_cosine, ["Row", "Column", "Cosine"])
correl_pd = correl.toPandas()
correl_pivot = correl_pd.pivot(index='Row',columns='Column',values='Cosine')

"""Output for Task 1"""
# Output accuracy score as a single text file
top_results.coalesce(1, shuffle = True).saveAsTextFile(accuracy)

# Output title_id, abstract_id pairs with corresponding similarity for further analysis
# Output will be a parquet file
columns = ['title_id', 'abstract_id', 'cosine']
sort_result = spark.createDataFrame(rel_sort, schema=columns)
sort_result.repartition(1).write.parquet(results)

# Takes a sample of 5 incorrect title and abstract pairings
# Outputs the samples as a csv for analysis
samples = rel_sort.filter(lambda x: x[0]!= x[1]).take(5)

sample_dict = []
for sample in samples:
    i, j, k = sample
    title = papers.filter(f.col('id') == i).select('title').first().title
    abstract_w =  papers.filter(f.col('id') == j).select('abstract').first().abstract
    abstract_c =  papers.filter(f.col('id') == i).select('abstract').first().abstract
    rows = {'title_id': i, 'title': title, 'abstract_wrong_id': j,
            'abstract_wrong': abstract_w,
            'abstract_correct': abstract_c,
            'similarity':k}
    sample_dict.append(rows)

samples_pd = pd.DataFrame(sample_dict)
samples_pd.to_csv(processed)

"""Output for Task 2"""
# Output heatmap image for the category similarity matrix 
sns.set_theme(style = 'ticks', rc={'figure.dpi': 300})
fig, ax = plt.subplots()
sns.heatmap(correl_pivot, cmap='afmhot_r', ax=ax)
ax.set_title('Categories: Cosine Similarity Matrix', fontweight='bold', fontsize=14)
ax.set_ylabel(None)
ax.set_xlabel(None)
fig.savefig(heatmap, bbox_inches="tight")

sc.stop()