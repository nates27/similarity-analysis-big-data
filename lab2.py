import sys
import re
import pyspark.sql.functions as f
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import SparkSession
import nltk
import math
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer


conf = SparkConf()\
    .setMaster("local[*]")\
    .set("spark.executor.memory", "4g")\
    .set("spark.driver.memory", "2g")

sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

arxiv = sys.argv[1] #1st argument
stopwords = sys.argv[2] #2nd argument

with open(stopwords, 'r') as file:
    stopwords_list=file.read().split('\n')

accuracy  = sys.argv[3]
results = sys.argv[4]

papers =  spark.read.json(arxiv)

"""Data Preprocessing"""
process_column = ['abstract', 'title']

for column in process_column:
    papers = papers.withColumn(column,f.lower(f.col(column)))
    papers = papers.withColumn(column,
                               f.regexp_replace(f.col(column), "(\\d|\\W)+", ' '))

for column in process_column:
    regexTokenizer = RegexTokenizer(inputCol=column, outputCol='words_'+column,\
                                     pattern="\\W")
    tokenized = regexTokenizer.transform(papers)
    stopwordsRemover = StopWordsRemover(inputCol='words_'+column,
                                         outputCol='filtered_'+column,
                                           stopWords=stopwords_list)
    papers = stopwordsRemover.transform(tokenized)
    papers = papers.withColumn(column,f.concat_ws(' ', f.col('filtered_'+column)))

papers = papers.withColumn('categories',f.lower(f.col('categories')))
papers = papers.withColumn('categories',
                               f.regexp_replace(f.col('categories'), "\s+", ""))

def get_wordnet_pos(treebank_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
    if treebank_tag.startswith('J'):
            return 'a'
    elif treebank_tag.startswith('V'):
            return 'v'
    elif treebank_tag.startswith('N'):
            return 'n'
    elif treebank_tag.startswith('R'):
            return 'r'
    else:
    # As default pos in lemmatization is Noun
        return 'n'

def lemmatize1(data_str):
    # expects a string
    list_pos = 0
    cleaned_str = ''
    lmtzr = WordNetLemmatizer()
    #text = data_str.split()
    tagged_words = nltk.pos_tag(data_str)
    for word in tagged_words:
        lemma = lmtzr.lemmatize(word[0], get_wordnet_pos(word[1]))
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str

sparkLemmer1 = f.udf(lambda x: lemmatize1(x), f.StringType())

papers = papers.select('id','categories','filtered_title',
                        sparkLemmer1('filtered_abstract').alias('abstract'))
papers = papers.select('id','categories','abstract'
                       ,sparkLemmer1('filtered_title').alias('title'))

# papers.printSchema()
# papers.show()

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
                                          (sum([i**2 for i in x.values()])) for k, v in x.items()}) \

def pairwise_dot_product(d1, d2):
    return sum(d1.get(key, 0) * d2.get(key, 0) for key in set(d1) & set(d2))

combinations_rdd = tf_t_idf.cartesian(tf_idf)
cosine_rdd = combinations_rdd.map(lambda x: (x[0][0],\
                                             (x[1][0], pairwise_dot_product(x[0][1], x[1][1]))))

rel_sort = cosine_rdd.groupByKey()\
    .mapValues(lambda x: sorted(x, key=lambda y: y[1], reverse=True))\
    .map(lambda x: (x[0], x[1][0][0], x[1][0][1]))

top_results = rel_sort.map(lambda x: ('accuracy', (x[0], x[1])))\
    .mapValues(lambda a: 1 if a[0] == a[1] else 0)\
    .reduceByKey(lambda u, w: u + w)\
    .mapValues(lambda f: f/n)

top_results.coalesce(1, shuffle = True).saveAsTextFile(accuracy)
columns = ['title_id', 'abstract_id', 'cosine']
sort_result = spark.createDataFrame(rel_sort, schema=columns)
sort_result.repartition(1).write.parquet(results)

"""Task 2"""

cat_rdd = papers_rdd.map(lambda x: ((x['id'], x['categories']), x['abstract']))\
    .mapValues(lambda l: re.split(r'[^\w]+',l))\
    .flatMapValues(lambda w: w)\
    .mapValues(lambda w: (w,1))\
    .map(lambda x: ((x[0][0],x[1][0]),(x[0][1],x[1][1])))\
    .reduceByKey(lambda a, b: (a[0], a[1]+b[1]))

cattf_rdd = cat_rdd.map(lambda x: ((x[1][0], x[0][1]),x[1][1]))\
    .reduceByKey(lambda a, b: a + b)

# for row in top_results.take(15):
#      print(row)

#rel_sort.saveAsTextFile(output)


sc.stop()