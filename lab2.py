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
    .setMaster("local[4]")\
    .set("spark.executor.memory", "4g")\
    .set("spark.driver.memory", "2g")

sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

arxiv = sys.argv[1] #1st argument
stopwords = sys.argv[2] #2nd argument

with open(stopwords, 'r') as file:
    stopwords_list=file.read().split('\n')

output  = sys.argv[3]

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
    .reduceByKey(lambda x, y: x + y)\
    .mapValues(lambda a: (a+1))

tf_df = tf_rdd.join(df_rdd)

tf_idf = tf_df.map(lambda x: ((x[1][0][0],x[0]),(x[1][0][1],x[1][1])))\
                   .mapValues(lambda a: (1+math.log10(a[0]))\
                                           *(math.log10(n/a[1])))\
                    .map(lambda x: (x[0][0], (x[0][1], x[1])))\
                    .groupByKey() \
                    .mapValues(lambda x: dict(x)) \
                    .mapValues(lambda x: {k: v / math.sqrt\
                                          (sum([i**2 for i in x.values()])) for k, v in x.items()}) \
#                    .flatMap(lambda x: [((x[0], w), tfidf) for w, tfidf in x[1].items()])

title_rdd = papers_rdd.map(lambda x: (x['id'], x['title']))\
    .mapValues(lambda l: re.split(r'[^\w]+',l))\
    .flatMapValues(lambda w: w)\
    .map(lambda w: ((w[0], w[1]),1))

tf_t_rdd = title_rdd.reduceByKey(lambda a, b: a + b)\
    .map(lambda x: (x[0][1], (x[0][0],x[1])))

tf_t_df = tf_t_rdd.leftOuterJoin(df_rdd)\
    .mapValues(lambda x: (x[0], 1) if x[1] is None else x)

tf_t_idf = tf_t_df.map(lambda x: ((x[1][0][0],x[0]),(x[1][0][1],x[1][1])))\
                   .mapValues(lambda a: (1+math.log10(a[0]))\
                                           *(math.log10(n/a[1])))\
                    .map(lambda x: (x[0][0], (x[0][1], x[1])))\
                    .groupByKey() \
                    .mapValues(lambda x: dict(x)) \
                    .mapValues(lambda x: {k: v / math.sqrt\
                                          (sum([i**2 for i in x.values()])) for k, v in x.items()}) \
#                    .flatMap(lambda x: [((x[0], w), tfidf) for w, tfidf in x[1].items()])
def pairwise_dot_product(d1, d2):
    return sum(d1.get(key, 0) * d2.get(key, 0) for key in set(d1) & set(d2))

key_title = tf_t_idf.keys()
key_abstract = tf_idf.keys()

combinations_rdd = tf_t_idf.cartesian(tf_idf)
cosine_rdd = combinations_rdd.map(lambda x: (x[0][0],\
                                             (x[1][0], pairwise_dot_product(x[0][1], x[1][1]))))

rel_sort = cosine_rdd.groupByKey().\
    mapValues(lambda x: sorted(x, key=lambda y: y[1], reverse=True))

top_results = rel_sort.map(lambda x: (x[0], (x[1][0][0], x[1][0][1])))


"""Task 2"""

cat_rdd = papers_rdd.map(lambda x: ((x['id'], x['categories']), x['abstract']))\
    .mapValues(lambda l: re.split(r'[^\w]+',l))\
    .flatMapValues(lambda w: w)\
    .mapValues(lambda w: (w,1))\
    .map(lambda x: ((x[0][0],x[1][0]),(x[0][1],x[1][1])))\
    .reduceByKey(lambda a, b: (a[0], a[1]+b[1]))

cattf_rdd = cat_rdd.map(lambda x: ((x[1][0], x[0][1]),x[1][1]))\
    .reduceByKey(lambda a, b: a + b)

for row in top_results.take(5):
     print(row)

top_results.saveAsTextFile(output)

sc.stop()