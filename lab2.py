import sys
import pyspark.sql.functions as f
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import SparkSession
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder\
    .getOrCreate()

arxiv = sys.argv[1] #1st argument
stopwords = sys.argv[2] #2nd argument

with open(stopwords, 'r') as file:
    stopwords_list=file.read().split('\n')

output  = sys.argv[3]

papers =  spark.read.json(arxiv)
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

papers.printSchema()
papers.show()