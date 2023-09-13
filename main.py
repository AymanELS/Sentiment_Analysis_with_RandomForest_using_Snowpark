import sys
import io
import os
import re
import snowflake.snowpark
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.functions import udf, udtf
from snowflake.snowpark.types import IntegerType, StringType, VariantType, DateType, PandasSeries, PandasSeriesType, StructField, StructType
from snowflake.snowpark.functions import table_function
import json


# Create Snowflake Session object
connection_parameters = json.load(open('connection.json'))
session = Session.builder.configs(connection_parameters).create()
session.sql_simplifier_enabled = True

snowflake_environment = session.sql('select current_user(), current_role(), current_database(), current_schema(), current_version(), current_warehouse()').collect()
snowpark_version = VERSION

# Current Environment Details
print('User                        : {}'.format(snowflake_environment[0][0]))
print('Role                        : {}'.format(snowflake_environment[0][1]))
print('Database                    : {}'.format(snowflake_environment[0][2]))
print('Schema                      : {}'.format(snowflake_environment[0][3]))
print('Warehouse                   : {}'.format(snowflake_environment[0][5]))
print('Snowflake version           : {}'.format(snowflake_environment[0][4]))
print('Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_version[2]))


# create file format to ingest training data
session.sql('''
    create or replace file format ff_pipe
        type = CSV
        field_delimiter = '|'
''')

# create the stage for python and model data
session.sql('create or replace stage raw_data').collect()
session.sql('create or replace stage model_data').collect()
session.sql('create or replace stage python_load').collect()
# upload the unstructured file and stop words to the stages
session.file.put('en_core_web_sm.zip','@model_data',overwrite=True)
session.file.put('training_data.txt','@raw_data',auto_compress=False,overwrite=True)

session.file.put('inference_data.txt','@raw_data',auto_compress=False,overwrite=True)

# refresh the stage
session.sql('alter stage raw_data_stage refresh').collect()



# check distribution of the data
session.table("TRAINING_DATA").show(20)

# Use seaborn for visualization
import seaborn as sns

df = session.table('TRAINING_DATA') \
    .group_by(F.col('SENTIMENT')) \
    .agg(F.count(F.col('PRODUCT_ID')).alias('COUNT')).to_pandas()

sns.set(rc={'figure.figsize':(20,8)})
sns.barplot(x='SENTIMENT',y='COUNT',data=df)



from datetime import date
from tokenize import String
from joblib import dump
import zipfile
import pickle
import cachetools
import numpy as np
import pandas as pd

import spacy

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RandomForest
from sklearn.feature_extraction.text import CountVectorizer


# Data Preprocessing: Remove Stopwords, Punctuation, and Lemmatize tokens

session.add_import('@model_data/en_core_web_sm.zip')

@cachetools.cached(cache={})
def load_file(import_dir):
    input_file = import_dir + 'en_core_web_sm.zip'
    output_dir = '/tmp/en_core_web_sm' + str(os.getpid())
            
    with zipfile.ZipFile(input_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        
    return spacy.load(output_dir + "/en_core_web_sm/en_core_web_sm-2.3.0")    



@udf(name='remove_stopwords_vect',packages=['spacy==2.3.5','cachetools'], session=session, is_permanent=True, replace=True, max_batch_size=10000,stage_location='python_load',)
def remove_stopwords_vect(raw_text: PandasSeries[str]) -> PandasSeries[str]:
    nlp = load_file(sys._xoptions['snowflake_import_directory'])
    stop_words = nlp.Defaults.stop_words

    result = []
    
    for s in raw_text:
        doc = nlp(s)
        text = [str(t.lemma_) for t in doc if  
                t not in stop_words 
                and not t.is_punct 
                and not t.is_currency
                and t.lemma_ != '-PRON-']
        text = list(map(lambda x: x.replace(' ', ''), text))
        text = list(map(lambda x: x.replace('\n', ''), text))
        result.append(' '.join(token.lower() for token in text))
    
    return pd.Series(result)


# Create and upload the UDF to binarize the labels (sentiment) of the data
@udf(name='convert_rating',
     is_permanent=True,
     replace=True,
     stage_location='python_load')

def convert_rating(x: str) -> int:
    if x == 'NEGATIVE':
        return -1
    elif x == 'NEUTRAL':
        return 0
    elif x == 'POSITIVE':
        return 1


## Take a look at the new format of the training data after applying the registered UDFs for data preprocessing
df = session.table('TRAINING_DATA') \
    .filter(
        F.col('REVIEWTEXT') != ''
    ) \
    .select( \
        F.col('PRODUCT_ID'),
        F.col('REVIEWDATE'),
        F.call_udf(
            'REMOVE_STOPWORDS_VECT',
            F.col('REVIEWTEXT')).alias('PROCESSED_REVIEW'),
        F.call_udf(
            'CONVERT_RATING',
            F.col('SENTIMENT')).alias('SENTIMENT')).show(20)


### Create the Procedure to train the ML model

def train_sentiment_model(session: snowflake.snowpark.Session) -> float:        
    # build a pd with review data
    df = session.table('TRAINING_DATA') \
        .filter(
            F.col('REVIEWTEXT') != '') \
        .select(
            F.call_udf(
                'REMOVE_STOPWORDS_VECT',
                F.col('REVIEWTEXT')).alias('PROCESSED_TEXT'),
            F.call_udf(
                'CONVERT_RATING',
                F.col('SENTIMENT')).alias('SENTIMENT')).toPandas()
    
    index = df.index
    df['RANDOM'] = np.random.randn(len(index))

    # Split the data into training and testing

    train = df[df['RANDOM'] <= 0.8] # 0.8
    test = df[df['RANDOM'] > 0.8] # 0.8
    
    # vectorize the data
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = vectorizer.fit_transform(train['PROCESSED_TEXT'])
    test_matrix = vectorizer.transform(test['PROCESSED_TEXT'])
    
    # split feature and label 
    x_train = train_matrix
    x_test = test_matrix
    y_train = train['SENTIMENT']
    y_test = test['SENTIMENT']
    
    # Random Forest Model
    lr = RandomForest(n_estimators=1500, random_state=0)
    lr.fit(x_train,y_train)
    predictions = lr.predict(x_test)

    model_output_dir = '/tmp'

    # Save model file
    model_file = os.path.join(model_output_dir, 'model.joblib')
    dump(lr, model_file)
    session.file.put(model_file, "@model_data",overwrite=True)

    # Save vectorizer file
    vect_file = os.path.join(model_output_dir, 'vectorizer.joblib')
    dump(vectorizer, vect_file)
    session.file.put(vect_file, "@model_data",overwrite=True)

    return accuracy_score(y_test, predictions)

# Register the model in UDF to call later for inference
session.sproc.register(name='train_sentiment_model',
                       func=train_sentiment_model, 
                       packages=['snowflake-snowpark-python','pandas','scikit-learn', 'joblib'],
                       replace=True, 
                       is_permanent=True,
                       stage_location='python_load')


# Train the model
session.sql('use warehouse data_science').collect()

df = session.table('TRAINING_DATA') \
    .filter(
        F.col('REVIEWTEXT') != '') \
    .select(
        F.call_udf(
            'REMOVE_STOPWORDS_VECT',
            F.col('REVIEWTEXT')).alias('PROCESSED_TEXT'),
        F.call_udf(
            'CONVERT_RATING',
            F.col('SENTIMENT')).alias('SENTIMENT')).toPandas()

session.call('TRAIN_SENTIMENT_MODEL')

# register the stored procedures in a UDF for batch inference

session.clear_packages()
session.clear_imports()
session.add_import('@MODEL_DATA/model.joblib.gz')
session.add_import('@MODEL_DATA/vectorizer.joblib.gz')

@cachetools.cached(cache={})
def load_model(file_name):
    model_file_path = sys._xoptions.get("snowflake_import_directory") + file_name
    return load(model_file_path)

columns = ('NEGATIVE','NEUTRAL','POSITIVE')
    
@udf(name='predict_sentiment_vect',
     is_permanent=True,
     replace=True,
     stage_location='python_load',
     max_batch_size=1000,
     input_types=[PandasSeriesType(StringType())], 
     return_type=PandasSeriesType(VariantType()),
     packages=['pandas','scikit-learn','cachetools','joblib'])     
def predict_sentiment_vector(sentiment_str):  
    model = load_model('model.joblib.gz')
    vectorizer = load_model('vectorizer.joblib.gz')                            
    
    result = []
    
    for s in sentiment_str:        
        matrix = vectorizer.transform([s])
        
        df = pd.DataFrame(model.predict_proba(matrix),columns=columns)
                
        response = df.loc[0].to_json()
        result.append(json.loads(response))
        
    return pd.Series(result)


# Create new table for inference data and preprocess it

session.table('inference_data').select(
    F.col('product_id'),
    F.col('review_date'),
    F.col('product_review'), 
    F.call_udf(
        'remove_stopwords_vect',
        F.col('product_review')).alias('processed_review')    
).write.save_as_table('inference_data_processed',mode="overwrite", table_type="temporary")


# Run the model on inference data and store the results in new column 'processed_review'
df = session.table('inference_data_processed').select(
    F.col('product_id'),
    F.col('review_date'),
    F.col('product_review'),
    F.col('processed_review'),
    F.call_udf(
        'predict_sentiment_vect',
        F.col('processed_review')).alias('sentiment'))

df = df.select(
    F.col('product_id'),
    F.col('review_date'),
    F.col('product_review'),
    F.col('processed_review'),
    F.col('sentiment')['NEGATIVE'].alias('negative'),
    F.col('sentiment')['NEUTRAL'].alias('neutral'),    
    F.col('sentiment')['POSITIVE'].alias('positive')
).write.save_as_table('inference_data_scored',mode="overwrite")

# view the results
session.table('inference_data_scored').select(
    F.col('product_id'),
    F.col('review_date'),
    F.col('product_review'),  
    F.col('positive'),
    F.col('neutral'),
    F.col('negative')).show(50)

