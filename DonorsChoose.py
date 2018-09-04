import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# from nltk import sent_tokenize, word_tokenize
# nltk.download('stopwords')
# from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold

# Sentiment Analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Light GBM
import lightgbm as lgb

# Folder containing data
DATA_PATH = 'C:/Users/Dave/Google Drive/Data Science Training/Python Scripts/Donors Choose/'

# Definitions


def process_timestamp(df):
    df['year'] = df['project_submitted_datetime'].\
        apply(lambda x: int(x.split('-')[0]))

    df['month'] = df['project_submitted_datetime'].\
        apply(lambda x: int(x.split('-')[1]))

    df['day_of_week'] = pd.to_datetime(df['project_submitted_datetime']
                                       ).dt.weekday

    df['hour'] = df['project_submitted_datetime'].\
        apply(lambda x: int(x.split(' ')[-1].split(':')[0]))

    df['minute'] = df['project_submitted_datetime'].\
        apply(lambda x: int(x.split(' ')[-1].split(':')[1]))

    df['project_submitted_datetime'] =\
        pd.to_datetime(df['project_submitted_datetime']).\
        values.astype(np.int64)
    return df


def extract_features(df):
    df['project_title_len'] = df['project_title'].apply(lambda x: len(str(x)))
    df['project_essay_1_len'] = df['project_essay_1'].\
        apply(lambda x: len(str(x)))
    df['project_essay_2_len'] = df['project_essay_2'].\
        apply(lambda x: len(str(x)))
    df['project_essay_3_len'] = df['project_essay_3'].\
        apply(lambda x: len(str(x)))
    df['project_essay_4_len'] = df['project_essay_4'].\
        apply(lambda x: len(str(x)))
    df['project_resource_summary_len'] = df['project_resource_summary'].\
        apply(lambda x: len(str(x)))

    df['project_title_wc'] = df['project_title'].\
        apply(lambda x: len(str(x).split(' ')))
    df['project_essay_1_wc'] = df['project_essay_1'].\
        apply(lambda x: len(str(x).split(' ')))
    df['project_essay_2_wc'] = df['project_essay_2'].\
        apply(lambda x: len(str(x).split(' ')))
    df['project_essay_3_wc'] = df['project_essay_3'].\
        apply(lambda x: len(str(x).split(' ')))
    df['project_essay_4_wc'] = df['project_essay_4'].\
        apply(lambda x: len(str(x).split(' ')))
    df['project_resource_summary_wc'] = df['project_resource_summary'].\
        apply(lambda x: len(str(x).split(' ')))
    return df


def get_polarity(text):
    textblob = TextBlob(text)
    pol = round(textblob.sentiment.polarity, 2)
    return pol


def get_subjectivity(text):
    textblob = TextBlob(text)
    subj = round(textblob.sentiment.subjectivity, 2)
    return subj


def get_vader_polarity(text):
    analyzer = SentimentIntensityAnalyzer()
    pol = analyzer.polarity_scores(text)
    pol = list(pol.values())
    return pol


def count_characters(df):
    KeyChars = ['!', '\?', '@', '#', '\$', '%', '&', '\*', '\(', '\[',
                '\{', '\|', '-', '_', '=', '\+', '\.', ':', ';', ',', '/',
                '\\\\r', '\\\\t', '\\"', '\.\.\.', 'etc', 'http']
    for c in KeyChars:
        df['n_' + c] = df['text'].\
        apply(lambda x: len(re.findall(c, x.lower())))
    return df

# Desired data types to read in training csv


dtypes = {
    'id': 'str',
    'project_id': 'str',
    'project_title': 'str',
    'project_grade_category': 'str',
    'project_subject_categories': 'str',
    'school_state': 'str',
    'project_subject_subcategories': 'str',
    'project_resource_summary': 'str',
    'project_essay_1': 'str',
    'project_essay_2': 'str',
    'project_essay_3': 'str',
    'project_essay_4': 'str',
    'project_submitted_datetime': 'str',
    'teacher_id': 'str',
    'teacher_prefix': 'str',
    'teacher_number_of_previously_posted_projects': 'uint64',
    'project_is_approved': 'uint8',
    }

print('Read Data...')

df_train = pd.read_csv(DATA_PATH + 'train.csv', dtype=dtypes)
df_test = pd.read_csv(DATA_PATH + 'test.csv')
df_all = pd.concat([df_train, df_test], axis=0)

dtypes_resources = {
    'id': 'str',
    'description': 'str',
    'quantity': 'uint32',
    'price': 'float32'
    }

resouces = pd.read_csv(DATA_PATH + 'resources.csv',
                       usecols=['id', 'quantity', 'price'],
                       dtype=dtypes_resources)


resouces = pd.DataFrame(resouces.groupby('id').agg(
    {
        'quantity': [
            'sum',
            'min',
            'max',
            'mean',
            'std',
            # lambda x: len(np.unique(x)),
        ],
        'price': [
            'count',
            'sum',
            'min',
            'max',
            'mean',
            'std',
            lambda x: len(np.unique(x)),
        ]}
    )).reset_index()

resouces.columns = ['_'.join(col) for col in resouces.columns]
resouces.rename(columns={'id_': 'id'}, inplace=True)
resouces['mean_price'] = resouces['price_sum'] / resouces['quantity_sum']

assert((df_train.shape[0] + df_test.shape[0]) == resouces.shape[0])
resouces.head()

# Join resources with training / test
df_train = pd.merge(df_train, resouces, on='id', how='left')
df_test = pd.merge(df_test, resouces, on='id', how='left')

del resouces
gc.collect()

# Preprocess categorical columns
print('Preprocess Data...')

cols = [
    'teacher_id',
    'teacher_prefix',
    'school_state',
    'project_grade_category',
    'project_subject_categories',
    'project_subject_subcategories'
]

for column in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_train[column].astype(str))
    df_train[column] = le.transform(df_train[column].astype(str))

for column in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_test[column].astype(str))
    df_test[column] = le.transform(df_test[column].astype(str))

del le
gc.collect()

# Preprocess timestamp

df_train = process_timestamp(df_train)
df_test = process_timestamp(df_test)

# Extract featuretures from the text: Charcter legth and word count
# of each essay

df_train = extract_features(df_train)
df_test = extract_features(df_test)

# Combine text fileds into single string

df_train['text'] = df_train.apply(lambda row: ''.join([
    str(row['project_title']), ' ',
    str(row['project_resource_summary']), ' ',
    str(row['project_essay_1']), ' ',
    str(row['project_essay_2']), ' ',
    str(row['project_essay_3']), ' ',
    str(row['project_essay_4'])
    ]), axis=1)

df_test['text'] = df_test.apply(lambda row: ''.join([
    str(row['project_title']), ' ',
    str(row['project_resource_summary']), ' ',
    str(row['project_essay_1']), ' ',
    str(row['project_essay_2']), ' ',
    str(row['project_essay_3']), ' ',
    str(row['project_essay_4'])
    ]), axis=1)

# Extract featuretures from the text: count of Key Characters

df_train = count_characters(df_train)
df_test = count_characters(df_test)

# Extract text polarity and subjectivity using TextBlob
print('Extracting text sentiment: TextBlob...')

df_train['text_polarity_TB'] = df_train.text.apply(get_polarity)
df_train['text_subj_TB'] = df_train.text.apply(get_subjectivity)

df_test['text_polarity'] = df_test.text.apply(get_polarity)
df_test['text_subj'] = df_test.text.apply(get_subjectivity)

# Extract  text polarity using Vader Sentiment analysis
print('Extracting text sentiment: VADER...')

df_train['text_polarity_Vader'] = df_train.text.apply(get_vader_polarity)
df_test['text_polarity_Vader'] = df_test.text.apply(get_vader_polarity)

df_train[['tp_vader_compond', 'tp_vader_neg', 'tp_vader_neu', 'tp_vader_pos']]\
        = pd.DataFrame(df_train.text_polarity_Vader.values.tolist(),
                       index=df_train.index)
df_test[['tp_vader_compond', 'tp_vader_neg', 'tp_vader_neu', 'tp_vader_pos']]\
        = pd.DataFrame(df_test.text_polarity_Vader.values.tolist(),
                       index=df_test.index)
df_train.head()

# Clean up dataframe
print('Cleaning Up...')

df_train.drop([
    'project_title',
    'project_resource_summary',
    'project_essay_1',
    'project_essay_2',
    'project_essay_3',
    'project_essay_4',
    'text_polarity_Vader'], axis=1, inplace=True)
df_test.drop([
    'project_title',
    'project_resource_summary',
    'project_essay_1',
    'project_essay_2',
    'project_essay_3',
    'project_essay_4',
    'text_polarity_Vader'], axis=1, inplace=True)

df_all = pd.concat([df_train, df_test], axis=0)

gc.collect()


# Text Vecorizer #
print('Vectorize Text...')

n_features = 1000

# tfidf = TfidfVectorizer(
#         max_features=n_features,
#         norm='l2',
#         )

p = PorterStemmer()


def wordPreProcess(text):
    return ' '.join([p.stem(x.lower()) for x in re.split('\W', text)
                     if len(x) >= 1])


count_vec = CountVectorizer(stop_words=None,
                            preprocessor=wordPreProcess,
                            max_features=n_features,
                            binary=True,
                            ngram_range=(1, 2))


count_vec.fit(df_all.text)

count_vec_train = np.array(count_vec.transform(df_train.text).
                           toarray(), dtype=np.float16)
count_vec_test = np.array(count_vec.transform(df_test.text).
                          toarray(), dtype=np.float16)

for i in range(n_features):
    df_train['count_vec_' + str(i)] = count_vec_train[:, i]
    df_test['count_vec_' + str(i)] = count_vec_test[:, i]

del count_vec_train, count_vec_test
gc.collect()

# Set up training and test sets
# df_train = pd.read_csv('C:/Users/Dave/Google Drive/Data Science Training/Python Scripts/Donors Choose/train_preprocessed2.csv', encoding = "ISO-8859-1")
# df_test = pd.read_csv('C:/Users/Dave/Google Drive/Data Science Training/Python Scripts/Donors Choose/test_preprocessed2.csv', encoding = "ISO-8859-1")

df_train.to_csv('C:/Users/Dave/Google Drive/Data Science Training/Python Scripts/Donors Choose/train_preprocessed_vec.csv')
df_test.to_csv('C:/Users/Dave/Google Drive/Data Science Training/Python Scripts/Donors Choose/test_preprocessed_vec.csv')

cols_to_drop = [
    'id',
    'teacher_id',
    'text',
    'project_is_approved'
    ]

X = df_train.drop(cols_to_drop, axis=1, errors='ignore')
y = df_train['project_is_approved']

X_test = df_test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = df_test['id'].values
feature_names = list(X.columns)

print('Light GBM...')
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
model_dict = defaultdict()
kf = RepeatedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=0)
auc_buf = []

for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 14,
        'num_leaves': 31,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0,
        'histogram_pool_size': 512
    }

    lgb_train = lgb.Dataset(
        X.loc[train_index],
        y.loc[train_index],
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X.loc[valid_index],
        y.loc[valid_index],
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance),
                        key=lambda x: x[1])[:: -1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(60):
            if i < len(tuples):
                print(tuples[i])
            else:
                break

        del importance, model_fnames, tuples

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    auc = roc_auc_score(y.loc[valid_index], p)

    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    auc_buf.append(auc)

    # Save individual models for comparison later
    model_dict[cnt] = model

    cnt += 1
    # if cnt > 0: # Comment this to run several folds
    #     break

    del model, lgb_train, lgb_valid, p
    gc.collect


auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))

preds = p_buf/cnt

subm = pd.DataFrame()
subm['id'] = id_test
subm['project_is_approved'] = preds
subm.to_csv('submission_lightGBM_COLAB.csv', index=False)

# SHAP model analysis
import shap

shap.initjs()

explainer = shap.TreeExplainer(model_dict[0])
shap_values = explainer.shap_values(X)

fig = shap.summary_plot(shap_values, X)

gc.collect()

# Print word from count_vec id
i = 0
for word, _ in sorted(count_vec.vocabulary_.items()):
    if i == 821:
        print(word)
    i += 1
