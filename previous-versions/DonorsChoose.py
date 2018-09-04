# Inital EDA and modeling

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import nltk
from tqdm import tqdm
import gc

from nltk import sent_tokenize, word_tokenize
# nltk.download('stopwords')
from nltk.corpus import stopwords
from string import punctuation

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedKFold
# from sklearn.decomposition import TruncatedSVD, PCA
# from sklearn.manifold import TSNE

# textblob
from textblob import TextBlob

# Light GBM
import lightgbm as lgb

plt.style.use('seaborn')

DATA_DIR = 'C:/Users/Dave/Google Drive/Data Science Training/Python Scripts/Donors Choose/'

# Definitions


def process_timestamp(df):
    df['year'] = df['project_submitted_datetime'].\
                    apply(lambda x: int(x.split('-')[0]))

    df['month'] = df['project_submitted_datetime'].apply(lambda x:
                                                         int(x.split('-')[1]))
    df['day_of_week'] =\
        pd.to_datetime(df['project_submitted_datetime']).dt.weekday

    df['hour'] = df['project_submitted_datetime'].apply(lambda x:
                                                        int(x.split(' ')[-1].
                                                        split(':')[0]))

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


def tokenizer(text):

    tokens = []

    # Break up sentence into individual word tokens
    tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]
    for w in tokens_:
        tokens += w

    # Remove stop words and punctuation
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    tokens = [w for w in tokens if w not in punctuation]
    tokens = [w for w in tokens if w not in [u"the", u"The", u"'s", u"n't",
                                             u"...", u"''", u"``", u"'",
                                             u"\u2014", u"\u2026", u"\u2013"]]

    # Remove NaN from essays
    tokens = [w for w in tokens if w not in [u"nan"]]

    # All lowercase
    tokens = list(map(lambda w: w.lower(), tokens))

    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    return(tokens)

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

# Used later on pre-rpcessed data
dtypes_preprocessed = {
    'id': 'str',
    'teacher_id': 'uint64',
    'teacher_prefix': 'uint64',
    'school_state': 'uint8',
    'project_submitted_datetime': 'uint64',
    'project_grade_category': 'uint64',
    'project_subject_categories': 'uint64',
    'project_subject_subcategories': 'uint64',
    'teacher_number_of_previously_posted_projects': 'uint64',
    'quantity_sum': 'uint64',
    'quantity_min': 'uint64',
    'quantity_max': 'uint64',
    'quantity_mean': 'float64',
    'quantity_std': 'float64',
    'price_count': 'uint64',
    'price_sum': 'float64',
    'price_min': 'float64',
    'price_max': 'float64',
    'price_mean': 'float64',
    'price_std': 'float64',
    'price_<lambda>': 'uint64',
    'mean_price': 'float64',
    'year': 'uint64',
    'month': 'uint64',
    'day_of_week': 'uint64',
    'hour': 'uint64',
    'minute': 'uint64',
    'project_title_len': 'uint64',
    'project_essay_1_len': 'uint64',
    'project_essay_2_len': 'uint64',
    'project_essay_3_len': 'uint64',
    'project_essay_4_len': 'uint64',
    'project_resource_summary_len': 'uint64',
    'project_title_wc': 'uint64',
    'project_essay_1_wc': 'uint64',
    'project_essay_2_wc': 'uint64',
    'project_essay_3_wc': 'uint64',
    'project_essay_4_wc': 'uint64',
    'project_resource_summary_wc': 'uint64',
    'text': 'str',
    'text_polarity': 'float64',
    'text_subj': 'float64'
    }

df_train = pd.read_csv(DATA_DIR + 'train.csv', dtype=dtypes)
df_test = pd.read_csv(DATA_DIR + 'test.csv')
df_all = pd.concat([df_train, df_test], axis=0)

dtypes_resources = {
    'id': 'str',
    'description': 'str',
    'quantity': 'uint64',
    'price': 'float64'
    }

resouces = pd.read_csv(DATA_DIR + 'resources.csv',
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

# Join resources with training
df_train = pd.merge(df_train, resouces, on='id', how='left')
df_test = pd.merge(df_test, resouces, on='id', how='left')

del resouces
gc.collect()

# EDA #

df_train.head()
df_train.info()

# project_grade_category
fig, ax = plt.subplots()

project_grade_categories = df_train.project_grade_category.value_counts()

project_grade_categories.index
project_grade_categories.values
ax.bar(project_grade_categories.index,
       project_grade_categories.values)
ax.set_ylabel('Number of Submissions')
ax.set_title('Distribution of Grades')
plt.show()


# project_subject_categories
# Can be a single category, or multiple categores separated by commas

df_train.project_subject_categories.describe()
print(df_train.project_subject_categories.unique())

project_subject_categories_counts =\
    df_train.project_subject_categories.value_counts()
print(project_subject_categories_counts)

categories = ['Applied Learning',
              'Care & Hunger',
              'Health & Sports',
              'History & Civics',
              'Literacy & Language',
              'Math & Science',
              'Music & The Arts',
              'Special Needs',
              'Warmth']

project_subject_categories_df =\
    pd.DataFrame(columns=['Total Instances'], index=categories)

for category in categories:
    num = sum(df_train.project_subject_categories.str.contains(category))
    project_subject_categories_df['Total Instances'][category] = num

fig, ax = plt.subplots()
ax = project_subject_categories_df.sort_values('Total Instances',
                                               ascending=False).\
                                    plot(kind='bar',
                                         legend=False,
                                         position=-1)
ax.set_ylabel("Total Instances", fontsize=12)
ax.set_title('Intances of Each Category in Project Subjects', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
# plt.savefig('example.png', bbox_inches='tight')
plt.close(fig)
plt.show()

# Project price mean

# Price percentile vs percentage accpeted
n_bins = 100
groups = df_train.project_is_approved.\
                  groupby(pd.qcut(df_train.mean_price, n_bins)).agg('mean')

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(df_train.price_mean, bins='auto')
ax1.set_xlim(0, 800)
ax1.set_xlabel('Price, $')
ax1.set_ylabel('Number of Applications')
ax1.set_title('Distribution of Project Costs')
ax2.scatter(np.arange(0, n_bins), groups.values, alpha=0.7)
ax2.axhline(df_train.project_is_approved.mean(), color='k', alpha=0.8)
ax2.text(n_bins*0.25, 0.85, 'Mean Success Rate', color='k', style='italic')
ax2.set_xlabel('Price Bin')
ax2.set_ylabel('Succes Rate')
ax2.set_title('Price Bin  vs Probability of Success')
fig.suptitle('Project Costs', fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig('files/ProjectCosts.png')
plt.show()

# States

state_counts = pd.DataFrame(df_train.school_state.value_counts())
print(state_counts)

# Read in state data for context (avail;able from US census)
state_data = pd.read_csv(DATA_DIR + 'state_census_data.csv')
state_data = pd.merge(state_data, state_counts,
                      left_on='state_code',
                      right_index=True)
state_data['app_to_pop_ratio'] =\
    (state_data.school_state / state_data.population_2017) * 100
state_data.head(2)

# Ratio of applications to state population
rank = pd.DataFrame(state_data.app_to_pop_ratio.values,
                    index=state_data.state_code,
                    columns=['app_to_pop_ratio'])
rank = rank.sort_values(by='app_to_pop_ratio', axis=0, ascending=False)
rank.head(5)
rank.tail(5)

# For annotations
VT_pop = state_data.loc[state_data.state_code == 'VT', 'population_2017'].\
                        values.item(0)
VT_app = state_data.loc[state_data.state_code == 'VT', 'school_state'].\
                        values.item(0)

DC_pop = state_data.loc[state_data.state_code == 'DC', 'population_2017'].\
                        values.item(0)
DC_app = state_data.loc[state_data.state_code == 'DC', 'school_state'].\
                        values.item(0)

SC_pop = state_data.loc[state_data.state_code == 'SC', 'population_2017'].\
                        values.item(0)
SC_app = state_data.loc[state_data.state_code == 'SC', 'school_state'].\
                        values.item(0)

CA_pop = state_data.loc[state_data.state_code == 'CA', 'population_2017'].\
                        values.item(0)
CA_app = state_data.loc[state_data.state_code == 'CA', 'school_state'].\
                        values.item(0)

TX_pop = state_data.loc[state_data.state_code == 'TX', 'population_2017'].\
                        values.item(0)
TX_app = state_data.loc[state_data.state_code == 'TX', 'school_state'].\
                        values.item(0)

fig3, ax3 = plt.subplots()
ax3 = sns.regplot('population_2017', 'school_state', data=state_data)
ax3.set_xlabel('State Population')
ax3.set_ylabel('Number of Applications')
ax3.set_title('Number of Applications vs State Population')
ax3.annotate('Vermont',
             xy=(VT_pop, VT_app),
             xytext=(0.5, 0.1),
             textcoords='axes fraction',
             horizontalalignment="center",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             bbox=dict(boxstyle="round", fc="w"))
ax3.annotate('DC',
             xy=(DC_pop, DC_app),
             xytext=(0.1, 0.6),
             textcoords='axes fraction',
             horizontalalignment="center",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             bbox=dict(boxstyle="round", fc="w"))
ax3.annotate('South\nCarolina',
             xy=(SC_pop, SC_app),
             xytext=(0.3, 0.8),
             textcoords='axes fraction',
             horizontalalignment="center",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             bbox=dict(boxstyle="round", fc="w"))
ax3.annotate('California',
             xy=(CA_pop, CA_app),
             xytext=(0.7, 0.9),
             textcoords='axes fraction',
             horizontalalignment="center",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             bbox=dict(boxstyle="round", fc="w"))
ax3.annotate('Texas',
             xy=(TX_pop, TX_app),
             xytext=(0.7, 0.2),
             textcoords='axes fraction',
             horizontalalignment="center",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             bbox=dict(boxstyle="round", fc="w"))
plt.show()

# State income vs Aplications
fig3, ax3 = plt.subplots()
ax3 = sns.regplot('income_2015', 'app_to_pop_ratio',
                  data=state_data, ci=0, fit_reg=False)
ax3.set_xlabel('Median State Income, 2015', fontsize=16)
ax3.set_ylabel('Rate of Applications', fontsize=16)
ax3.set_title('Rate of Applications vs State Income', fontsize=16)
plt.savefig('files/income_vs_application.png', bbox_inches='tight')
plt.show()

income = pd.DataFrame(np.array(state_data.income_2015),
                      index=state_data.state_code,
                      columns=['income_2015'])
income = income.sort_values(by='income_2015', axis=0, ascending=False)
income.head(10)
income.tail(10)

df_train.head()


# Number of applications

df_train_success = df_train.loc[df_train.project_is_approved == 1]
df_train_unsuccess = df_train.loc[df_train.project_is_approved == 0]

df_train.teacher_number_of_previously_posted_projects.describe()

df_train_success.teacher_number_of_previously_posted_projects.describe()
df_train_unsuccess.teacher_number_of_previously_posted_projects.describe()


# Preprocessing #

# Preprocess categorical columns

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

# Pre-process text
# 1) Extract featuretures from the text: Charagter legth and word count
# of each essy

df_train = extract_features(df_train)
df_test = extract_features(df_test)

# 2) Combine text fileds into single string

df_train['text'] = df_train.apply(lambda row: ''.join([
    str(row['project_title']), ' ',
    str(row['project_resource_summary']), ' ',
    str(row['project_essay_1']), ' ',
    str(row['project_essay_2']), ' ',
    str(row['project_essay_3']), ' ',
    str(row['project_essay_4'])
    ]), axis=1)

# df_train['essay_token'] =\
#          df_train.apply(lambda row: tokenizer(row['essay']), axis=1)

df_test['text'] = df_test.apply(lambda row: ''.join([
    str(row['project_title']), ' ',
    str(row['project_resource_summary']), ' ',
    str(row['project_essay_1']), ' ',
    str(row['project_essay_2']), ' ',
    str(row['project_essay_3']), ' ',
    str(row['project_essay_4'])
    ]), axis=1)

# pol_sub = df_train.text.apply(get_pol_subj)

df_train['text_polarity'] = df_train.text.apply(get_polarity)
df_train['text_subj'] = df_train.text.apply(get_subjectivity)

df_test['text_polarity'] = df_test.text.apply(get_polarity)
df_test['text_subj'] = df_test.text.apply(get_subjectivity)

df_train.drop([
    'project_title',
    'project_resource_summary',
    'project_essay_1',
    'project_essay_2',
    'project_essay_3',
    'project_essay_4'], axis=1, inplace=True)
df_test.drop([
    'project_title',
    'project_resource_summary',
    'project_essay_1',
    'project_essay_2',
    'project_essay_3',
    'project_essay_4'], axis=1, inplace=True)

# Savw pre-processed data

# df_train.to_csv('train_preprocessed.csv')
# df_test.to_csv('test_preprocessed.csv')

df_train = pd.read_csv('train_preprocessed.csv',
                       dtype=dtypes_preprocessed,
                       encoding='ISO-8859-1')
df_test = pd.read_csv('test_preprocessed.csv',
                      dtype=dtypes_preprocessed,
                      encoding='ISO-8859-1')

df_all = pd.concat([df_train, df_test], axis=0)


gc.collect()

# Review pre-processed dataframes
df_train.head()
df_train.shape
df_test.shape

print(df_train.columns)
print(df_test.columns)

# Create Model #

# Tfidf

n_features = 4000

tfidf = TfidfVectorizer(
        max_features=n_features,
        norm='l2',
        )

tfidf.fit(df_all.text)

tfidf_train = np.array(tfidf.transform(df_train.text).toarray(),
                       dtype=np.float16)
tfidf_test = np.array(tfidf.transform(df_test.text).toarray(),
                      dtype=np.float16)

for i in range(n_features):
    df_train['tfidf_' + str(i)] = tfidf_train[:, i]
    df_test['tfidf_' + str(i)] = tfidf_test[:, i]

del tfidf_train, tfidf_test
gc.collect()

# Set up training and test sets

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
print(X.shape, X_test.shape)

# del df_train, df_test
gc.collect()

# X = X.as_matrix()
# y = y.as_matrix()
# X_test = X_test.as_matrix()

# Light GBM #

cnt = 0
p_buf = []
n_splits = 5
n_repeats = 3
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
                        key=lambda x: x[1])[::-1]
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

    cnt += 1
    # if cnt > 0: # Comment this to run several folds
    #     break

    del model, lgb_train, lgb_valid, p
    gc.collect


auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))

preds = p_buf/cnt

# Prepare submission
subm = pd.DataFrame()
subm['id'] = id_test
subm['project_is_approved'] = preds
subm.to_csv('submission_lightGBM.csv', index=False)

# tfidf

# df_train_sample = df_train.sample(frac=0.15, replace=False, random_state=42)

# X = df_train_sample.text
# y = df_train_sample.project_is_approved

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                      random_state = 42)

# tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer, min_df=10,
#                                    ngram_range=(1, 2))

# tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# tfidf_test = tfidf_vectorizer.transform(X_test)

# nb_classifier = MultinomialNB()

# nb_classifier.fit(tfidf_train, y_train)

# preds = nb_classifier.predict(tfidf_test)

# score = metrics.confusion_matrix(y_test, preds)

# print(score)

# print(len(preds))
