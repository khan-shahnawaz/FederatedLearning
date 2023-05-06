import os
import pandas as pd
import requests
import json
import numpy as np
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
#Check if the file adult.data exists or not
if not os.path.exists('data/adult/adult.data'):
    #If not, download the file
    print('Downloading the dataset...')
    r = requests.get(data_url)
    with open('data/adult/adult.data', 'w') as f:
        f.write(r.text)
    print('Done!')
    
test_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
#Check if the file adult.test exists or not
if not os.path.exists('data/adult/adult.test'):
    #If not, download the file
    print('Downloading the dataset...')
    r = requests.get(test_data_url)
    with open('data/adult/adult.test', 'w') as f:
        f.write(r.text)
    print('Done!')
train_folder = 'data/adult/data/train'
test_folder = 'data/adult/data/test'

#Convert the dataset into json and divide clients based on native country
print('Converting the dataset into json and dividing clients based on native country...')
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status',
    'occupation', 'relationship', 'race', 'sex', 'captial-gain', 'capital-loss', 
    'hours-per-week', 'native-country', 'income'
]

train_df = pd.read_csv('data/adult/adult.data', header=None, names=column_names)
test_df = pd.read_csv('data/adult/adult.test', header=None, names=column_names)
train_df.replace(' ?', np.nan, inplace=True)
test_df.replace(' ?', np.nan, inplace=True)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
countries = train_df['native-country'].unique()
#Remove those countries which do not have occurance in test_df
countries = [country for country in countries if country in test_df['native-country'].unique()]

train_df = train_df[train_df['native-country'].isin(countries)]
test_df = test_df[test_df['native-country'].isin(countries)]


# Convert the categorial data into numerical data
train_df['income'] = train_df['income'].apply(lambda x: 0 if x == ' <=50K' else 1)
train_df['income']=train_df['income'].astype('category')
test_df['income'] = test_df['income'].apply(lambda x: 0 if x == ' <=50K.' else 1)
test_df['income']=test_df['income'].astype('category')
#Print number of positive and negative examples in train and test data
print('Number of positive examples in train data: {}'.format(len(train_df[train_df['income'] == 1])))
print('Number of negative examples in train data: {}'.format(len(train_df[train_df['income'] == 0])))
print('Number of positive examples in test data: {}'.format(len(test_df[test_df['income'] == 1])))
print('Number of negative examples in test data: {}'.format(len(test_df[test_df['income'] == 0])))

int_to_country = {i: country for i, country in enumerate(countries)}
#Apply the same categorial encoding to both train and test data
for col in train_df.columns:
    if train_df[col].dtype == 'object' and col != 'income':
        if col != 'native-country':
            if not pd.api.types.is_numeric_dtype(train_df[col]):
                if len(train_df[col].unique()) == 2:
                    train_df[col] = (train_df[col] == train_df[col].unique()[0]).astype(float)
                    test_df[col] = (test_df[col] == test_df[col].unique()[0]).astype(float)
                else:
                    test_df = pd.merge(test_df.drop(columns=col), pd.get_dummies(test_df[col].astype('category')), left_index=True, right_index=True)
                    train_df = pd.merge(train_df.drop(columns=col), pd.get_dummies(train_df[col].astype('category')), left_index=True, right_index=True)

            continue
        mapper = {k: v for v, k in enumerate((pd.concat([train_df[col], test_df[col]])).unique())}
        train_df[col] = train_df[col].astype('category').map(mapper)
        test_df[col] = test_df[col].astype('category').map(mapper)
        if col == 'native-country':
            int_to_country = {v: k for k, v in mapper.items()}
data_to_dump = {'num_samples': [], 'users': [], 'user_data': {}}
for country in train_df['native-country'].unique():
    country_name = int_to_country[country]
    tdf = train_df[train_df['native-country'] == country].drop(['native-country'], axis=1)
    data_to_dump['num_samples'].append(len(tdf))
    data_to_dump['users'].append(country_name)
    data_to_dump['user_data'][country] = {'x': [], 'y': []}
    data_to_dump['user_data'][country]['x'] = tdf.drop(['income'], axis=1).to_numpy(np.float32).tolist()
    data_to_dump['user_data'][country]['y'] = tdf['income'].to_numpy(np.float32)[:,None].tolist()

#Print the index of sex column
index = {column: index for index, column in enumerate(train_df.columns)}
with open(train_folder + '/mytrain.json', 'w') as outfile:
    json.dump(data_to_dump, outfile)
#Repeat the same process for test data
data_to_dump = {'num_samples': [], 'users': [], 'user_data': {}}
for country in test_df['native-country'].unique():
    country_name = int_to_country[country]
    tdf = test_df[test_df['native-country'] == country].drop(['native-country'], axis=1)
    tdf = tdf.sample(frac=1).reset_index(drop=True)
    data_to_dump['num_samples'].append(len(tdf))
    data_to_dump['users'].append(country_name)
    data_to_dump['user_data'][country] = {'x': [], 'y': []}
    data_to_dump['user_data'][country]['x'] = tdf.drop(['income'], axis=1).to_numpy(np.float32).tolist()
    data_to_dump['user_data'][country]['y'] = tdf['income'].to_numpy(np.float32)[:,None].tolist()
with open(test_folder + '/mytest.json', 'w') as outfile:
    json.dump(data_to_dump, outfile)


print('Done!')