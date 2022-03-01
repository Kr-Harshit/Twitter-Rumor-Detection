from ast import literal_eval
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler


def tweet_meta_data_util(tweet_data, user_data):
 data = tweet_data.merge(user_data[['tweetId', 'created_at']], on='tweetId', suffixes=['_tweet', '_user'], how='inner')
 data['created_at_tweet'] = data['created_at_tweet'].apply(lambda date_time_str : datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S'))
 data['created_at_user'] = data['created_at_user'].apply(lambda date_time_str : datetime.strptime(date_time_str, '%a %b %d %H:%M:%S +0000 %Y'))
 data['n_days'] = data['created_at_tweet'] - data['created_at_user']
 data['n_days'] = data['n_days'].apply(lambda x : x.days)
 data['labels'] = data['labels'].astype('int')
 data = data[['favorite_count','retweet_count','n_symbols','n_urls','n_hashtags','n_user_mentions', 'n_days','labels' ]]

 X = data[['retweet_count','n_urls','n_hashtags','n_user_mentions', 'n_days']]
 y = data['labels']

 return X, y


def user_tweet_meta_util(tweet_data, user_data):
    user_data['verified'] = user_data['verified'].map({True:1, False:0})
    user_unique = pd.DataFrame(user_data.groupby('userId').mean())
    user_unique['verified']  = user_unique['verified'].apply(lambda x: x > 0.5 )
    user_unique = user_unique.reset_index()
    tweet_data['labels'] = tweet_data['labels'].map({'true':1, 'false':1, 'unverified':1, 'non-rumor':0})
    x = pd.pivot_table(tweet, values='tweetId', columns=['labels'], index=['userId'], fill_value=0, aggfunc="count")
    x = x.reset_index()
    data = user_unique.merge(x, on='userId')
    data = data.rename(columns={0:"non-rumor", 1:"rumor"})
    data['labels'] = np.where(data['non-rumor'] >= data['rumor'], 'non-rumor', 'rumor')
    data.drop([ 'userId', 'rumor', 'non-rumor'], axis=1, inplace=True)
    le = LabelEncoder()
    data['verified'] = le.fit_transform(data['verified'])
    le_labels = LabelEncoder()
    data['labels'] = le_labels.fit_transform(data['labels'])
    scaler = StandardScaler()
    data[['followers_count', 'statuses_count', 'friends_count']] = scaler.fit_transform(data[['followers_count', 'statuses_count', 'friends_count']])
    X = data[['verified', 'followers_count', 'statuses_count', 'friends_count']]
    y = data['labels']

    return X, y