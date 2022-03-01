''' This utility package is responsible for extracting feature from a given source-tweet and saving it into CSV '''

import pandas as pd
import numpy as np
import os
import json
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm


class FeatureExtractor():
    def __init__(self, directory):
        self.dir = directory
        self.user_features = ['userId', 'name', 'screen_name', 'tweetId', 'verfifed', 'followers_count', 'statuses_count',
                     'friends_count', 'favourites_count', 'listed_count', 'location','created_at','profile_image_url','profile_background_image_url','default_profile_image',
                     'default_profile','profile_use_background_image','url'
                    ] 
        self.tweet_features = ['tweetId', 'text', 'source', 'created_at', 'is_reply', 'retweet_count', 'favorite_count',
                      'is_quote_status', 'entities_symbols', 'n_symbols', 'entities_user_mentions', 'n_user_mentions',
                      'entities_hashtags','n_hashtags','entities_url','n_url','label'
                     ]

    def __read_data(self, tweet_file):
        file = open(tweet_file)
        tweet = json.loads(file.read())
        file.close();
        return tweet

    def __get_user_data(self, user_df, data, tweetId):
        user_data = {
            'userId':data['id'],
            'name':data['name'],
            'screen_name':data['screen_name'],
            'tweetId': tweetId,
            'verfifed':data['verified'],
            'followers_count':data['followers_count'],
            'statuses_count':data['statuses_count'],
            'friends_count':data['friends_count'],
            'favourites_count':data['favourites_count'],
            'listed_count':data['listed_count'],
            'location':data['location'],
            'created_at':datetime.strptime(data['created_at'], '%a %b %d %H:%M:%S +0000 %Y'),
            'profile_image_url': data['profile_image_url'],
            'profile_background_image_url':data['profile_background_image_url'],
            'default_profile_image':data['default_profile_image'],
            'default_profile':data['default_profile'],
            "profile_use_background_image": data['profile_use_background_image'],
            'url':data['url']
        }
        user_df.loc[len(user_df.index)] = user_data;

    def __get_source(self, link):
        soup = BeautifulSoup(link, "html.parser")
        return soup.string

    def __get_tweet_data(self, tweet_df, data, label):
        tweet_data = {
            'tweetId':data['id'],
            'text':data['text'],
            'source': self.__get_source(data['source']),
            'created_at':datetime.strptime(data['created_at'], '%a %b %d %H:%M:%S +0000 %Y'),
            'is_reply': data["in_reply_to_status_id"],
            'retweet_count':data['retweet_count'],
            'favorite_count':data['favorite_count'],
            'is_quote_status':data['is_quote_status'],
            'entities_symbols': [x['text'] for x in data['entities']['symbols']],
            'n_symbols' : len(data['entities']['symbols']),
            'entities_user_mentions':[x['screen_name'] for x in data['entities']['user_mentions']],
            'n_user_mentions': len(data['entities']['user_mentions']),
            'entities_hashtags':[x for x in data['entities']['hashtags']],
            'n_hashtags': len(data['entities']['hashtags']),
            'entities_url':[x['expanded_url'] for x in data['entities']['urls']],
            'n_url':len(data['entities']['urls']),
            'label':label
        }
        tweet_df.loc[len(tweet_df.index)] = tweet_data

    def get_data(self, df):
        ''' Returns tweet dataframe and user dataframe'''
        
        user_df = pd.DataFrame(columns=self.user_features)
        tweet_df = pd.DataFrame(columns=self.tweet_features)

        for tweet in tqdm(os.listdir(self.dir), desc="Reading..."):
            tweetId = tweet[:-5]
            # print(tweetId)
            tweet_data = self.__read_data(os.path.join(self.dir, tweet))
            label = None
            if df:
                label = df.loc[df['tweetId'] == int(tweetId), 'labels'].values[0]
            self.__get_tweet_data(tweet_df, tweet_data, label)
            user_data = tweet_data['user']
            self.__get_user_data(user_df, user_data, tweetId)

        return tweet_df, user_df

    def get_reaction(self, df):
        pass

    def save(self, tweet_df, user_df, save_dir):
        tweet_df.to_csv(os.path.join(save_dir, 'tweet.csv'), index=False)
        user_df.to_csv(os.path.join(save_dir, 'user.csv'), index=False)



