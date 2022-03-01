import time
import json
import os
import sys
import tweepy
import twitter_api
from tqdm import tqdm
from pprint import pprint


class TweetCollector():
    def __init__(self):
        auth = tweepy.OAuthHandler(twitter_api.API_KEY, twitter_api.API_SECRET_KEY)
        auth.set_access_token(twitter_api.ACCESS_TOKEN, twitter_api.ACESS_TOKEN_SECRET)
        self.api = tweepy.API(auth,  wait_on_rate_limit=True)

    def get_data_from_id(self, save_Dir, tweetID_list = []):
        not_collected_tweets = []
        collected = 0
        for i, id in tqdm(enumerate(tweetID_list), desc= "downloading..."):
            try:
                tweet = self.api.get_status(id)
                if not os.path.exists(save_Dir):
                    os.makedirs(save_Dir)
                file = open(os.path.join(save_Dir, str(id)+'.json'), "w+")
                tweet_json = tweet._json
                json.dump(tweet_json, file, indent=6)
                collected += 1
                file.close()
            except tweepy.RateLimitError as e:
                print("Twitter api rate limit reached".format(e))
                time.sleep(60)
                continue
            except tweepy.TweepError as e:
                print("Tweepy error occured:{}".format(e))
                print('Error for id = {}'.format(id) )
                not_collected_tweets.append(id)
                continue
            except KeyboardInterrupt as e:
                print('Total tweets collected: ', i-1)
        print(collected, 'number of tweets collected')
        return not_collected_tweets

    def get_data_from_statement(self, save_Dir, statement):
        search = statement + " -filter:retweets"
        tweets = tweepy.Cursor(self.api.search,
                                q = search, 
                                lang='en').items(5)
        for tweet in tqdm(tweets, desc="Saving...."):
            tweet_json = tweet._json
            if not os.path.exists(save_Dir):
                os.makedirs(save_Dir)
            file = open(os.path.join(save_Dir, tweet_json['id_str'] + '.json'), "w+")
            json.dump(tweet_json, file, indent=6)
            file.close()    
        
    def get_reactions(self, save_Dir="", tweetID_name_list = []):
        ''' fetch replies to a specific tweet ID ''' 
        if save_Dir != "" and  not os.path.exists(save_Dir):
            os.makedirs(save_Dir)
        for i, tweet in tqdm(enumerate(tweetID_name_list), desc='downloading...'):
            tweet_id, name = tweet[0], '@'+tweet[1]
            # print(tweet_id, name)
            try: 
                for tweet in tweepy.Cursor(self.api.search, q='to:{}'.format(name), timeout=99999).items(100):
                    if hasattr(tweet, 'in_reply_to_status_id'):
                        if (tweet._json['in_reply_to_status_id'] == tweet_id):
                            print(tweet) 
                            if save_Dir == "":
                                continue
                            with open(os.path.join(save_Dir, tweet_id, str(tweet.id)+".json"), 'w') as file:
                                print(tweet._json)
                                json.dumps(tweet._json, file, indent=6)
                    break;
            except tweepy.RateLimitError as e:
                print("Twitter api rate limit reached".format(e))
                time.sleep(60)
                continue
            except tweepy.TweepError as e:
                print("Tweepy error occured:{}".format(e))
                continue
            except KeyboardInterrupt as e:
                print('Total tweets collected: ', i-1)

    def basic_reactions(self, username, tweetID):
        username='@'+username
        print(username, tweetID)
        print(type(tweetID))
        replies=[]
        for tweet in tweepy.Cursor(self.api.search, q='to:@'+username, timeout=99999).items(100):
            print(tweet._json)
            if (tweet._json , 'in_reply_to_status_id'):
                if (tweet._json['in_reply_to_status_id'] == tweetID):
                    replies.append(tweet)
            break
        print(replies)
        
            
