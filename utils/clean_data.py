import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
# !pip install ekphrasis
# !pip install tweet-preprocessor
import preprocessor as p
import re
from textblob import TextBlob
import seaborn as sns
from scipy.stats import pointbiserialr
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
stop_list = stopwords.words('english')
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import os, re, csv, math, codecs


# For Training
import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
#from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

# For array, dataset, and visualizing
import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set_style('darkgrid')

import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import STOPWORDS,WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata


from keras.preprocessing import text,sequence
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Embedding
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

# !pip install textblob
# !python -m textblob.download_corpora


def meta_data_preprocessing(tweet, user):
  user['created_at'] = user['created_at'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
  tweet['created_at'] = tweet['created_at'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
  current_datetime = datetime.now()
  user['account_age'] = user['created_at'].apply(lambda x : (current_datetime - x).days)
  user['screen_name_len'] = user['screen_name'].apply(len)
  user['verfifed'] = user['verfifed'].astype(np.int64)
  user['default_profile'] = user['default_profile'].astype(np.int64)
  user['default_profile_image'] = user['default_profile_image'].astype(np.int64)
  user['profile_use_background_image'] = user['profile_use_background_image'].astype(np.int64)
  user['profile_image_url'] = user['profile_image_url'].apply(lambda x : x is not np.nan )
  user['profile_image_url'] = user['profile_image_url'].astype(np.int64)
  user['profile_background_image_url'] = user['profile_background_image_url'].apply(lambda x : x is not np.nan )
  user['profile_background_image_url'] = user['profile_background_image_url'].astype(np.int64)
  user['has_location'] = user['location'].apply(lambda x : x is not np.nan)
  user['has_location'] = user['has_location'].astype(np.int64)
  user['has_url'] = user['url'].apply(lambda x : x is not np.nan)
  user['has_url'] = user['has_url'].astype(np.int64)
  user[['followers_count', 'statuses_count','friends_count',
      'favourites_count', 'listed_count', 'account_age']] = user[['followers_count', 'statuses_count','friends_count',
      'favourites_count', 'listed_count', 'account_age']].apply(lambda x  : np.log(x+1))
  # cleaning text data 
  for i,v in enumerate(tweet['text']):
    tweet.loc[i,'text'] = p.clean(v)

  
  def preprocess_data(data):
    #Removes Numbers
    data = data.astype(str).str.replace('\d+', '')
    lower_text = data.str.lower()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    w_tokenizer =  TweetTokenizer()
  
    def lemmatize_text(text):
      return [(lemmatizer.lemmatize(w)) for w \
                          in w_tokenizer.tokenize((text))]

    def remove_punctuation(words):
      new_words = []
      for word in words:
        new_word = re.sub(r'[^\w\s]', '', (word))
        if new_word != '':
          new_words.append(new_word)
      return new_words

    words = lower_text.apply(lemmatize_text)
    words = words.apply(remove_punctuation)
    return pd.DataFrame(words)

  pre_tweets = preprocess_data(tweet['text'])
  tweet['text'] = pre_tweets
  stop_words = set(stopwords.words('english'))
  tweet['text'] = tweet['text'].apply(lambda x: [item for item in \
                                      x if item not in stop_words])
  
  def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

  tweet['polarity'] =  tweet['text'].apply(lambda x : analyze_sentiment(' '.join(x)))
  tweet['is_reply'] = tweet['is_reply'].isnull()
  tweet['is_reply'] = tweet['is_reply'].apply(lambda x : not x)
  tweet['is_reply'] = tweet['is_reply'].astype(np.int64)
  tweet['is_quote_status'] = tweet['is_quote_status'].astype(np.int64)
  tweet['text_len'] = tweet['text'].apply(len)
  tweet['post_age'] = tweet['created_at'].apply(lambda x : (current_datetime-x).days)
  tweet[['retweet_count','favorite_count','n_symbols', 'n_user_mentions','n_hashtags']] = tweet[['retweet_count','favorite_count','n_symbols', 'n_user_mentions','n_hashtags']].apply(lambda x : np.log(x+1))
  merge = tweet.merge(user, on='tweetId', suffixes=['_tweet', '_user'])
  merge['posted_in'] = merge['created_at_tweet'] - merge['created_at_user']
  merge['posted_in'] =  merge['posted_in'].apply(lambda x : x.days)
  df = merge[['is_reply','verfifed', 'followers_count',
       'retweet_count', 'favorite_count', 'is_quote_status', 'n_symbols', 
       'n_user_mentions', 'n_hashtags','n_url', 'polarity', 'text_len', 'post_age',
       'statuses_count', 'friends_count', 'favourites_count', 'listed_count',
       'profile_image_url', 'profile_background_image_url', 'default_profile_image',
       'default_profile', 'profile_use_background_image', 'account_age',
       'screen_name_len', 'has_location', 'has_url', 'posted_in', 'label']]
  df.loc[:, 'label'] = df['label'].replace({'non-rumor':False, 'true':True, 'false':True, 'unverified':True})
  df['label'] = df['label'].astype(np.int64)
  selected_features = ['profile_use_background_image','profile_background_image_url',
                     'default_profile_image','default_profile','verfifed', 'text_len',
                     'n_url', 'post_age', 'statuses_count', 'listed_count', 'n_symbols',
                     'posted_in', 'n_user_mentions', 'polarity', 'favourites_count', 'favorite_count',
                     'screen_name_len', 'n_hashtags']
  sel_num_features = ['text_len','n_url', 'post_age', 'statuses_count', 'listed_count', 'n_symbols',
                     'posted_in', 'n_user_mentions', 'polarity', 'favourites_count', 'favorite_count',
                     'screen_name_len', 'n_hashtags']

  sel_cat_features = ['profile_use_background_image','profile_background_image_url',
                      'default_profile_image','default_profile','verfifed']
  X = df[selected_features]
  y = df['label']
  from pickle import load
  sc_x = load(open('/content/drive/MyDrive/Rumor Detection/Scaler/scaler.pkl', 'rb'))
  X[sel_num_features] = sc_x.transform(X[sel_num_features])
  return X, y

def tt_preprocess(db):  
  data=[db['text'],db['text'].apply(len),db['label']]
  headers=["text","length","target"]
  text_df=pd.concat(data,axis=1,keys=headers)
  text_df['target']=text_df['target'].astype(int)
  text_df['clean_tweet']=''
  import preprocessor as p
  def word_abbrev(word):
      return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word
    
  def replace_abbrev(text):
      string = ""
      for word in text.split():
          string += word_abbrev(word) + " "        
      return string
  abbreviations = {
    "$" : " dollar ",
    "â‚¬" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "f*ck" : "fuck",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
    }
  #removing link, reserved word, emoji,smiley,number
  p.set_options(p.OPT.URL
  ,p.OPT.RESERVED
  ,p.OPT.EMOJI
  ,p.OPT.SMILEY
  ,p.OPT.NUMBER)

  for index,row in text_df.iterrows():
    text_df['clean_tweet'][index] = p.clean(text_df['text'][index])
  from wordsegment import load,segment
  # Removing contractions
  import contractions
  for index,row in text_df.iterrows():
    expanded_words = []
    for word in text_df['clean_tweet'][index].split():
      expanded_words.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_words)
    text_df['clean_tweet'][index]=expanded_text
  # Replace abbreviations
  for index,row in text_df.iterrows():
    text_df['clean_tweet'][index] = replace_abbrev(text_df['clean_tweet'][index])
  # removing hashtags, mentions, splitting hashtags
  load()
  for index,row in text_df.iterrows():
    text_df['clean_tweet'][index] = ' '.join(segment(text_df['clean_tweet'][index]))
  ####
  stop_words = set(stopwords.words('english'))
  stop_words.add('k')
  # removing stop words
  for index,row in text_df.iterrows():
    text_df['clean_tweet'][index] = ' '.join([word for word in text_df['clean_tweet'][index].split() if word not in (stop_words)])

  return text_df

def further_process_1(text_df):
  import numpy as np
  raw_docs = text_df['clean_tweet'].tolist()
  text_df['clean_len']=''
  text_df['clean_len']=text_df['clean_tweet'].apply(len)
  max_seq_len = np.round(text_df['clean_len'].mean()+text_df['clean_len'].std()).astype(int)
  MAX_NB_WORDS=100000
  embed_dim = 300
  tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
  tokenizer.fit_on_texts(text_df['clean_tweet'])  #leaky
  word_seq = tokenizer.texts_to_sequences(raw_docs)
  word_index = tokenizer.word_index
  print("dictionary size: ", len(word_index))

  #pad sequences
  word_seq = sequence.pad_sequences(word_seq, maxlen=max_seq_len)
  return word_seq

def reaction_preprocessing(tweets):

  def weighted_sentiments_avg(tweet, df, weight_col, colname):
    tweet[colname+"_weighted_polarity"] = tweet['sentiments']*weight_col
    x  = pd.DataFrame(tweet.groupby('tweetId')[colname+"_weighted_polarity"].mean())
    # df = df.merge(x, left_index=True, right_index=True)
    return x

  labels = pd.DataFrame(columns=['tweetId', 'label'])

  for id in tweets['tweetId'].unique():
    label = tweets[tweets['tweetId'] == id]['label'].unique()[0]
    labels.loc[len(labels.index)] = [id, label]

  labels = labels.set_index('tweetId')
  labels.head()

  avg = tweets.groupby('tweetId').mean()
  df = pd.DataFrame(avg)

  df = df.merge(labels, left_index=True, right_index=True)

  x = weighted_sentiments_avg(tweets.copy(), df, tweets['favorite_count'] , 'favorite')
  df = df.merge(x, left_index=True, right_index=True)

  x = weighted_sentiments_avg(tweets.copy(), df, tweets['retweet_count'] , 'retweet')
  df = df.merge(x, left_index=True, right_index=True)

  x = weighted_sentiments_avg(tweets.copy(), df, tweets['retweet_count']+tweets['favorite_count'] , 'favorite_retweet')
  df = df.merge(x, left_index=True, right_index=True)

  df['label'] = df['label'].replace({'non-rumours':False, 'rumours':True})
  df['label'] = df['label'].astype(np.int64)
  df[['retweet_count', 'favorite_count' ,'favorite_weighted_polarity', 'n_hashtag',
      'sentiments', 'retweet_weighted_polarity', 'favorite_retweet_weighted_polarity'] ] = df[['retweet_count', 'favorite_count' ,'favorite_weighted_polarity', 'n_hashtag',
                                                                          'sentiments', 'retweet_weighted_polarity', 'favorite_retweet_weighted_polarity'] ].astype(np.int64)
    
  return df