from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

from tweepy import API
from tweepy import Cursor

import twitter_credentials

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, mpld3

import datetime as dt

import re
from textblob import TextBlob

import spacy
from collections import Counter
nlp = spacy.load('en')

from wtforms import Form, StringField, validators



class TwitterClient():

    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
         return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_friends):
            num_friends.append(friend)
        return num_friends

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets =[]
        for tweet in Cursor(self.twitter_client.home_timeline).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets



class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth



class Streamer():

    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_authenticator.authenticate_twitter_app()

        stream = Stream(auth, listener)
        stream.filter(track=hash_tag_list)



class TwitterListener(StreamListener):

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on data: %s" % str(e))
        return True

    def on_error(self, status):
        if status == 420:
            return False
        print(status)



class TweetAnalyzer():


    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def find_most_common_words(self, tweet_string):
        parsed_tweet_string = nlp(tweet_string)
        cleaned = [ token.text for token in parsed_tweet_string if token.is_stop != True and token.is_punct != True and token.pos_ == ('NOUN' or 'VERB' or 'PROPN') ]
        word_count = Counter(cleaned)
        most_common_words = word_count.most_common(15)
        return most_common_words 
        

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])

        # df['id'] = np.array([tweet.id for tweet in tweets])
        # df['len'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        # df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])      
        df['sentiment'] = np.array([self.analyze_sentiment(tweet) for tweet in df['tweets']])

        return df
      
 
    def get_daily_total(self, df):
        count_df = df['date'].groupby(df['date'].dt.floor('d')).size().reset_index(name='count') 
        daily_totals = count_df['count']
        return daily_totals


class Tweeter():

    def __init__(self):
        self.twitter_client = TwitterClient()
        self.tweet_analyzer = TweetAnalyzer()
        self.api = TwitterClient().get_twitter_client_api()


    ## staying DRY
    def get_tweets(self, user, count):
        tweets = self.api.user_timeline(screen_name=user, count=count)
        return tweets


    def latest_tweets(self, user, count):
        tweets = self.get_tweets(user, count)
        df = self.tweet_analyzer.tweets_to_data_frame(tweets)
        latest = df[['tweets', 'date']].head(5)
        return latest


    def most_popular(self, user, count):
        tweets = self.get_tweets(user, count)
        df = self.tweet_analyzer.tweets_to_data_frame(tweets)
        most_popular = df.sort_values(by=['likes'], ascending=False)
        popular_selection = most_popular.head(5)
        return popular_selection


    def positivity_rating(self, user, count):
        tweets = self.get_tweets(user, count)
        df = self.tweet_analyzer.tweets_to_data_frame(tweets)
        positivity = df['sentiment']
        overall_positivity = np.mean(positivity)
        return overall_positivity


    def posting_frequency(self, user, count):
        tweets = self.get_tweets(user, count)
        df = self.tweet_analyzer.tweets_to_data_frame(tweets)
        daily_totals = self.tweet_analyzer.get_daily_total(df)
        fig = plt.figure()
        plt.plot(daily_totals)
        plt.xlabel('Number of daily tweets')
        htmlfig = mpld3.fig_to_html(fig)
        return htmlfig
    

    def most_common_words(self, user, count):       
        tweets = self.get_tweets(user, count)
        tweet_array = np.array([ tweet.text for tweet in tweets ])
        tweet_bag = np.array2string(tweet_array)    
        most_common_words = self.tweet_analyzer.find_most_common_words(tweet_bag)
        return most_common_words


