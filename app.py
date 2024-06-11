from pydantic import BaseModel
import tweepy
import pandas as pd
import nltk
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from fastapi import FastAPI, Request
from math import log
from fastapi.middleware.cors import CORSMiddleware
#nltk.download('punkt')
#nltk.download('stopwords')
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class TweetClassifier(object):
    def __init__(self, trainData, method = 'tf-idf'):
        self.tweets, self.labels = trainData['message'], trainData['label']
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_depressive = dict()
        self.prob_positive = dict()
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.tf_depressive[word] + 1) / (self.depressive_words + \
                                                                len(list(self.tf_depressive.keys())))
        for word in self.tf_positive:
            self.prob_positive[word] = (self.tf_positive[word] + 1) / (self.positive_words + \
                                                                len(list(self.tf_positive.keys())))
        self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets


    def calc_TF_and_IDF(self):
        noOfMessages = self.tweets.shape[0]
        self.depressive_tweets, self.positive_tweets = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_tweets = self.depressive_tweets + self.positive_tweets
        self.depressive_words = 0
        self.positive_words = 0
        self.tf_depressive = dict()
        self.tf_positive = dict()
        self.idf_depressive = dict()
        self.idf_positive = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.tweets.iloc[i])
            count = list() #To keep track of whether the word has ocured in the message or not.
                           #For IDF
            for word in message_processed:
                if self.labels.iloc[i]:
                    self.tf_depressive[word] = self.tf_depressive.get(word, 0) + 1
                    self.depressive_words += 1
                else:
                    self.tf_positive[word] = self.tf_positive.get(word, 0) + 1
                    self.positive_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels.iloc[i]:
                    self.idf_depressive[word] = self.idf_depressive.get(word, 0) + 1
                else:
                    self.idf_positive[word] = self.idf_positive.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_depressive = dict()
        self.prob_positive = dict()
        self.sum_tf_idf_depressive = 0
        self.sum_tf_idf_positive = 0
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.tf_depressive[word]) * log((self.depressive_tweets + self.positive_tweets) \
                                                          / (self.idf_depressive[word] + self.idf_positive.get(word, 0)))
            self.sum_tf_idf_depressive += self.prob_depressive[word]
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.prob_depressive[word] + 1) / (self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))

        for word in self.tf_positive:
            self.prob_positive[word] = (self.tf_positive[word]) * log((self.depressive_tweets + self.positive_tweets) \
                                                          / (self.idf_depressive.get(word, 0) + self.idf_positive[word]))
            self.sum_tf_idf_positive += self.prob_positive[word]
        for word in self.tf_positive:
            self.prob_positive[word] = (self.prob_positive[word] + 1) / (self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))


        self.prob_depressive_tweet, self.prob_positive_tweet = self.depressive_tweets / self.total_tweets, self.positive_tweets / self.total_tweets

    def classify(self, processed_message):
        pDepressive, pPositive = 0, 0
        for word in processed_message:
            if word in self.prob_depressive:
                pDepressive += log(self.prob_depressive[word])
            else:
                if self.method == 'tf-idf':
                    pDepressive -= log(self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))
                else:
                    pDepressive -= log(self.depressive_words + len(list(self.prob_depressive.keys())))
            if word in self.prob_positive:
                pPositive += log(self.prob_positive[word])
            else:
                if self.method == 'tf-idf':
                    pPositive -= log(self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))
                else:
                    pPositive -= log(self.positive_words + len(list(self.prob_positive.keys())))
            pDepressive += log(self.prob_depressive_tweet)
            pPositive += log(self.prob_positive_tweet)
        return pDepressive >= pPositive

    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result
    
def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words
    # Include the process_message function code here


def fetch_tweets(username: str):
    """
    Fetches recent tweets for the specified username.
    """
    client = tweepy.Client(TWITTER_BEARER_TOKEN)
    user_id = client.get_user(username=username).data.id
    tweets = []
    responses = tweepy.Paginator(client.get_users_tweets, user_id, max_results=100, limit=1)
    for response in responses:
        for tweet in response.data:
            tweets.append(tweet.text)
    return tweets

# Load preprocessed data
tweets = pd.read_csv('sentiment_tweets3.csv')

# Initialize classifier
sc_tf_idf = TweetClassifier(tweets, 'tf-idf')
sc_tf_idf.train()


@app.get("/test")
def read_root():
    return {"message": "Welcome to Twitter Sentiment Analysis with FastAPI!"}

class ModelInput(BaseModel):
    user_handle: str
    
@app.post("/analyze_tweets")
async def analyze_tweets(input_parameters: ModelInput):
    user_handle = input_parameters.user_handle
    if user_handle:
        try:
            sc_tf_idf.train()
            tweets = fetch_tweets(user_handle)
            print("fetched")
            updated_tweets = []  # Create a new list to store updated tweets
            sentiments = []
            for tweet in tweets:
                processed_message = process_message(tweet)
                prediction = sc_tf_idf.classify(processed_message)
                updated_tweet = tweet + (",Depressive" if prediction else ",'Positive'")  # Append sentiment to tweet
                updated_tweets.append(updated_tweet)
                sentiments.append(prediction)
            average_sentiment = sum(sentiments) / len(sentiments) if len(sentiments) > 0 else 0
            print(average_sentiment)
            print(len(sentiments))
            return {"user_handle": user_handle, "tweets": updated_tweets, "sentiments": sentiments}
        except tweepy.TweepError as e:
            return {"error": f"Error fetching tweets: {e}"}
    else:
        return {"error": "Please provide a Twitter handle."}


@app.post("/analyze_tweets_average")
async def analyze_tweets(input_parameters: ModelInput):
    user_handle = input_parameters.user_handle   
    if user_handle:
        try:
            
            tweets = fetch_tweets(user_handle)
            print("fetched")
            sentiments = []
            for tweet in tweets:
                processed_message = process_message(tweet)
                prediction = sc_tf_idf.classify(processed_message)
                sentiments.append(prediction)

            # Calculate average sentiment score
            print('working')
            average_sentiment = sum(sentiments) / len(sentiments) if len(sentiments) > 0 else 0

            # Convert average sentiment score to label
            result = "depressed" if average_sentiment >= 0.5 else "not depressed"
            print(average_sentiment)
            return {"user_handle": user_handle, "sentiment": result}
        except tweepy.TweepError as e:
            return {"error": f"Error fetching tweets: {e}"}
    else:
        return {"error": "Please provide a Twitter handle."}