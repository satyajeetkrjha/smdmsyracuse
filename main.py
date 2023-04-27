import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from textblob import TextBlob
import nltk
import ssl
from nltk.corpus import stopwords
from country import Country


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download()



from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from PIL import Image
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud


#dropping the
#tweets_df = df.drop(['UserId','UserName','Verified','Followers','Retweeted','Language'], axis=1)
#print(tweets_df.columns)


#Method to preprocess and clean the tweeter data
def preprocessing_tweets(tweet):
    try:
        # To remove duplicate records
        global tweets_df
        tweets_df = tweets_df.drop_duplicates()
        # Check NA values for Timestamp
        tweets_df[tweets_df['Timestamp'].isna()]
        # Remove missing values
        tweets_df.dropna(subset=['Timestamp'], inplace=True)
        # Filter to get only English tweets
        tweets_df = tweets_df[tweets_df['Language'] == 'en'].reset_index(drop=True)
        # Filter to get only Known Country  tweets
        tweets_df = tweets_df[tweets_df['Country'] != 'Unknown'].reset_index(drop=True)

        # To Remove URL from tweet text
        tweets_df['Tweet'] = tweets_df['Tweet'].apply(lambda x: re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', x))
        # To Remove mention (@user)
        tweets_df['Tweet'] = tweets_df['Tweet'].apply(lambda x: re.sub(r'@\w+', '', x))

        # To Remove Punctuation
        tweets_df['Tweet_punc'] = tweets_df['Tweet'].apply(lambda x: re.sub('[^\w\s]', '', x))

        # remove eomjis during proceprocessing

        import emoji
        #Removing emojis from tweets
        def remove_emojis(tweet):
            return emoji.replace_emoji(tweet, replace=" ")
        #Converting tweets to lowercase
        tweet = tweet.lower()
        tweet = tweet.strip(' "\'')
        tweet = remove_emojis(tweet)
        #Tokenizing the words
        tweet_tokens = word_tokenize(tweet)
        #Removing stopwords from tweets
        tweetsFilterted = [word for word in tweet_tokens if not word in stop_words]
        return " ".join(tweetsFilterted)

    except:
        print("An error occurred while Pre processing")


def sentiment_analysis():
    def Subjectivity(tweet):
        return TextBlob(tweet).sentiment.subjectivity

    def get_Positive_Negative_Labelling(score):
        if score == 0:
            return 'Neutral'
        elif score < 0:
            return 'Negative'
        else:
            return 'Positive'

    def getPolarity(tweet):
        return TextBlob(tweet).sentiment.polarity

    tweets_df['subjectivity'] = tweets_df['Tweet'].apply(Subjectivity)
    tweets_df['polarity'] = tweets_df['Tweet'].apply(getPolarity)
    tweets_df['sentiment'] = tweets_df['polarity'].apply(get_Positive_Negative_Labelling)



def getMostCommonWords():
    # we try to get 50 most common words that appears in tweets
    from collections import Counter
    words = Counter(" ".join(tweets_df["Tweet"]).split()).most_common(50)
    print(words)


def ShowBarChart():
    figure ,ax = plt.subplots(figsize=(6,6))
    sns.countplot(x='sentiment', data=tweets_df)
    ax.set_xlabel('Sentimens')
    ax.set_ylabel('Percentage of each ')
    ax.set_title('Chatgpt based sentiment analysis Overall')
    plt.show()



def top10Countries():
    # Top 10 countries discussing ChatGPT
    df_1 = tweets_df
    df_country = pd.DataFrame(df_1[df_1['Country'] != 'Unknown']['Country'].value_counts())
    df_country = df_country.reset_index()
    total = df_country['count'].sum()
    percentage = round(df_country['count'] / total * 100, 1)
    df_country['Percentage'] = percentage
    print(df_country[0:10])
    plt.bar(df_country[0:10]['Country'], df_country[0:10]['Percentage'])
    plt.xlabel('Countries')
    plt.ylabel('Percentage')
    plt.title('Top 10 countries discussing about ChatGPT')
    plt.legend()
    plt.show()



#Method to display top five countries with positive,negative and neutral tweets.
def showCountryChart():
    try:
        #Remove country with Unknown value and group by sentiment ,country and count tweets based on sentiments
        group = tweets_df[tweets_df['Country'] != 'Unknown'].groupby(['sentiment', 'Country']).apply(
            lambda x: pd.Series(dict(
                positive_tweets=(x.sentiment == 'Positive').sum(),
                negative_tweets=(x.sentiment == 'Negative').sum(),
                neutral_tweets=(x.sentiment == 'Neutral').sum()
            )))
        # print(group)
        negative_tweets = group['negative_tweets']
        #Sort in descending order and take top 5 countries.
        negative_tweets = negative_tweets[
            negative_tweets.index.get_level_values('sentiment') == 'Negative'].sort_values(
            ascending=False).nlargest(5)
        print(negative_tweets)
        negative_tweets.to_csv('neg.csv')
        df_neg = pd.read_csv('neg.csv')

        #Plot a bar graph
        if df_neg.size > 0:
            sns.barplot(x='Country',
                        y='negative_tweets',
                        data=df_neg)
            plt.xlabel('Country')
            plt.ylabel('Negative Tweets')
            plt.title('Top 5 Countries with most Negative Tweets on ChatGPT')

            plt.show()

        positive_tweets = group['positive_tweets']
        # Sort in descending order and take top 5 countries.
        positive_tweets = positive_tweets[
            positive_tweets.index.get_level_values('sentiment') == 'Positive'].sort_values(
            ascending=False).nlargest(5)
        print(positive_tweets)
        positive_tweets.to_csv('pos.csv')
        df_pos = pd.read_csv('pos.csv')
        # Plot a bar graph
        if df_pos.size > 0:
            sns.barplot(x='Country',
                        y='positive_tweets',
                        data=df_pos)
            plt.xlabel('Country')
            plt.ylabel('Positive Tweets')
            plt.title('Top 5 Countries with most Positive Tweets on ChatGPT')

            plt.show()

        neutral_tweets = group['neutral_tweets']
        # Sort in descending order and take top 5 countries.
        neutral_tweets = neutral_tweets[neutral_tweets.index.get_level_values('sentiment') == 'Neutral'].sort_values(
            ascending=False).nlargest(5)
        print(neutral_tweets)
        neutral_tweets.to_csv('neu.csv')
        df_neu = pd.read_csv('neu.csv')
        # Plot a bar graph
        if df_neu.size > 0:
            sns.barplot(x='Country',
                        y='neutral_tweets',
                        data=df_neu)
            plt.xlabel('Country')
            plt.ylabel('Neutral Tweets')
            plt.title('Top 5 Countries with most Neutral Tweets on ChatGPT')

            plt.show()

    except:
        print("An error occurred while showing country graph")




def wordCloud():
    tweets = " ".join(item for item in tweets_df['Tweet'])
    wc = WordCloud(background_color="white",
                   max_words=500,
                   height = 1600,
                   width=1600).generate(tweets)
    plt.figure(figsize=(10,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()




def PositiveWordCloud():
    positivetweets =tweets_df[tweets_df.sentiment == 'Positive']
    tweets = ' '.join([word for word in positivetweets['Tweet']])
    wc = WordCloud(background_color="white",
                   max_words=500,
                   height=1600,
                   width=1600).generate(tweets)
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def NegativeWordCloud():
    positivetweets =tweets_df[tweets_df.sentiment == 'Negative']
    tweets = ' '.join([word for word in positivetweets['Tweet']])
    wc = WordCloud(background_color="white",
                   max_words=500,
                   height=1600,
                   width=1600).generate(tweets)
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()



if __name__ == "__main__":

    # read from csv first
    country = Country()
    country.fetchCountry()
    tweets_df = pd.read_csv('country.csv')
    print(tweets_df.head().to_string())
    print(tweets_df.columns)
    print("Total unique tweets are ", tweets_df.shape[0])

    tweets_df.Tweet = tweets_df['Tweet'].apply(preprocessing_tweets)
    sentiment_analysis()
    ShowBarChart()
    top10Countries()
    showCountryChart()
    wordCloud()
    PositiveWordCloud()
    NegativeWordCloud()
    getMostCommonWords()












