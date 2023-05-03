#Collect Data from snscrape to CSV file
try:
    import pandas as pd
    import json
    import snscrape.modules.twitter as sntwitter
    import pandas as pd
except Exception as e:
    print("Some Modules are Missing ",e)

#Method to collect twitter data
class tweetCollection(object):
    query = 'ChatGPT since:2022-12-01 until:2023-04-11 , lang:en'
    limit = 50000
    tweets_1 = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets_1) == limit:
            break
        else:
            tweets_1.append([tweet.id,
                             tweet.user.username,
                             tweet.user.verified,
                             tweet.user.location,
                             tweet.user.followersCount,
                             tweet.rawContent,
                             tweet.date,
                             tweet.retweetedTweet,
                             tweet.lang])
    df_1 = pd.DataFrame(tweets_1, columns=['UserId',
                                           'UserName',
                                           'Verified',
                                           'Location',
                                           'Followers',
                                           'Tweet',
                                           'Timestamp',
                                           'Retweeted',
                                           'Language'])
    df_1.to_csv('ChatGPT.csv', index=False)

if __name__ == "__main__":
    tweetCollection()