
## Course Name : CIS 600 Social Media and Data Mining
## Project Name : A Corpus based study of Twitter Sentiments towards ChatGPT
Project By: Deepthi, Gagana, Kalyani, Rahul, Satyajeet, Sushmitha


Sentiment Analysis on ChatGPT

ChatGPT is a language model generates natural language responses to a given prompt or input​

In January 2023, ChatGPT acquired 100 million monthly active users​

User’s Feedback:​

Positive – Expressing gratitude and praising the ability to provide information​

Neutral – Using as a convenient tool without expressing any emotions​

Negative – Expressing frustration and showing concerns over its impact on human employment

Data Collection : Scraping tweets from Twitter

Data Preprocessing : Duplication removal, lowercasing and noise removal (punctuation, stopwords, URLs, @users)

Extracting features : Retrieving geographical info from a user’s profile location and timestamp info

Categorizing and Classifying : Classify tweets into positive, neutral, or negative and Identifying the most discussed topics related to ChatGPT

Data Visualization: Graphically represent the extracted data

## Installation Instructions

1 .Clone the repo first from github .

2 .After cloning ,you need to few pip installs to have all necessary tools .

    pip install seaborn
    pip install matplotlib
    pip install numpy 
    pip install pandas 
    pip install textblob
    pip install nltk
    pip install nrclex
    pip install geopy langdetect
    pip install tqdm certifi 
    pip install googletrans==3.1.0a0
    pip install wordcloud
    pip install re
    pip install collections
    pip install emoji 
    pip install plotly
    
    
    
    
3.After all the libraries are successfully installed, unzip the file and import the project. 

4.Run main.py and 

5.From nltk download popup download all

6.GetMostreq.py to get The 10 most frequently most occurred words in Tweets and Identifying the most frequently discussed topic in twitter about ChatGPT


 





## Screenshots

![App Screenshot](https://github.com/satyajeetkrjha/smdmsyracuse/blob/master/Top10cCountries.png)

![App Screenshot](https://github.com/satyajeetkrjha/smdmsyracuse/blob/master/CountrySpecSenti.png)

![App Screenshot](https://github.com/satyajeetkrjha/smdmsyracuse/blob/master/Top5Neg.png)

![App Screenshot](https://github.com/satyajeetkrjha/smdmsyracuse/blob/master/Top5Neutral.png)

![App Screenshot](https://github.com/satyajeetkrjha/smdmsyracuse/blob/master/Top5Pos.png)

![App Screenshot](https://github.com/satyajeetkrjha/smdmsyracuse/blob/master/overallsentiment.png)

![App Screenshot](https://github.com/satyajeetkrjha/smdmsyracuse/blob/master/AllWordCloud.png)

![App Screenshot](https://github.com/satyajeetkrjha/smdmsyracuse/blob/master/NegWordCloud.png)

![App Screenshot](https://github.com/satyajeetkrjha/smdmsyracuse/blob/master/PosWordCloud.png)

![App Screenshot](https://github.com/satyajeetkrjha/smdmsyracuse/blob/master/Screenshot%202023-04-28%20at%207.41.56%20PM.png)
