import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import concurrent.futures
from geopy.geocoders import Nominatim
import certifi
import ssl
import geopy.geocoders
from langdetect import detect
from googletrans import Translator
ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx

class Country:
    def __init__(self):
        pass
    def fetchCountry(self):
        df = pd.read_csv('Twitter_Data.csv')
        df.shape
        df['Location'].fillna(value='Unknown',inplace=True)
        # Unkown location's country is assigned as "Unknown"
        df.loc[df['Location']=='Unknown','Country'] = 'Unknown'
        df_country = df[df['Country']!='Unknown'].copy()
        df_country.shape
        df_country.Country.value_counts()
        locator = Nominatim(user_agent = 'myLocation')
        translator = Translator()

        #Create cache or load if exists
        cache_path = ('geocode_cache.pickle')
        if os.path.exists(cache_path) and os.path.getsize(cache_path)>0:
             with open(cache_path, 'rb') as f:
                geocode_cache = pickle.load(f)
        else:
            geocode_cache = {}
            geocode_cache['Unknown'] = 'Unknown'


        def get_country(location):
            # check if cache has result stored
            try:
                if location in geocode_cache:
                    return geocode_cache[location]
                else:
                      loc = locator.geocode(location)
                      if loc:
                          country = loc.raw['display_name'].split(',')[-1].strip()
                          # store in cache
                          geocode_cache[location] = country
                          return country
                      else:
                          return "Unknown"
            except:
                      return "Unknown"


        # function for batch processing
        def process_locations(df, max_workers, batch_size=100):
            locations = df['Location'].to_list()
            n = len(locations)
            country_list = ['Unknown'] * n
            batches = [(i, locations[i:i + batch_size]) for i in range(0, n, batch_size)]

            # process each batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(process_batch, idx, batch): batch for idx, batch in batches}
                for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(future_to_batch)):
                    start_index, batch_result = future.result()
                    for i, country in enumerate(batch_result):
                        idx = i + start_index
                        if idx < n:
                            country_list[idx] = country
            # store the geocode cache to file
            with open(cache_path, 'wb') as f:
                pickle.dump(geocode_cache, f)
            return country_list

        #function to process each batch
        def process_batch(start_idx, batch):
            batch_result = [get_country(location) for location in batch]
            return start_idx, batch_result

        df_2 = df[0:].copy()
        df_2['Country'] = process_locations(df_2,10)
        df_1 = df.copy()
        # map the values in 'location' column to the corresponding country using the geocode_cache file
        df_1['Country'] = df_1['Location'].map(geocode_cache)
        df_1['Country'] = df_1['Country'].fillna('Unknown')
        df_1['Country'].value_counts()

        with open('geocode_cache.pickle', 'rb') as f:
            country_dict = pickle.load(f)
        df_1['Country_name'] = df_1['Country'].map(country_dict).fillna(df_1['Country'])
        df_1 = df_1.drop(columns=['Country'])
        df_1 = df_1.rename(columns={'Country_name': 'Country'})
        df_country_2 = df_2[['User','Timestamp', 'Tweet', 'Country']].copy()

        for i, row in df_country_2.iterrows():
            tweet_text = row['Country']
            try:
                language = detect(tweet_text)
                if language != 'en':
                    tweet_text = translator.translate(tweet_text, dest='en').text
                df_country_2.at[i, 'Country'] = tweet_text
            except:
                pass
        df_country_2.to_csv('country.csv')
