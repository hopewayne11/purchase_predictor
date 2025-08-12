import pandas as pd
import numpy as np
import datetime
#from textblob import TextBlob
from pytrends.request import TrendReq
import snscrape.modules.twitter as sntwitter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#simulate dataset
np.random.seed(42)
df= pd.DataFrame({
    "search_volume": np.random.randint(20, 100, 100),
    "trend_growth": np.random.uniform(-10, 20, 100),
    "tweet_count": np.random.randint(100, 5000, 100),
    "tweet_sentiment": np.random.uniform(-1, 1, 100),
    "reddit_mentions": np.random.randint(0, 301, 100),
    "location_score": np.random.uniform(0,1,100),
    "bought": np.random.choice([0,1], 100, p=[0.6, 0.4])
})

#Train Model
x= df.drop("bought", axis=1)
y= df["bought"]
#x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

model= LogisticRegression()
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)

#Evaluate
print("\n=== MODEL TRAINING REPORT ===")
print("Accuracy:", round(accuracy_score(y_test, model.predict(x_test))*100,2),"%")

#Live features extractor
def get_live_features(product_name, country_code="", city_name=""):
    #print(f"Fetching live data for: {product_name} ({country_code or "Worldwide"}) {city_name}...")
    
    #Google Trends
    pytrends= TrendReq()
    try:
        pytrends.build_payload([product_name], timeframe="now 7-d", geo=country_code)
        interest_df= pytrends.interest_over_time()
        if not interest_df.empty:
            search_volume= int(interest_df[product_name].mean())
            trend_growth= ((interest_df[product_name].iloc[-1]- interest_df[product_name].iloc[0])/max(1, interest_df[product_name].iloc[0]))*100
        else:
            search_volume= 0
            trend_growth= 0
            
    except Exception as e:
        print("Google Trends error:", e)
        search_volume= 0
        trend_growth= 0
        
    #Twitter scraping
    tweet_count= 0
    sentiment_total= 0
    tweet_limit= 100
    date_since= (datetime.date.today()-datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    
    try:
        query= f"{product_name} {city_name}".strip()+f"since: {date_since}"
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >=tweet_limit:
                break
            tweet_count+=1
            #sentiment_total+= TextBlob(tweet.content).sentiment.polarity
    except Exception as e:
        print("Twitter Scraping error", e)
        
    tweet_sentiment= round(sentiment_total/tweet_count, 3) if tweet_count>0 else 0
    location_score= round(np.clip(tweet_count/100,0,1), 2)
    reddit_mentions= np.random.randint(0, 300)
    
    
    features= pd.DataFrame([{
        "search_volume": search_volume,
        "trend_growth": trend_growth,
        "tweet_count": tweet_count,
        "tweet_sentiment": tweet_sentiment,
        "reddit_mentions": reddit_mentions,
        "location_score": location_score
        
    }])
    
    print("Live data collected:")
    print(features)
    return features

while True:
    print("\n============ PRODUCT PREDICTOR ==============")
    product= input("Enter a product name( or type exit to quit): ").strip()
    if product.lower()== "exit":
        print("Goodbye!")
        break
    
    location_choice= input("\n Choose location mode:\n[1] Global \n[2] Country+City \nChoice(1/2): ").strip()
    
    if location_choice== "2":
        country= input("Enter 2-letter country code(e.g ZM): ").strip().upper()
        city= input("Enter city name(e.g Lusaka): ").strip()
    else:
        country= ""
        city= ""
        
    features= get_live_features(product, country, city)
    
    try:
        prob= model.predict_proba(features)[0][1]
        print(f"\n Predicted Purchase Likelihood for '{product}': {round(prob*100,2)}%")
    except Exception as e:
        print("Prediction failed:", e)