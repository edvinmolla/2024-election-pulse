

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

trump_reviews = pd.read_csv("https://raw.githubusercontent.com/edvinmolla/dotfiles/master/Trumpall2.csv")
biden_reviews = pd.read_csv("https://raw.githubusercontent.com/edvinmolla/dotfiles/master/Bidenall2.csv")

textblob1 = TextBlob(trump_reviews["text"][10])
textblob2 = TextBlob(biden_reviews["text"][500])

def find_pol(review):
    return TextBlob(review).sentiment.polarity
trump_reviews["Sentiment Polarity"] = trump_reviews["text"].apply(find_pol)
biden_reviews["Sentiment Polarity"] = biden_reviews["text"].apply(find_pol)

trump_reviews["Expression Label"] = np.where(trump_reviews["Sentiment Polarity"]>0, "positive", "negative")
trump_reviews["Expression Label"][trump_reviews["Sentiment Polarity"]==0]="Neutral"


biden_reviews["Expression Label"] = np.where(biden_reviews["Sentiment Polarity"]>0, "positive", "negative")
biden_reviews["Expression Label"][trump_reviews["Sentiment Polarity"]==0]="Neutral"


reviews1 = trump_reviews[trump_reviews['Sentiment Polarity'] == 0.0000]
cond1=trump_reviews['Sentiment Polarity'].isin(reviews1['Sentiment Polarity'])
trump_reviews.drop(trump_reviews[cond1].index, inplace = True)


reviews2 = biden_reviews[biden_reviews['Sentiment Polarity'] == 0.0000]
cond2=biden_reviews['Sentiment Polarity'].isin(reviews1['Sentiment Polarity'])
biden_reviews.drop(biden_reviews[cond2].index, inplace = True)


# Donald Trump
np.random.seed(10)
remove_n =324
drop_indices = np.random.choice(trump_reviews.index, remove_n, replace=False)
df_subset_trump = trump_reviews.drop(drop_indices)

# Joe Biden
np.random.seed(10)
remove_n =31
drop_indices = np.random.choice(biden_reviews.index, remove_n, replace=False)
df_subset_biden = biden_reviews.drop(drop_indices)


count_1 = df_subset_trump.groupby('Expression Label').count()
negative_per1 = (count_1['Sentiment Polarity'][0]/1000)*10
positive_per1 = (count_1['Sentiment Polarity'][1]/1000)*100

count_2 = df_subset_biden.groupby('Expression Label').count()
negative_per2 = (count_2['Sentiment Polarity'][0]/1000)*100
positive_per2 = (count_2['Sentiment Polarity'][1]/1000)*100

Politicians = ['Joe Biden', 'Donald Trump']
lis_pos = [positive_per1, positive_per2]
lis_neg = [negative_per1, negative_per2]

fig = go.Figure(data=[
    go.Bar(name='Positive', x=Politicians, y=lis_pos),
    go.Bar(name='Negative', x=Politicians, y=lis_neg)
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()

