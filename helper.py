from urlextract import URLExtract
# from gensim.parsing.preprocessing import remove_stopwords
import calendar
from wordcloud import WordCloud
import pandas as pd
from collections import Counter

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)


def monthly_timeline(selected_user,df,k=None):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    if k is not None:
        df = df[df['predicted_sentiment']==k]

    monthly_timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(monthly_timeline.shape[0]):
        time.append(monthly_timeline['month'][i] + "-" + str(monthly_timeline['year'][i]))

    monthly_timeline['time'] = time

    return monthly_timeline


def daily_timeline(selected_user,df,k=None):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    if k is not None:
        df = df[df['predicted_sentiment']==k]

    daily_timeline = df.groupby('date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user,df,k=None):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    if k is not None:
        df = df[df['predicted_sentiment'] == k]

    return df['day_name'].value_counts()


def month_activity_map(selected_user,df,k=None):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    if k is not None:
        df = df[df['predicted_sentiment'] == k]

    return df['month'].value_counts()


def activity_heatmap(selected_user,df,k=None):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    if k is not None:
        df = df[df['predicted_sentiment'] == k]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    day_order = list(calendar.day_name)
    user_heatmap = user_heatmap.reindex(day_order)
    user_heatmap = user_heatmap.dropna(how="all")
    sorted_period = sorted(user_heatmap.columns, key=lambda x: int(x.split('-')[0]))
    user_heatmap = user_heatmap[sorted_period]

    return user_heatmap


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'user': 'User', 'count': 'Percentage'})
    return x,df

def percentage(df,k):
    df = round((df['user'][df['predicted_sentiment']==k].value_counts() / df[df['predicted_sentiment']==k].shape[0]) * 100, 2).reset_index().rename(columns={'user': 'User', 'count': 'Percentage'})
    return df

def contribution(df,k):
    return df['user'][df['predicted_sentiment'] == k].value_counts().head()

def remove_stop_words(df,k=None):
    new_df = df[df['user'] != 'group_notification']
    new_df = new_df[new_df['message'] != '<Media omitted>\n']

    f = open('stop_words.txt', 'r')
    stop_words = f.read()
    words = []

    if k is not None:
        for message in new_df['message'][new_df['predicted_sentiment'] == k]:
            for word in message.lower().split():
                if word not in stop_words:
                    words.append(word)
    else:
        for message in new_df['message']:
            for word in message.lower().split():
                if word not in stop_words:
                    words.append(word)

    return words


def create_wordcloud(selected_user,df,k=None):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    words = remove_stop_words(df,k)

    df_wc = wc.generate(" ".join(words))
    return df_wc


def most_common_words(selected_user,df,k=None):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    words = remove_stop_words(df,k)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df
