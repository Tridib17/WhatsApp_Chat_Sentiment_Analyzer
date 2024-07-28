import customtkinter as ctk
import tkinter as tk
import helper
import re
import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import time

from CTkXYFrame import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandastable import Table
from tkinter import ttk
from tkinter import messagebox

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


class SentimentAnalyzer(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 1280) // 2
        y = (screen_height - 720) // 2
        self.geometry(f"{1280}x{720}+{x}+{y}")
        self.minsize(1280, 720)
        self.title("Whatsapp Chat Sentiment Analyzer")

        self.frame = ctk.CTkFrame(self)
        self.frame.pack(pady=20, padx=60, fill="both", expand=True)
        
        self.df = self.temp()
        # self.df = parent.df
        print(self.df)

        # fetch unique users
        user_list = self.df['user'].unique().tolist()
        user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0,"Overall")


        self.selected_user_frame = ctk.CTkFrame(self.frame)
        self.selected_user_frame.pack(side="left", fill="y", padx=(0, 10))

        self.stats = ctk.CTkLabel(self.selected_user_frame, text="Show analysis w.r.t.", font=ctk.CTkFont(size=20, weight="bold"))
        # self.stats.pack(expand=True)
        self.stats.place(relx=0.5, rely=0.45, anchor="center")
        self.selected_user_entry = ctk.CTkOptionMenu(self.selected_user_frame, values=user_list, command=self.select_user)
        # self.selected_user_entry.pack(expand=True)
        self.selected_user_entry.place(relx=0.5, rely=0.49, anchor="center")
        self.show_analysis_button = ctk.CTkButton(self.selected_user_frame, text="Show Analysis", command=self.show_chat_sentiment_analysis)
        self.show_analysis_button.place(relx=0.5, rely=0.54, anchor="center")

        self.select_user()
        self.progress_value = 0
        self.processing_complete = False
        
        self.update()
        self.focus_force()


    def temp(self):
        file = open("WhatsApp Chat with SXC MCMS 2k23.txt", 'r', encoding='utf8')
        self.data = file.read()
        file.close()

        pattern = '\[?\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?(?:\s(?:AM|PM|am|pm))?\]?(?:\s-\s)?'
        messages = re.split(pattern, self.data)[1:]
        messages = [message for message in messages if message != None]
        dates = re.findall(pattern, self.data)
        dates = [date.strip('[ ] - ') for date in dates]

        self.df = pd.DataFrame({'user_message': messages, 'message_date': dates})

        # convert message_date type
        self.df['message_date'] = pd.to_datetime(self.df['message_date'], format='mixed', dayfirst=True)

        self.df.rename(columns={'message_date': 'date'}, inplace=True)

        users = []
        messages = []
        for message in self.df['user_message']:
            entry = re.split('([\w\W]+?):\s', message)
            if entry[1:]:  # user name
                users.append(entry[1])
                messages.append(" ".join(entry[2:]))
            else:
                users.append('group_notification')
                messages.append(entry[0])

        self.df['user'] = users
        self.df['message'] = messages
        self.df.drop(columns=['user_message'], inplace=True)

        self.df['only_date'] = self.df['date'].dt.date
        self.df['year'] = self.df['date'].dt.year
        self.df['month_num'] = self.df['date'].dt.month
        self.df['month'] = self.df['date'].dt.month_name()
        self.df['day'] = self.df['date'].dt.day
        self.df['day_name'] = self.df['date'].dt.day_name()
        self.df['hour'] = self.df['date'].dt.hour
        self.df['minute'] = self.df['date'].dt.minute
        self.df['second'] = self.df['date'].dt.second
        self.df.drop(columns=['date'], inplace=True)
        self.df.rename(columns={'only_date': 'date'}, inplace=True)

        period = []
        for hour in self.df[['day_name', 'hour']]['hour']:
            if hour == 23:
                period.append(str(hour) + "-" + str('00'))
            elif hour == 0:
                period.append(str('00') + "-" + str(hour + 1))
            else:
                period.append(str(hour) + "-" + str(hour + 1))

        self.df['period'] = period
        return self.df


    def model_creation(self):
        # Remove entries having user as group_notification
        self.df = self.df[self.df['user'] != 'group_notification']
        
        # Function to clean text data
        def clean_text(text):
            # Convert to lowercase
            text = text.lower()
            
            # Tokenize
            words = word_tokenize(text)
            
            # Remove special characters
            words = [re.sub(f"[{re.escape(string.punctuation)}]", "", word) for word in words]
            
            # Remove stop words
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words and word != '']
            
            # Stemming
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
            
            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
            
            return ' '.join(words)

        # Apply text cleaning
        self.df['cleaned_message'] = self.df['message'].apply(clean_text)

        # Function to label data using VADER
        sid = SentimentIntensityAnalyzer()

        def label_data_with_vader(message):
            scores = sid.polarity_scores(message)
            if scores['compound'] >= 0.05:
                return 1
            elif scores['compound'] <= -0.05:
                return -1
            else:
                return 0

        # Apply labeling function to a subset of the data (20% of the data for labeling)
        subset_df = self.df.sample(frac=0.2, random_state=42)
        subset_df['sentiment'] = subset_df['message'].apply(label_data_with_vader)

        # Include the labeled subset in the main dataframe
        self.df.loc[subset_df.index, 'sentiment'] = subset_df['sentiment']

        # Drop rows without sentiment labels (NaNs)
        labeled_df = self.df.dropna(subset=['sentiment'])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            labeled_df['cleaned_message'].tolist(), 
            labeled_df['sentiment'].tolist(), 
            test_size=0.2, 
            random_state=42
        )

        # Train the custom sentiment analysis model
        def create_vocabulary_and_frequencies(messages):
            vocabulary = set()
            word_freqs = {}
            for message in messages:
                words = message.split()
                vocabulary.update(words)
                for word in words:
                    if word not in word_freqs:
                        word_freqs[word] = 0
                    word_freqs[word] += 1
            return vocabulary, word_freqs

        def calculate_probabilities(messages, sentiments, vocabulary):
            total_messages = len(messages)
            total_positive = sum(1 for sentiment in sentiments if sentiment == 1)
            total_negative = sum(1 for sentiment in sentiments if sentiment == -1)
            total_neutral = sum(1 for sentiment in sentiments if sentiment == 0)

            prob_positive = total_positive / total_messages
            prob_negative = total_negative / total_messages
            prob_neutral = total_neutral / total_messages

            word_probabilities = {
                'positive': {},
                'negative': {},
                'neutral': {}
            }

            for word in vocabulary:
                word_in_positive = sum(1 for message, sentiment in zip(messages, sentiments) if word in message and sentiment == 1)
                word_in_negative = sum(1 for message, sentiment in zip(messages, sentiments) if word in message and sentiment == -1)
                word_in_neutral = sum(1 for message, sentiment in zip(messages, sentiments) if word in message and sentiment == 0)

                word_probabilities['positive'][word] = (word_in_positive + 1) / (total_positive + len(vocabulary))
                word_probabilities['negative'][word] = (word_in_negative + 1) / (total_negative + len(vocabulary))
                word_probabilities['neutral'][word] = (word_in_neutral + 1) / (total_neutral + len(vocabulary))

            return prob_positive, prob_negative, prob_neutral, word_probabilities

        vocabulary, word_freqs = create_vocabulary_and_frequencies(X_train)
        prob_positive, prob_negative, prob_neutral, word_probabilities = calculate_probabilities(X_train, y_train, vocabulary)

        # Prediction function
        def predict_sentiment(message, vocabulary, prob_positive, prob_negative, prob_neutral, word_probabilities):
            words = message.split()
            log_prob_positive = np.log(prob_positive)
            log_prob_negative = np.log(prob_negative)
            log_prob_neutral = np.log(prob_neutral)

            for word in words:
                if word in vocabulary:
                    log_prob_positive += np.log(word_probabilities['positive'].get(word, 1 / (len(vocabulary) + len(X_train))))
                    log_prob_negative += np.log(word_probabilities['negative'].get(word, 1 / (len(vocabulary) + len(X_train))))
                    log_prob_neutral += np.log(word_probabilities['neutral'].get(word, 1 / (len(vocabulary) + len(X_train))))

            if log_prob_positive > log_prob_negative and log_prob_positive > log_prob_neutral:
                return 1
            elif log_prob_negative > log_prob_positive and log_prob_negative > log_prob_neutral:
                return -1
            else:
                return 0

        # Evaluate the model
        correct_predictions = 0
        for message, actual_sentiment in zip(X_test, y_test):
            predicted_sentiment = predict_sentiment(message, vocabulary, prob_positive, prob_negative, prob_neutral, word_probabilities)
            if predicted_sentiment == actual_sentiment:
                correct_predictions += 1

        accuracy = correct_predictions / len(y_test)
        print(f"Model Accuracy: {accuracy*100:.2f}")

        # Adding the sentiment column to the original dataframe using the trained model
        self.df['predicted_sentiment'] = self.df['cleaned_message'].apply(lambda x: predict_sentiment(x, vocabulary, prob_positive, prob_negative, prob_neutral, word_probabilities))
        print(self.df[['message', 'predicted_sentiment']])


    def timeline(self):
        # Create a scrollable frame inside the "Timeline" tab
        self.timeline_frame = CTkXYFrame(self.tabview.tab("Timeline"))
        self.timeline_frame.pack(fill="both", expand=True)

        # Monthly Timeline
        self.monthly_timeline_label = ctk.CTkLabel(
            self.timeline_frame, text="Monthly Timeline", font=ctk.CTkFont(size=30, weight="bold")
        )
        self.monthly_timeline_label.pack(fill="x")

        sentiments = ['positive', 'neutral', 'negative']
        sentiment_values = [1, 0, -1]
        colors = ['green', 'blue', 'red']

        # Retrieve and filter data for Monthly Timeline
        monthly_timeline_data = []
        for value in sentiment_values:
            monthly_timeline = helper.monthly_timeline(self.selected_user, self.df, value)
            if not monthly_timeline.empty:
                monthly_timeline_data.append((monthly_timeline, value))

        # Plot Monthly Timeline
        num_monthly_subplots = len(monthly_timeline_data)
        self.monthly_timeline_fig, self.monthly_timeline_ax = plt.subplots(1, num_monthly_subplots, figsize=(7 * num_monthly_subplots, 7))
        if num_monthly_subplots == 1:
            self.monthly_timeline_ax = [self.monthly_timeline_ax]
        for ax, (data, value) in zip(self.monthly_timeline_ax, monthly_timeline_data):
            sentiment = sentiments[sentiment_values.index(value)]
            ax.clear()
            ax.plot(data['time'], data['message'], color=colors[sentiment_values.index(value)])
            ax.tick_params(axis='x', rotation=90)
            ax.set_title(f'{sentiment.capitalize()} Messages')
            ax.set_xlabel('Months')
            ax.set_ylabel('Messages')
            ax.xaxis.labelpad = 20
            ax.yaxis.labelpad = 20

        self.monthly_timeline_fig.tight_layout(pad=5)
        self.monthly_timeline_canvas = FigureCanvasTkAgg(self.monthly_timeline_fig, master=self.timeline_frame)
        self.monthly_timeline_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)
        self.monthly_timeline_canvas.draw()

        # Daily Timeline
        self.daily_timeline_label = ctk.CTkLabel(
            self.timeline_frame, text="Daily Timeline", font=ctk.CTkFont(size=30, weight="bold")
        )
        self.daily_timeline_label.pack(pady=(50, 0), fill="x")

        # Retrieve and filter data for Daily Timeline
        daily_timeline_data = []
        for value in sentiment_values:
            daily_timeline = helper.daily_timeline(self.selected_user, self.df, value)
            if not daily_timeline.empty:
                daily_timeline_data.append((daily_timeline, value))

        # Plot Daily Timeline
        num_daily_subplots = len(daily_timeline_data)
        self.daily_timeline_fig, self.daily_timeline_ax = plt.subplots(1, num_daily_subplots, figsize=(7 * num_daily_subplots, 7))
        if num_daily_subplots == 1:
            self.daily_timeline_ax = [self.daily_timeline_ax]
        for ax, (data, value) in zip(self.daily_timeline_ax, daily_timeline_data):
            sentiment = sentiments[sentiment_values.index(value)]
            ax.clear()
            ax.plot(data['date'], data['message'], color=colors[sentiment_values.index(value)])
            ax.tick_params(axis='x', rotation=90)
            ax.set_title(f'{sentiment.capitalize()} Messages')
            ax.set_xlabel('Days')
            ax.set_ylabel('Messages')
            ax.xaxis.labelpad = 20
            ax.yaxis.labelpad = 20

        self.daily_timeline_fig.tight_layout(pad=5)
        self.daily_timeline_canvas = FigureCanvasTkAgg(self.daily_timeline_fig, master=self.timeline_frame)
        self.daily_timeline_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)
        self.daily_timeline_canvas.draw()


    def activity_map(self):
        # Create a scrollable frame inside the "Activity Map" tab
        self.activity_map_frame = CTkXYFrame(self.tabview.tab("Activity Map"))
        self.activity_map_frame.pack(fill="both", expand=True)

        self.monthly_activity_map_label = ctk.CTkLabel(
            self.activity_map_frame, text="Monthly Activity Map", font=ctk.CTkFont(size=30, weight="bold")
        )
        self.monthly_activity_map_label.pack(fill="x")

        sentiments = ['positive', 'neutral', 'negative']
        sentiment_values = [1, 0, -1]
        colors = ['green', 'blue', 'red']

        # Retrieve and filter data for Most Busy Month
        monthly_activity_map_data = []
        for value in sentiment_values:
            month = helper.month_activity_map(self.selected_user, self.df, value)
            if not month.empty:
                monthly_activity_map_data.append(month)

        # Plot Most Busy Month
        monthly_activity_map_subplots = len(monthly_activity_map_data)
        self.monthly_activity_map_fig, self.monthly_activity_map_ax = plt.subplots(1, monthly_activity_map_subplots, figsize=(6 * monthly_activity_map_subplots, 6))
        if monthly_activity_map_subplots == 1:
            self.monthly_activity_map_ax = [self.monthly_activity_map_ax]
        for ax, data, sentiment, color in zip(self.monthly_activity_map_ax, monthly_activity_map_data, sentiments, colors):
            ax.clear()
            ax.bar(data.index, data.values, color=color)
            ax.set_title(f'{sentiment.capitalize()} Messages')
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlabel('Months')
            ax.set_ylabel('Messages')
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 20

        self.monthly_activity_map_fig.tight_layout(pad=5)
        self.monthly_activity_map_canvas = FigureCanvasTkAgg(self.monthly_activity_map_fig, master=self.activity_map_frame)
        self.monthly_activity_map_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)
        self.monthly_activity_map_canvas.draw()


        self.daily_activity_map_label = ctk.CTkLabel(
            self.activity_map_frame, text="Daily Activity Map", font=ctk.CTkFont(size=30, weight="bold")
        )
        self.daily_activity_map_label.pack(pady=(50, 0), fill="x")

        # Retrieve and filter data for Most Busy Day
        daily_activity_map_data = []
        for value in sentiment_values:
            day = helper.week_activity_map(self.selected_user, self.df, value)
            if not day.empty:
                daily_activity_map_data.append(day)

        # Plot Most Busy Day
        daily_activity_map_subplots = len(daily_activity_map_data)
        self.daily_activity_map_fig, self.daily_activity_map_ax = plt.subplots(1, daily_activity_map_subplots, figsize=(6 * daily_activity_map_subplots, 6))
        if daily_activity_map_subplots == 1:
            self.daily_activity_map_ax = [self.daily_activity_map_ax]
        for ax, data, sentiment, color in zip(self.daily_activity_map_ax, daily_activity_map_data, sentiments, colors):
            ax.clear()
            ax.bar(data.index, data.values, color=color)
            ax.set_title(f'{sentiment.capitalize()} Messages')
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlabel('Days')
            ax.set_ylabel('Messages')
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 20

        self.daily_activity_map_fig.tight_layout(pad=5)
        self.daily_activity_map_canvas = FigureCanvasTkAgg(self.daily_activity_map_fig, master=self.activity_map_frame)
        self.daily_activity_map_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)
        self.daily_activity_map_canvas.draw()


        self.weekly_activity_map_label = ctk.CTkLabel(
            self.activity_map_frame, text="Weekly Activity Map", font=ctk.CTkFont(size=30, weight="bold")
        )
        self.weekly_activity_map_label.pack(pady=(50, 0), fill="x")

        # Retrieve and filter data for Weekly Activity Map
        weekly_activity_data = []
        for value in sentiment_values:
            user_heatmap = helper.activity_heatmap(self.selected_user, self.df, value)
            if not user_heatmap.empty:
                weekly_activity_data.append(user_heatmap)

        # Plot Weekly Activity Map
        num_weekly_activity_subplots = len(weekly_activity_data)
        self.weekly_activity_map_fig, self.weekly_activity_map_ax = plt.subplots(1, num_weekly_activity_subplots, figsize=(6 * num_weekly_activity_subplots, 6))
        if num_weekly_activity_subplots == 1:
            self.weekly_activity_map_ax = [self.weekly_activity_map_ax]

        for ax, data, sentiment in zip(self.weekly_activity_map_ax, weekly_activity_data, sentiments):
            ax.clear()
            sns.heatmap(data, ax=ax, cmap="YlGnBu")
            ax.set_title(f'{sentiment.capitalize()} Messages')
            ax.tick_params(axis='x', rotation=90)
            ax.tick_params(axis='y', rotation=0)
            ax.set_xlabel('Time')
            ax.set_ylabel('Days')
            ax.xaxis.labelpad = 20
            ax.yaxis.labelpad = 20

        self.weekly_activity_map_fig.tight_layout(pad=5)
        self.weekly_activity_map_canvas = FigureCanvasTkAgg(self.weekly_activity_map_fig, master=self.activity_map_frame)
        self.weekly_activity_map_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)
        self.weekly_activity_map_canvas.draw()


    def user_analysis(self):
        self.user_analysis_frame = CTkXYFrame(self.tabview.tab("User Analysis"))
        self.user_analysis_frame.pack(fill="both", expand=True)

        self.chat_contribution_label = ctk.CTkLabel(self.user_analysis_frame, text="Most Busy Users", font=ctk.CTkFont(size=30, weight="bold"))
        self.chat_contribution_label.pack(fill="x")

        sentiments = ['positive', 'neutral', 'negative']
        sentiment_values = [1, 0, -1]
        colors = ['green', 'blue', 'red']

        contribution_data = []
        for value in sentiment_values:
            new_df = helper.contribution(self.df, value)
            if not new_df.empty:
                contribution_data.append(new_df)

        num_subplots = len(contribution_data)
        self.most_busy_users_fig, self.most_busy_users_ax = plt.subplots(1, num_subplots, figsize=(7 * num_subplots, 7))
        if num_subplots == 1:
            self.most_busy_users_ax = [self.most_busy_users_ax]

        for ax, data, sentiment, color in zip(self.most_busy_users_ax, contribution_data, sentiments, colors):
            ax.clear()
            ax.bar(data.index, data.values, color=color)
            ax.tick_params(axis='x', rotation=90)
            ax.set_title(f'{sentiment.capitalize()} Messages')
            ax.set_xlabel('Users')
            ax.set_ylabel('Messages')
            ax.xaxis.labelpad = 20
            ax.yaxis.labelpad = 20

        self.most_busy_users_fig.tight_layout(pad=5)
        self.most_busy_users_canvas = FigureCanvasTkAgg(self.most_busy_users_fig, master=self.user_analysis_frame)
        self.most_busy_users_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)
        self.most_busy_users_canvas.draw()


        self.users_contribution_label = ctk.CTkLabel(self.user_analysis_frame, text="Chat Contribution of Users", font=ctk.CTkFont(size=30, weight="bold"))
        self.users_contribution_label.pack(pady=(50, 0), fill="x")

        self.tables_frame = ctk.CTkFrame(self.user_analysis_frame)
        self.tables_frame.pack(fill="x", expand=True)

        table_data = []
        for value in sentiment_values:
            new_df = helper.percentage(self.df, value)
            if not new_df.empty:
                table_data.append((new_df, value))

        if table_data:
            for new_df, value in table_data:
                sentiment = sentiments[sentiment_values.index(value)]
                table_frame = ctk.CTkFrame(self.tables_frame)
                table_frame.pack(side="left", pady=10, fill="x", expand=True)
                table_label = ctk.CTkLabel(
                    table_frame, text=f"Most {sentiment.capitalize()} Contribution", font=ctk.CTkFont(size=20, weight="bold")
                )
                table_label.pack(side="top", padx=10, pady=10)
                table_sub_frame = ctk.CTkFrame(table_frame)
                table_sub_frame.pack(side="top", padx=10, pady=10)
                
                headers = ["User", "Percentage"]
                table = Table(table_sub_frame, headers, new_df)
                scrollbar = ttk.Scrollbar(table_sub_frame, orient="vertical", command=table.yview)
                table.configure(yscroll=scrollbar.set)
                table.pack(side="left", fill="y", padx=10, pady=10)
                scrollbar.pack(side="left", fill="y", pady=10)


    def word_analysis(self):
        self.word_analysis_frame = CTkXYFrame(self.tabview.tab("Word Analysis"))
        self.word_analysis_frame.pack(fill="both", expand=True)

        self.wordcloud_label = ctk.CTkLabel(self.word_analysis_frame, text="Wordcloud", font=ctk.CTkFont(size=30, weight="bold"))
        self.wordcloud_label.pack(fill="x")

        sentiments = ['positive', 'neutral', 'negative']
        sentiment_values = [1, 0, -1]

        # Retrieve and filter data for Wordclouds
        wordcloud_data = []
        for value in sentiment_values:
            try:
                df_wc = helper.create_wordcloud(self.selected_user, self.df, value)
                wordcloud_data.append((df_wc, value))
            except:
                pass

        # Plot Wordclouds
        num_wordcloud_subplots = len(wordcloud_data)
        self.wordcloud_fig, self.wordcloud_ax = plt.subplots(1, num_wordcloud_subplots, figsize=(7 * num_wordcloud_subplots, 7))
        if num_wordcloud_subplots == 1:
            self.wordcloud_ax = [self.wordcloud_ax]
        for ax, (data, value) in zip(self.wordcloud_ax, wordcloud_data):
            sentiment = sentiments[sentiment_values.index(value)]
            ax.clear()
            ax.imshow(data)
            ax.set_title(f'{sentiment.capitalize()} Wordcloud')
            ax.set_xticks([])
            ax.set_yticks([])

        self.wordcloud_fig.tight_layout(pad=5)
        self.wordcloud_canvas = FigureCanvasTkAgg(self.wordcloud_fig, master=self.word_analysis_frame)
        self.wordcloud_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)
        self.wordcloud_canvas.draw()

        self.most_common_words_label = ctk.CTkLabel(self.word_analysis_frame, text="Most common words", font=ctk.CTkFont(size=30, weight="bold"))
        self.most_common_words_label.pack(pady=(50, 0), fill="x")

        # Retrieve and filter data for Most Common Words
        most_common_words_data = []
        for value in sentiment_values:
            most_common_words_df = helper.most_common_words(self.selected_user, self.df, value)
            if not most_common_words_df.empty:
                most_common_words_data.append((most_common_words_df, value))

        # Plot Most Common Words
        num_most_common_words_subplots = len(most_common_words_data)
        self.most_common_words_fig, self.most_common_words_ax = plt.subplots(1, num_most_common_words_subplots, figsize=(7 * num_most_common_words_subplots, 7))
        if num_most_common_words_subplots == 1:
            self.most_common_words_ax = [self.most_common_words_ax]
        for ax, (data, value) in zip(self.most_common_words_ax, most_common_words_data):
            sentiment = sentiments[sentiment_values.index(value)]
            ax.clear()
            ax.barh(data[0], data[1])
            ax.set_title(f'{sentiment.capitalize()} Most Common Words')
            ax.set_xlabel('Count')
            ax.set_ylabel('Words')
            ax.xaxis.labelpad = 20
            ax.yaxis.labelpad = 10

        self.most_common_words_fig.tight_layout(pad=5)
        self.most_common_words_canvas = FigureCanvasTkAgg(self.most_common_words_fig, master=self.word_analysis_frame)
        self.most_common_words_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)
        self.most_common_words_canvas.draw()


    def start_loading_animation(self):
        self.loading_frame = ctk.CTkFrame(self.frame)
        self.loading_frame.pack(side="bottom", pady=(20, 0), fill="x")
        self.loading_label = ctk.CTkLabel(self.loading_frame, text="Loading...", font=ctk.CTkFont(size=20, weight="bold"))
        self.loading_label.pack(pady=10)
        self.progress = ctk.CTkProgressBar(self.loading_frame, orientation="horizontal", mode="determinate")
        self.progress.pack(pady=10)
        self.progress.set(0)
        self.update_idletasks()  # Ensure the loading animation is displayed
    
    def stop_loading_animation(self):
        self.loading_frame.pack_forget()

    def update_progress(self, value):
        if value >= 80:
            self.loading_label.configure(text="Finalizing...")
        self.progress.set(value / 100)

    def show_success_message(self):
        messagebox.showinfo("Success", "Analysis loaded successfully!")

    def select_user(self, user="Overall"):
        self.selected_user = user

    def show_chat_sentiment_analysis(self):
        if self.selected_user is None:
            messagebox.showwarning("Warning", "Please select a user.")
            return

        self.show_analysis_button.configure(state="disabled")
        self.start_loading_animation()

        # Run data processing in a separate thread
        thread = threading.Thread(target=self._process_chat_sentiment_analysis)
        thread.start()

    def _process_chat_sentiment_analysis(self):
        # Destroy existing frames if any
        for widget in self.frame.winfo_children():
            if (hasattr(self, "tabview") and widget == self.tabview):
                widget.pack_forget()

        steps = 5
        for i in range(steps):
            time.sleep(1)  # Simulate time-consuming processing step
            progress = int((i + 1) / steps * 100)
            self.after(0, self.update_progress, progress)

        self.after(0, self.finalize_analysis)

    def finalize_analysis(self):
        self.model_creation()

        self.tabview = ctk.CTkTabview(self.frame)
        self.tabview.pack(fill="both", expand=True)
        self.tabview._segmented_button.configure(font=ctk.CTkFont(size=30, weight="bold"))
        self.tabview.add("Timeline")
        self.tabview.add("Activity Map")
        if self.selected_user == "Overall":
            self.tabview.add("User Analysis")
        self.tabview.add("Word Analysis")

        self.timeline()
        self.activity_map()

        if self.selected_user != "Overall":
            try:
                if self.tabview.tab("User Analysis").winfo_exists():
                    self.tabview.delete("User Analysis")
            except Exception:
                pass
        else:
            self.user_analysis()

        self.word_analysis()

        self.stop_loading_animation()
        self.show_success_message()
        self.show_analysis_button.configure(state="normal")


class Table(ttk.Treeview):
    def __init__(self, master, headers, dataframe):
        super().__init__(master, columns=headers, show="headings")
        self.dataframe = dataframe
        self.tree = None
        self.sort_column = None
        self.sort_ascending = True

        # Create Treeview style
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"))
        self.create_table()

    def create_table(self):
        # Define columns
        for col in self.dataframe.columns:
            self.heading(col, text=col, anchor="center")
            self.column(col, anchor="center", stretch=True)

        # Insert data into the treeview
        for _, row in self.dataframe.iterrows():
            self.insert('', 'end', values=list(row))

        # Bind double-click event to column headers for sorting
        self.bind('<Double-1>', self.on_double_click)

    def on_double_click(self, event):
        # Identify the region clicked
        region = self.identify_region(event.x, event.y)
        if region == "heading":
            column = self.identify_column(event.x)
            column = self.heading(column)['text']
            self.sort_by(column)

    def sort_by(self, column):
        if self.sort_column == column:
            self.sort_ascending = not self.sort_ascending
        else:
            self.sort_ascending = True

        self.sort_column = column

        sorted_df = self.dataframe.sort_values(by=column, ascending=self.sort_ascending)
        self.update_table(sorted_df)

    def update_table(self, dataframe):
        # Clear current data
        for item in self.get_children():
            self.delete(item)

        # Insert new data
        for _, row in dataframe.iterrows():
            self.insert('', 'end', values=list(row))


if __name__ == "__main__":
    app = SentimentAnalyzer(ctk.CTk())

    app.mainloop()
