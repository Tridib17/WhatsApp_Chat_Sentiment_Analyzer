import customtkinter as ctk
import tkinter as tk
import helper
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import threading
import time
from tkinter import messagebox
from tkinter import ttk

class ChatAnalyzer(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 1280) // 2
        y = (screen_height - 720) // 2
        self.geometry(f"{1280}x{720}+{x}+{y}")
        self.minsize(1280, 720)
        self.title("Whatsapp Chat Analyzer")

        self.frame = ctk.CTkFrame(self)
        self.frame.pack(pady=20, padx=60, fill="both", expand=True)
        
        # self.df = self.temp()
        self.df = parent.df
        print(self.df)

        # fetch unique users
        user_list = self.df['user'].unique().tolist()
        try:
            user_list.remove('group_notification')
        except:
            pass

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
        self.show_analysis_button = ctk.CTkButton(self.selected_user_frame, text="Show Analysis", command=self.show_chat_analysis)
        self.show_analysis_button.place(relx=0.5, rely=0.54, anchor="center")

        # self.tabview.tab("Timeline").grid_columnconfigure(0, weight=1)
        # self.tabview.tab("Activity Map").grid_columnconfigure(0, weight=1)

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



    def show_stats(self):
        self.stats_frame = ctk.CTkFrame(self.frame)
        self.stats_frame.pack(pady=(0, 10), fill="x")
        self.statistics = ctk.CTkLabel(self.stats_frame, text="Top Statistics", font=ctk.CTkFont(size=30, weight="bold"))
        self.statistics.place(relx=0.5, rely=0.2, anchor="center")

        self.total_messages_label = ctk.CTkLabel(self.stats_frame, text="Total Messages", font=ctk.CTkFont(size=25, weight="bold"))
        self.total_messages_label.place(relx=0.2, rely=0.5, anchor="center")
        self.total_messages_entry = ctk.CTkLabel(self.stats_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.total_messages_entry.place(relx=0.2, rely=0.7, anchor="center")

        self.total_words_label = ctk.CTkLabel(self.stats_frame, text="Total Words", font=ctk.CTkFont(size=25, weight="bold"))
        self.total_words_label.place(relx=0.4, rely=0.5, anchor="center")
        self.total_words_entry = ctk.CTkLabel(self.stats_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.total_words_entry.place(relx=0.4, rely=0.7, anchor="center")

        self.media_shared_label = ctk.CTkLabel(self.stats_frame, text="Media Shared", font=ctk.CTkFont(size=25, weight="bold"))
        self.media_shared_label.place(relx=0.6, rely=0.5, anchor="center")
        self.media_shared_entry = ctk.CTkLabel(self.stats_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.media_shared_entry.place(relx=0.6, rely=0.7, anchor="center")

        self.links_shared_label = ctk.CTkLabel(self.stats_frame, text="Links Shared", font=ctk.CTkFont(size=25, weight="bold"))
        self.links_shared_label.place(relx=0.8, rely=0.5, anchor="center")
        self.links_shared_entry = ctk.CTkLabel(self.stats_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.links_shared_entry.place(relx=0.8, rely=0.7, anchor="center")

        num_messages, num_words, num_media_messages, num_links = helper.fetch_stats(self.selected_user, self.df)

        self.total_messages_entry.configure(text=num_messages)
        self.total_words_entry.configure(text=num_words)
        self.media_shared_entry.configure(text=num_media_messages)
        self.links_shared_entry.configure(text=num_links)


    def timeline(self):
        self.timeline_frame = ctk.CTkScrollableFrame(self.tabview.tab("Timeline"))
        self.timeline_frame.pack(fill="both", expand=True)

        self.monthly_timeline_label = ctk.CTkLabel(self.timeline_frame, text="Monthly Timeline", font=ctk.CTkFont(size=30, weight="bold"))
        self.monthly_timeline_label.pack(fill="x")

        self.monthly_timeline_fig, self.monthly_timeline_ax = plt.subplots(figsize=(7, 7))
        self.monthly_timeline_canvas = FigureCanvasTkAgg(self.monthly_timeline_fig, master=self.timeline_frame)
        self.monthly_timeline_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)

        self.daily_timeline_label = ctk.CTkLabel(self.timeline_frame, text="Daily Timeline", font=ctk.CTkFont(size=30, weight="bold"))
        self.daily_timeline_label.pack(pady=(50, 0), fill="x")

        self.daily_timeline_fig, self.daily_timeline_ax = plt.subplots(figsize=(7, 7))
        self.daily_timeline_canvas = FigureCanvasTkAgg(self.daily_timeline_fig, master=self.timeline_frame)
        self.daily_timeline_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)


        monthly_timeline = helper.monthly_timeline(self.selected_user, self.df)
        self.monthly_timeline_ax.clear()  # Clear the previous plot
        self.monthly_timeline_ax.plot(monthly_timeline['time'], monthly_timeline['message'],color='green')
        self.monthly_timeline_ax.tick_params(axis='x', rotation=90)
        self.monthly_timeline_fig.tight_layout(pad=5)
        self.monthly_timeline_ax.set_xlabel('Months')
        self.monthly_timeline_ax.set_ylabel('Messages')
        self.monthly_timeline_ax.xaxis.labelpad = 20
        self.monthly_timeline_ax.yaxis.labelpad = 20
        self.monthly_timeline_canvas.draw()  # Draw the updated figure
        
        daily_timeline = helper.daily_timeline(self.selected_user, self.df)
        self.daily_timeline_ax.clear()  # Clear the previous plot
        self.daily_timeline_ax.plot(daily_timeline['date'], daily_timeline['message'],color='black')
        self.daily_timeline_ax.tick_params(axis='x', rotation=90)
        self.daily_timeline_fig.tight_layout(pad=5)
        self.daily_timeline_ax.set_xlabel('Days')
        self.daily_timeline_ax.set_ylabel('Messages')
        self.daily_timeline_ax.xaxis.labelpad = 20
        self.daily_timeline_ax.yaxis.labelpad = 20
        self.daily_timeline_canvas.draw()  # Draw the updated figure


    def activity_map(self):
        self.activity_map_frame = ctk.CTkScrollableFrame(self.tabview.tab("Activity Map"))
        self.activity_map_frame.pack(fill="both", expand=True)

        self.most_busy_month_label = ctk.CTkLabel(self.activity_map_frame, text="Most busy month", font=ctk.CTkFont(size=30, weight="bold"))
        self.most_busy_month_label.pack(fill="x")
        self.most_busy_month_fig, self.most_busy_month_ax = plt.subplots(figsize=(4, 5))
        self.most_busy_month_canvas = FigureCanvasTkAgg(self.most_busy_month_fig, master=self.activity_map_frame)
        self.most_busy_month_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)

        self.most_busy_day_label = ctk.CTkLabel(self.activity_map_frame, text="Most busy day", font=ctk.CTkFont(size=30, weight="bold"))
        self.most_busy_day_label.pack(pady=(50, 0), fill="x")
        self.most_busy_day_fig, self.most_busy_day_ax = plt.subplots(figsize=(4, 5))
        self.most_busy_day_canvas = FigureCanvasTkAgg(self.most_busy_day_fig, master=self.activity_map_frame)
        self.most_busy_day_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)

        self.weekly_activity_map_label = ctk.CTkLabel(self.activity_map_frame, text="Weekly Activity Map", font=ctk.CTkFont(size=30, weight="bold"))
        self.weekly_activity_map_label.pack(pady=(50, 0), fill="x")

        self.weekly_activity_map_fig, self.weekly_activity_map_ax = plt.subplots(figsize=(13, 7))
        self.weekly_activity_map_canvas = FigureCanvasTkAgg(self.weekly_activity_map_fig, master=self.activity_map_frame)
        self.weekly_activity_map_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)



        busy_month = helper.month_activity_map(self.selected_user, self.df)
        self.most_busy_month_ax.clear()  # Clear the previous plot
        self.most_busy_month_ax.bar(busy_month.index, busy_month.values, color='orange')
        # self.most_busy_month_ax.tick_params(axis='x', rotation=90)
        # self.most_busy_month_fig.tight_layout(pad=5)
        self.most_busy_month_ax.set_xlabel('Months')
        self.most_busy_month_ax.set_ylabel('Messages')
        self.most_busy_month_ax.xaxis.labelpad = 10
        self.most_busy_month_ax.yaxis.labelpad = 20
        self.most_busy_month_canvas.draw()  # Draw the updated figure

        busy_day = helper.week_activity_map(self.selected_user, self.df)
        self.most_busy_day_ax.clear()  # Clear the previous plot
        self.most_busy_day_ax.bar(busy_day.index, busy_day.values, color='purple')
        # self.most_busy_day_ax.tick_params(axis='x', rotation=90)
        # self.most_busy_day_fig.tight_layout(pad=5)
        self.most_busy_day_ax.set_xlabel('Days')
        self.most_busy_day_ax.set_ylabel('Messages')
        self.most_busy_day_ax.xaxis.labelpad = 10
        self.most_busy_day_ax.yaxis.labelpad = 20
        self.most_busy_day_canvas.draw()  # Draw the updated figure

        user_heatmap = helper.activity_heatmap(self.selected_user, self.df)
        self.weekly_activity_map_canvas.get_tk_widget().destroy()
        self.weekly_activity_map_fig, self.weekly_activity_map_ax = plt.subplots(figsize=(13, 8))
        self.weekly_activity_map_canvas = FigureCanvasTkAgg(self.weekly_activity_map_fig, master=self.activity_map_frame)
        self.weekly_activity_map_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)
        sns.heatmap(user_heatmap, ax=self.weekly_activity_map_ax)
        self.weekly_activity_map_ax.tick_params(axis='x', rotation=90)
        self.weekly_activity_map_fig.tight_layout(pad=5)
        self.weekly_activity_map_ax.set_xlabel('Time')
        self.weekly_activity_map_ax.set_ylabel('Days')
        self.weekly_activity_map_ax.xaxis.labelpad = 20
        self.weekly_activity_map_ax.yaxis.labelpad = 20
        self.weekly_activity_map_canvas.draw()
        

    def user_analysis(self):
        self.user_analysis_frame = ctk.CTkScrollableFrame(self.tabview.tab("User Analysis"))
        self.user_analysis_frame.pack(fill="both", expand=True)

        self.most_busy_users_label = ctk.CTkLabel(self.user_analysis_frame, text="Most Busy Users", font=ctk.CTkFont(size=30, weight="bold"))
        self.most_busy_users_label.pack(fill="x")

        self.most_busy_users_fig, self.most_busy_users_ax = plt.subplots(figsize=(7, 7))
        self.most_busy_users_canvas = FigureCanvasTkAgg(self.most_busy_users_fig, master=self.user_analysis_frame)
        self.most_busy_users_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)

        self.users_contribution_label = ctk.CTkLabel(self.user_analysis_frame, text="Chat Percentage", font=ctk.CTkFont(size=30, weight="bold"))
        self.users_contribution_label.pack(pady=(50, 0), fill="x")

        self.table_frame = ctk.CTkFrame(self.user_analysis_frame)
        self.table_frame.pack(padx=50, pady=10, expand=True)

        # finding the busiest users in the group(Group level)
        x, new_df = helper.most_busy_users(self.df)
        
        self.most_busy_users_ax.clear()  # Clear the previous plot
        self.most_busy_users_ax.bar(x.index, x.values, color='red')
        self.most_busy_users_ax.tick_params(axis='x', rotation=90)
        self.most_busy_users_fig.tight_layout(pad=5)
        self.most_busy_users_ax.set_xlabel('Users')
        self.most_busy_users_ax.set_ylabel('Messages')
        self.most_busy_users_ax.xaxis.labelpad = 20
        self.most_busy_users_ax.yaxis.labelpad = 20
        self.most_busy_users_canvas.draw()  # Draw the updated figure

        headers = ["User", "Percentage"]
        self.table = Table(self.table_frame, headers, new_df)
        scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscroll=scrollbar.set)
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")



    def word_analysis(self):
        self.word_analysis_frame = ctk.CTkScrollableFrame(self.tabview.tab("Word Analysis"))
        self.word_analysis_frame.pack(fill="both", expand=True)

        self.wordcloud_label = ctk.CTkLabel(self.word_analysis_frame, text="Wordcloud", font=ctk.CTkFont(size=30, weight="bold"))
        self.wordcloud_label.pack(fill="x")

        self.wordcloud_fig, self.wordcloud_ax = plt.subplots(figsize=(7, 7))
        self.wordcloud_canvas = FigureCanvasTkAgg(self.wordcloud_fig, master=self.word_analysis_frame)
        self.wordcloud_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)

        self.most_common_words_label = ctk.CTkLabel(self.word_analysis_frame, text="Most common words", font=ctk.CTkFont(size=30, weight="bold"))
        self.most_common_words_label.pack(pady=(50, 0), fill="x")

        self.most_common_words_fig, self.most_common_words_ax = plt.subplots(figsize=(7, 7))
        self.most_common_words_canvas = FigureCanvasTkAgg(self.most_common_words_fig, master=self.word_analysis_frame)
        self.most_common_words_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)


        # WordCloud
        df_wc = helper.create_wordcloud(self.selected_user, self.df)

        self.wordcloud_ax.clear()  # Clear the previous plot
        self.wordcloud_ax.imshow(df_wc)
        self.wordcloud_ax.set_xticks([])
        self.wordcloud_ax.set_yticks([])

        # most common words
        most_common_words_df = helper.most_common_words(self.selected_user, self.df)

        self.most_common_words_ax.clear()  # Clear the previous plot
        self.most_common_words_ax.barh(most_common_words_df[0], most_common_words_df[1])
        self.most_common_words_fig.tight_layout(pad=5)
        self.most_common_words_ax.set_xlabel('Count')
        self.most_common_words_ax.set_ylabel('Words')
        self.most_common_words_ax.xaxis.labelpad = 15
        self.most_common_words_ax.yaxis.labelpad = 7
        self.most_common_words_canvas.draw()  # Draw the updated figure



    # def show_chat_analysis(self, selected_user="Overall"):
    #     self.selected_user = selected_user
 
    #     self.show_stats()

    #     self.timeline()

    #     self.activity_map()
 
    #     if self.selected_user != "Overall":
    #         try:
    #             if self.tabview.tab("User Analysis").winfo_exists():
    #                 self.tabview.delete("User Analysis")
    #         except Exception:
    #             pass
    #     else:
    #         try:
    #             if not self.tabview.tab("User Analysis").winfo_exists():
    #                 pass
    #         except Exception:
    #             self.tabview.insert(2, "User Analysis")
            
    #         self.user_analysis_frame = ctk.CTkScrollableFrame(self.tabview.tab("User Analysis"))
    #         self.user_analysis_frame.pack(fill="both", expand=True)

    #         self.most_busy_users_label = ctk.CTkLabel(self.user_analysis_frame, text="Most Busy Users", font=ctk.CTkFont(size=30, weight="bold"))
    #         self.most_busy_users_label.pack(fill="x")

    #         self.most_busy_users_fig, self.most_busy_users_ax = plt.subplots(figsize=(7, 7))
    #         self.most_busy_users_canvas = FigureCanvasTkAgg(self.most_busy_users_fig, master=self.user_analysis_frame)
    #         self.most_busy_users_canvas.get_tk_widget().pack(padx=50, pady=10, fill="both", expand=True)

    #         self.users_contribution_label = ctk.CTkLabel(self.user_analysis_frame, text="Chat Percentage", font=ctk.CTkFont(size=30, weight="bold"))
    #         self.users_contribution_label.pack(pady=(50, 0), fill="x")

    #         self.table_frame = ctk.CTkFrame(self.user_analysis_frame)
    #         self.table_frame.pack()

    #         self.user_analysis()

    #     self.word_analysis()


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

    def show_chat_analysis(self):
        if self.selected_user is None:
            messagebox.showwarning("Warning", "Please select a user.")
            return

        self.show_analysis_button.configure(state="disabled")
        self.start_loading_animation()

        # Run data processing in a separate thread
        thread = threading.Thread(target=self._process_chat_analysis)
        thread.start()

    def _process_chat_analysis(self):
        # Destroy existing frames if any
        for widget in self.frame.winfo_children():
            if (hasattr(self, "stats_frame") and widget == self.stats_frame) or (hasattr(self, "tabview") and widget == self.tabview):
                widget.pack_forget()

        steps = 5
        for i in range(steps):
            time.sleep(1)  # Simulate time-consuming processing step
            progress = int((i + 1) / steps * 100)
            self.after(0, self.update_progress, progress)

        self.after(0, self.finalize_analysis)

    def finalize_analysis(self):
        self.show_stats()

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
    app = ChatAnalyzer(ctk.CTk())

    app.mainloop()
