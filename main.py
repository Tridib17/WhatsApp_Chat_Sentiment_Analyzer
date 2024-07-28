from tkinter import Menu
from tkinter import filedialog, messagebox

import customtkinter as ctk
import tkinter as tk
import threading
import time
import re
import pandas as pd
import chat_analysis
import sentiment_analysis


class ChatSentimentAnalyzer(ctk.CTk):
    def __init__(self):
        super().__init__()
        # self.iconbitmap("icon.ico")
        self.title("Whatsapp Chat and Sentiment Analyzer")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 600) // 2
        y = (screen_height - 400) // 2
        self.geometry(f"{600}x{400}+{x}+{y}")
        self.resizable(False, False)

        self.frame = ctk.CTkFrame(self)
        self.frame.pack(pady=20, padx=60, fill="both", expand=True)

        
        self.update()
        # self.focus_force()
        self.focus_set()
        # self.attributes('-topmost', True)

        # Add controls to the frame
        self.select = ctk.CTkButton(self.frame, text="Select a Whatsapp Chat file", command=self.load_chats)
        self.select.pack(side="top", padx=10, pady=10)
        self.select_name = ctk.CTkLabel(self.frame, text="")
        self.select_name.pack(side="top", padx=10, pady=5)
        self.preprocess_button = ctk.CTkButton(self.frame, text="Preprocess", command=self.start_preprocessing)
        self.preprocess_button.pack(side="top", padx=10, pady=10)
        self.preprocess_button.configure(state="disabled")
        self.chat_analysis = ctk.CTkButton(self.frame, text="Chat Analysis")
        self.chat_analysis.pack(side="top", padx=10, pady=10)
        self.chat_analysis.configure(state="disabled")
        self.sentiment_analysis = ctk.CTkButton(self.frame, text="Sentiment Analysis")
        self.sentiment_analysis.pack(side="top", padx=10, pady=10)
        self.sentiment_analysis.configure(state="disabled")

        self.loading_frame = ctk.CTkFrame(self.frame)
        self.loading_label = ctk.CTkLabel(self.loading_frame, text="")
        self.loading_label.pack(pady=10)
        self.progress = ctk.CTkProgressBar(self.loading_frame, orientation="horizontal", mode="determinate")
        self.progress.pack(pady=10)
        self.progress.set(0)

    def load_chats(self):
        file_path = filedialog.askopenfilename(parent=self, filetypes=[("Text files", "*.txt")])
        if file_path:
            file_name = file_path.split("/")[-1]
            self.select_name.configure(text=file_name)
            self.file_path = file_path
            self.preprocess_button.configure(state="normal")
        else:
            self.select_name.configure(text="")
            self.preprocess_button.configure(state="disabled")
            self.chat_analysis.configure(state="disabled")
            self.sentiment_analysis.configure(state="disabled")
            self.loading_frame.pack_forget()


    def start_preprocessing(self):
        self.preprocess_button.configure(state="disabled")
        self.loading_frame.pack(side="bottom", pady=(20, 0), fill="x")
        self.loading_label.configure(text="Loading...", font=ctk.CTkFont(size=15, weight="bold"))
        threading.Thread(target=self.preprocess).start()


    def preprocess(self):
        try:
            total_steps = 5
            current_step = 0

            def update_progress(step):
                self.progress.set(step / total_steps)

            with open(self.file_path, 'r', encoding='utf8') as file:
                self.data = file.read()
            current_step += 1
            update_progress(current_step)

            # pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
            pattern = '\[?\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?(?:\s(?:AM|PM|am|pm))?\]?(?:\s-\s)?'
            messages = re.split(pattern, self.data)[1:]
            messages = [message for message in messages if message != None]
            
            dates = re.findall(pattern, self.data)
            dates = [date.strip('[ ] - ') for date in dates]
            current_step += 1
            update_progress(current_step)
            
            self.df = pd.DataFrame({'user_message': messages, 'message_date': dates})
            
            # convert message_date type
            self.df['message_date'] = pd.to_datetime(self.df['message_date'], format='mixed', dayfirst=True)

            self.df.rename(columns={'message_date': 'date'}, inplace=True)
            current_step += 1
            update_progress(current_step)
            
            self.df[['user', 'message']] = self.df['user_message'].apply(lambda x: pd.Series(re.split('([\\w\\W]+?):\\s', x)[1:3]
                                                                    if re.split('([\\w\\W]+?):\\s', x)[1:]
                                                                    else ['group_notification', re.split('([\\w\\W]+?):\\s', x)[0]]))
            self.df.drop(columns=['user_message'], inplace=True)
            current_step += 1
            update_progress(current_step)
            
            self.df['only_date'] = self.df['date'].dt.date
            self.df['year'] = self.df['date'].dt.year
            self.df['month_num'] = self.df['date'].dt.month
            self.df['month'] = self.df['date'].dt.month_name()
            self.df['day'] = self.df['date'].dt.day
            self.df['day_name'] = self.df['date'].dt.day_name()
            self.df['hour'] = self.df['date'].dt.hour
            self.df['minute'] = self.df['date'].dt.minute
            self.df['second'] = self.df['date'].dt.second
            self.df['period'] = self.df['hour'].apply(lambda x: f"{x:02d}-{(x + 1) % 24:02d}")
            self.df.drop(columns=['date'], inplace=True)
            self.df.rename(columns={'only_date': 'date'}, inplace=True)
            current_step += 1
            update_progress(current_step)

            self.loading_label.configure(text="Preprocessing completed", font=ctk.CTkFont(size=15, weight="bold"))
            messagebox.showinfo("Success", "Preprocessing completed successfully!")
            self.chat_analysis.configure(state="normal")
            self.chat_analysis.configure(command=lambda: chat_analysis.ChatAnalyzer(self))
            self.sentiment_analysis.configure(state="normal")
            self.sentiment_analysis.configure(command=lambda: sentiment_analysis.SentimentAnalyzer(self))

        except Exception:
            messagebox.showerror("Error", "Invalid Whatsapp Chat file")
            self.loading_frame.pack_forget()



if __name__ == "__main__":
    app = ChatSentimentAnalyzer()

    ctk.set_appearance_mode("System")

    app.mainloop()
