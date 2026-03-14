##Email Spam Detection

This project performs basic analysis of email messages to understand the difference between spam and non-spam emails. The program cleans the email text and visualizes frequently used words using word clouds.

The goal is to demonstrate simple Natural Language Processing (NLP) techniques using Python.

## Dataset

The dataset file **Emails.csv** contains two columns:

* **label** – indicates whether the email is spam or ham (not spam)
* **text** – the actual email message content

## Main Steps

1. Load the email dataset.
2. Visualize the distribution of spam and non-spam emails.
3. Balance the dataset to avoid bias.
4. Clean the text by removing punctuation and common stopwords.
5. Generate word clouds to show the most frequent words in spam and non-spam emails.

## Technologies Used

Python, Pandas, Matplotlib, Seaborn, NLTK, WordCloud.

## Project Structure

Email_Spam_Detection/

email_spam.py
Emails.csv
models/
results/

## Running the Project

Install the required libraries:

pip install numpy pandas matplotlib seaborn nltk wordcloud tensorflow scikit-learn keras

Then run:

python email_spam.py

The program will process the dataset and display visualizations.

