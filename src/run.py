import praw
import pandas as pd
import datetime as dt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
# from nltk.tokenize import RegexpTokenizer
import string
import config as cg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
from pprint import pprint
import collections

topics_dict = {"title": [],
               "vote_score": [],
               "id": [],
               "comments_number": [],
               "url": [],
               "created": [],
               "subreddit": [],
               'subreddit_created': [],
               'subreddit_subscribers_number': [],
               'subreddit_nsfw': []
               }

contractions = {"can't": "cannot",
                "I'm": "I am",
                "I've": "I have",
                "isn't": "is not",
                }
contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))

stop_words = set(stopwords.words("english"))
# ps = PorterStemmer()
lem = WordNetLemmatizer()
# vectorizer = TfidfVectorizer()



def gen_date(created):
    return dt.datetime.fromtimestamp(created)


def subreddit_list():
    pass


class RedditData():
    def connect_reddit_call(self):
        self.reddit = praw.Reddit(client_id=cg.client_id,
                             client_secret=cg.client_secret,
                             user_agent=cg.user_agent,
                             username=cg.username,
                             password=cg.password)
        return self.reddit

    def get_reddit_data(self, subreddit, extraction_limit, topics):
        subreddits = self.reddit.subreddit(subreddit)
        top_subreddit = subreddits.top(limit=extraction_limit)  # top
        for submission in top_subreddit:
            topics["title"].append(submission.title)
            topics["vote_score"].append(submission.score)
            topics["id"].append(submission.id)
            topics["url"].append(submission.url)
            topics["comments_number"].append(submission.num_comments)
            topics["created"].append(submission.created)
            topics["subreddit"].append(submission.subreddit)
            topics['subreddit_created'].append(submission.subreddit.created_utc)
            topics['subreddit_subscribers_number'].append(submission.subreddit.subscribers)
            topics['subreddit_nsfw'].append(submission.subreddit.over18)
        return topics


def expand_contractions(title, contractions_dict=contractions):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, title)


def fill_csv(topics, output_path):
    topics_data = pd.DataFrame.from_dict(topics)
    _timestamp = topics_data["created"].apply(gen_date)
    topics_data = topics_data.assign(timestamp=_timestamp)
    topics_data.to_csv(output_path)


def remove_stopwords(title):
    filtered = []
    for word in title:
        if word not in stop_words:
            filtered.append(word)
    return filtered


def reduce_to_lemmatization(title):
    lemmatization_words = []
    for word in title:
        lemmatization_words.append(lem.lemmatize(word))
    return lemmatization_words


def tokenize_punctuation(title):
    return title.translate(str.maketrans('', '', string.punctuation))


def subject_extractor(title):
    noun_list = []
    for word in title:
        if 'VB' in word[1]:
            noun_list.append(word[0])
            # print (word[0])
    return noun_list


def cluster_texts(texts, clusters=3):
    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
    return clustering


def process_text(title_df):
    # Individual Data Node
    # Convert 's and 'm to full length meanings.
    title_df['title'] = title_df['title'].apply(expand_contractions)
    # Removing punctuation from the sentence
    title_df['title'] = title_df['title'].apply(tokenize_punctuation)
    # Breaking into Words
    title_df['title'] = title_df['title'].apply(word_tokenize)
    # Removing Stop Words
    title_df['title'] = title_df['title'].apply(remove_stopwords)
    # Lemmatizing to Base form.
    title_df['title'] = title_df['title'].apply(reduce_to_lemmatization)
    return title_df


def text_analysis(input_file):
    columns = ['timestamp', 'title','comments_number', 'vote_score']
    title_df = pd.read_csv(input_file, usecols=columns)
    title_df['title_bkp'] = title_df['title']
    # title_df = process_text(title_df)
    # Combined Analysis
    title_combined = ' '.join(str(single_title) for single_title in title_df['title'].tolist())
    title_combined = word_tokenize(title_combined)
    # print (title_combined)
    clusters = cluster_texts(title_combined, 7)
    pprint(dict(clusters))
    # fdist = FreqDist(title_combined)
        # Entire Data Summary
    # for i, feature in enumerate(vectorizer.get_feature_names()):
    #     print(i, feature)
    # fdist.plot(30, cumulative=False)
    # plt.show()
    # print(fdist.most_common(25))
    # title_df['title'] = title_df['title'].apply(pos_tag)
    # title_df['main_subject'] = title_df['title'].apply(subject_extractor)
    # print(title_df['title'])
    title_df.to_csv(cg.analysed_output)


if __name__ == '__main__':
    pass
    # reddit = RedditData()
    # reddit.connect_reddit_call()
    # topics_dict = reddit.get_reddit_data(cg.subreddit, cg.extraction_limit, topics_dict)
    # fill_csv(topics_dict, cg.output)
    text_analysis(cg.output)