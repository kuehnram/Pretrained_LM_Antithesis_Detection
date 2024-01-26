import math
from typing import List
import nltk
import pandas
import spacy
import re
import os.path
from nltk.corpus import stopwords
from PosTags import PosTag
from typing import Dict
import textdistance

__author__ = "Anonymous"
__license__ = "MIT License"
__version__ = "1.0"
__maintainer__ = "Anonymous"
__email__ = "Anonymous"

"""
Second part of antithesis detection algorithm: The parallelism detection

For each entry, all combinations of parallel phrases are extracted. 
A phrase is defined by the occurrence of punctuation marks or the keywords "und"(and) and "als" (as). 
Parallelism is defined by repeating spacy POS-tags.
Parallel phrases need at least 2 words per phrase.
Levenshtein distance, so no perfect parallelism required if more than three words.
If no parallel phrases can be found, (1) quotation marks, (2) stopwords, (3) quotation marks and stopwords, are removed. 
In the end, everything is written to a csv file
"""

stop_words = stopwords.words('german')
parallel_phrases_dict: Dict[str, Dict[str, str]] = {}
parallel_phrases_list = []
nlp = spacy.load("de_dep_news_trf", disable=['parser', 'ner'])


def parallelism_detection_algorithm(post, original_post):
    is_parallel = False
    post = post.lower()
    doc = nlp(post)
    token_list = []
    word_list = []
    for token in doc:
        if token.pos == PosTag.PROPN:  # proper nouns are treated as nouns
            token_list.append(92)  # number for PosTag.Noun
        else:
            token_list.append(token.pos)
        word_list.append(str(token))
    last_pos = 0
    token_separated_list: List[List[int]] = []
    word_separated_list: List[List[str]] = []
    # split the sentence on a puntuation mark/space e.g., . , :, "und", "als"
    # keep a list with the postags of the tokens
    # and keep a list with the words
    for i in range(len(token_list)):
        if doc[i].pos in [PosTag.PUNCT, PosTag.SPACE] or doc[i].text == "und" or doc[i].text == "als":
            token_separated_list.append((token_list[last_pos:i]))
            word_separated_list.append(word_list[last_pos:i])
            last_pos = i + 1

    for i in range(len(token_separated_list)):
        for j in range(i + 1, len(token_separated_list)):
            if i != j and token_separated_list[i] != [] and token_separated_list[j] != []:
                length_of_smaller_list = min(len(token_separated_list[i]), len(token_separated_list[j]))
                if length_of_smaller_list > 3:  # minimum length of 3, otherwise perfect parallelism required
                    levenshtein_threshold = math.ceil(
                        length_of_smaller_list / 100 * 25)  # 75 % of parallelism/same postags
                else:
                    levenshtein_threshold = 0

                if (textdistance.levenshtein.distance(token_separated_list[i],
                                                      token_separated_list[j])) <= levenshtein_threshold:
                    # Now it is parallel; take only phrases that are longer than one word
                    if word_separated_list[i] != "" and word_separated_list[j] != "" and \
                            len(word_separated_list[i]) > 1 and len(word_separated_list[j]) > 1:
                        templist = [original_post, word_separated_list[i], word_separated_list[j]]
                        parallel_phrases_list.append(templist)
                        is_parallel = True
    return is_parallel


def parallelism_data_preparation(data):
    for post in data:
        if post != "nan":
            original_post = post.replace("\n", "")
            is_parallel = parallelism_detection_algorithm(post, original_post)
            if not is_parallel:
                no_quot_post = post.replace("\'", " ")
                # # Remove distracting double quotes
                no_quot_post = no_quot_post.replace("\"", "")
                no_quot_post = no_quot_post.replace("`", "")
                no_quot_post = no_quot_post.replace("´", "")
                no_quot_post = no_quot_post.replace("„", "")
                no_quot_post = no_quot_post.replace("“", "")
                is_parallel = parallelism_detection_algorithm(no_quot_post, original_post)

                if not is_parallel:
                    post_split_to_words = nltk.word_tokenize(original_post)
                    post_wo_stopwords = [word for word in post_split_to_words if word.lower() not in stop_words]
                    filtered_post_no_stop = " ".join(post_wo_stopwords)
                    is_parallel = parallelism_detection_algorithm(filtered_post_no_stop, original_post)

                    if not is_parallel:
                        post_split_to_words = nltk.word_tokenize(no_quot_post)
                        post_wo_stopwords_wo_quot = [word for word in post_split_to_words if
                                                     word.lower() not in stop_words]
                        filtered_post = " ".join(post_wo_stopwords_wo_quot)
                        is_parallel = parallelism_detection_algorithm(filtered_post, original_post)


def main():
    filepath = "./TelegramData/"
    filenames = [
        os.path.join(filepath, "reitschusterde.csv")
        # possibility to add more files here
    ]

    for file in filenames:
        print(f"File that is currently read: {file}")
        df = pandas.read_csv(file)
        data = df.raw_text.astype(str).values.tolist()

        # Cleaning: Remove Emails
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
        # remove urls/links
        data = [re.sub(r'http\S+', '', sent) for sent in data]
        # Remove hashtags
        data = [re.sub(r'#', '', sent) for sent in data]
        # Remove hyphen and replace by space
        # just as in: https://link.springer.com/chapter/10.1007/978-3-030-55187-2_23 and [17] and [18]
        data = [re.sub("-", " ", sent) for sent in data]
        data = [re.sub("\n", " ", sent) for sent in data]

        parallelism_data_preparation(data)

    df = pandas.DataFrame(parallel_phrases_list, columns=["original_post", "phrase1", "phrase2"])
    df.to_csv("output_parallel_phrases.csv", encoding="utf-8")


if __name__ == '__main__':
    main()
