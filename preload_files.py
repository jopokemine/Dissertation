import os
import gzip
import csv
import codecs
import json
import re
from datetime import datetime
# from config import data


############################################
# Generic                                  #
############################################


def load_files(*filepaths, open_func=open, line_eval_func=None):
    for file in filepaths:
        print(f"    Loading {file.split('/')[-1]}...")
        with open_func(file) as f:
            for line in f:
                yield line if line_eval_func is None else line_eval_func(line)


def load_csv_files(*filepaths, delimiter=','):
    for file in filepaths:
        print(f"    Loading {file.split('/')[-1]}...")
        with open(file, mode="rb") as f:
            lines = []
            for line in f:
                try:
                    line = line.decode("utf-8")
                except UnicodeDecodeError:
                    continue  # Ignore any lines with non-decodable strings in
                lines.append(line)

            csv_reader = csv.DictReader(lines, delimiter=delimiter)
            for row in csv_reader:
                yield row


def write_pairs_from_iter_to_file(datafile: str, pair_iter):
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    with open(datafile, 'a', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        while True:
            try:
                pair = next(pair_iter)
            except StopIteration:
                break
            try:
                writer.writerow(pair)
            except UnicodeEncodeError:
                continue  # Ignore any lines with non-encodable strings in


def write_pairs_from_list_to_file(datafile: str, pairs: list):
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    with open(datafile, 'a', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in pairs:
            writer.writerow(pair)


def empty_formatted_lines_file(path):
    print(f"Emptying {path}")
    open(path, 'w').close()


# get_full_path = lambda files, path: [os.path.join(path, file) for file in files]


############################################
# Amazon QA Dataset                        #
############################################


def load_amazon_dataset(path_to_datafiles):
    print("Loading Amazon dataset...")
    _, _, filenames = next(os.walk(path_to_datafiles))
    filepaths = [os.path.join(path_to_datafiles, f) for f in filenames]
    multiple_answers = filter(lambda f: 'multiple' in f, filepaths)
    single_answers = filter(lambda f: 'multiple' not in f, filepaths)
    ma_lines = load_files(*multiple_answers, open_func=gzip.open, line_eval_func=eval)
    sa_lines = load_files(*single_answers, open_func=gzip.open, line_eval_func=eval)
    ma_pairs = format_multiple_answer_amazon_data(ma_lines)
    sa_pairs = format_single_answer_amazon_data(sa_lines)
    return ma_pairs, sa_pairs


def format_single_answer_amazon_data(line_it):
    while True:
        try:
            obj = next(line_it)
        except StopIteration:
            break
        yield [obj['question'], obj['answer']]


def format_multiple_answer_amazon_data(line_it):
    while True:
        try:
            obj = next(line_it)
        except StopIteration:
            break
        for question in obj['questions']:
            for answer in question['answers']:
                yield [question['questionText'], answer['answerText']]


############################################
# Convai Dataset                           #
############################################

def load_convai_dataset(path_to_datafiles):
    # TODO: Finish
    print("Loading Convai dataset...")
    _, _, filenames = next(os.walk(path_to_datafiles))
    filepaths = [os.path.join(path_to_datafiles, f) for f in filenames]
    datafiles = filter(lambda f: 'data' in f, filepaths)
    lines = load_files(*datafiles, line_eval_func=json.load)
    line = next(lines)
    for i in range(500):
        for j in range(len(line[i]['dialog'])):
            print(f"CONVO {i}: {line[i]['dialog'][j]['sender']}: {line[i]['dialog'][j]['text']}")
    # print(line[1])
    # print(line[0]['dialog'][1])
    # print(line[0]['dialog'][2])
    # print(line[0]['dialog'][3])
    # print(line[0]['dialog'][4])
    # print(line[0]['dialog'][5])
    # print(line[0]['dialog'][6])
    # print(next(lines)[0]['dialog'][2])
    # print(line[:20])
    # print(eval(next(lines)[1:-2]))


############################################
# Squad Train Dataset                      #
############################################


def load_squad_train_dataset(path_to_datafiles):
    # FIXME
    print("Loading Squad Train dataset")
    _, _, filenames = next(os.walk(path_to_datafiles))
    objs = load_files(*[os.path.join(path_to_datafiles, f) for f in filenames])
    print(next(objs))


############################################
# Opensubtitles Dataset                    #
############################################


def load_opensubtitles_dataset(path_to_datafiles):
    # TODO
    print("Loading Opensubtitles dataset...")
    _, _, filenames = next(os.walk(path_to_datafiles))
    filepaths = [os.path.join(path_to_datafiles, f) for f in filenames]
    datafiles = filter(lambda f: '.gz' not in f, filepaths)
    lines = load_files(path_to_datafiles, *datafiles)
    i = 0
    while i < 50:
        try:
            line = next(lines)
        except StopIteration:
            break
        print(line)
        i += 1


############################################
# Cornell Dataset                          #
############################################


def load_cornell_dataset(path_to_datafiles):
    print("Loading Cornell dataset...")
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
    lines = loadLines(os.path.join(path_to_datafiles, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    conversations = loadConversations(os.path.join(path_to_datafiles, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)
    pairs = extractSentencePairs(conversations)
    return pairs


# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i + 1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


############################################
# Cornell Dataset                          #
############################################


def load_QA_dataset(path_to_datafiles):
    print("Loading QA dataset...")
    _, dirs, _ = next(os.walk(path_to_datafiles))
    datafiles = []
    for d in dirs:
        _, _, filenames = next(os.walk(os.path.join(path_to_datafiles, d)))
        datafiles.extend([os.path.join(path_to_datafiles, d, f) for f in filenames])
    lines = load_csv_files(*datafiles, delimiter="\t")
    while True:
        try:
            line = next(lines)
        except StopIteration:
            break
        if line['Question'] != "NULL" and line['Answer'] != "NULL":
            yield [line['Question'], line['Answer']]


############################################
# Twitter Customer Support Dataset         #
############################################


def load_twitter_dataset(path_to_datafiles):
    print("Loading Twitter Customer Support dataset...")
    _, _, filenames = next(os.walk(path_to_datafiles))
    datafiles = [os.path.join(path_to_datafiles, file) for file in filenames]
    lines = load_csv_files(*datafiles)
    responses = {}
    in_response_to = {}
    while True:
        try:
            line = next(lines)
        except StopIteration:
            break

        words = line['text'].split(' ')
        line['text'] = ' '.join(words[1:]) if words[0][0] == '@' else line['text']

        if line['in_response_to_tweet_id'] in responses.keys():
            orig_tweet = responses[line['in_response_to_tweet_id']]['text']
            del responses[line['in_response_to_tweet_id']]
            yield [orig_tweet, line['text']]
        else:
            responses[line['response_tweet_id']] = line

        if line['response_tweet_id'] in in_response_to.keys():
            tweet = in_response_to[line['response_tweet_id']]['text']
            del in_response_to[line['response_tweet_id']]
            yield [line['text'], tweet]
        else:
            responses[line['in_response_to_tweet_id']] = line


############################################
# Testing Section                          #
############################################


def load_reddit_dataset(path_to_datafiles):
    print("Loading Reddit dataset...")
    _, _, filenames = next(os.walk(path_to_datafiles))
    files = [os.path.join(path_to_datafiles, f) for f in filenames]
    datafiles = filter(lambda f: '.gz' in f, files)
    lines = load_files(*datafiles, open_func=gzip.open, line_eval_func=json.loads)
    for _ in range(5):
        try:
            line = next(lines)
        except StopIteration:
            None
        print(line.keys(), end="\n\n\n")


def load_reddit_txt(path_to_datafiles):
    print("Loading Reddit dataset...")
    datafile = os.path.join(path_to_datafiles, "RS_2011-01")
    lines = load_files(datafile, line_eval_func=json.loads)
    for _ in range(5):
        try:
            line = next(lines)
        except StopIteration:
            break
        print(line.keys(), end="\n\n\n")


############################################
# Testing Section                          #
############################################


now = datetime.now()
timestamp = now.strftime("%d%m%y-%H%M%S")
unique_file = f"data/formatted_lines-{timestamp}.txt"

# empty_formatted_lines_file(unique_file)
# ma_pairs, sa_pairs = load_amazon_dataset("data/amazon_qa")
# write_pairs_from_iter_to_file(unique_file, ma_pairs)
# write_pairs_from_iter_to_file(unique_file, sa_pairs)
# # load_convai_dataset("data/convai_dataset")
# # # load_squad_train_dataset("data/squad_train_dataset")
# # # load_opensubtitles_dataset("data/opensubtitles")
# cornell_pairs = load_cornell_dataset("data/cornell movie-dialogs corpus")
# write_pairs_from_list_to_file(unique_file, cornell_pairs)
# QA_pairs = load_QA_dataset("data/Question_Answer_Dataset_v1.2")
# write_pairs_from_iter_to_file(unique_file, QA_pairs)
# twitter_pairs = load_twitter_dataset("data/twitter_customer_support/twcs")
# write_pairs_from_iter_to_file(unique_file, twitter_pairs)

load_reddit_dataset("data/reddit_full_data")
load_reddit_txt("data")

# print(f"Loaded! Lines written to {unique_file}")
