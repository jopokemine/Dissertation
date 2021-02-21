import os
import gzip
import csv
import codecs
import json
import re
from itertools import chain
from .config import data, DATA_DIR


############################################
# Generic                                  #
############################################

def load_files(*filepaths, open_func=open, line_eval_func=None):
    for file in filepaths:
        print(f"    Loading {file.split('/')[-1]}...")
        with open_func(file) as f:
            for line in f:
                yield line if line_eval_func is None else line_eval_func(line)


def load_tsv_files(*filepaths, delimiter=','):
    for file in filepaths:
        with open(file) as f:
            read_csv = csv.reader(f, delimiter=delimiter)
            for line in read_csv:
                yield line


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


def write_pairs(datafile):
    def decorator(function):
        def wrapper(*args, **kwargs):
            empty_file(datafile)

            pairs = function(*args, **kwargs)
            pair_iter = iter(pairs) if isinstance(pairs, list) else pairs

            delimiter = str(codecs.decode('\t', "unicode_escape"))
            with open(datafile, 'a', encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=delimiter, lineterminator='\n')
                while True:
                    try:
                        pair = next(pair_iter)
                        writer.writerow(pair)
                    except UnicodeEncodeError:
                        continue
                    except StopIteration:
                        break  # Have reached end of iterator, stop.
        return wrapper
    return decorator


@write_pairs(os.path.join(DATA_DIR, "formatted_lines_combined.txt"))
def combine_datasets(*datafiles):
    print(f"Combining {', '.join([file.split('/')[-1] for file in datafiles])}...")
    return load_tsv_files(*datafiles, delimiter='\t')


def empty_file(filepath):
    if os.path.exists(filepath):
        print(f"Emptying {filepath}")
        open(filepath, 'w').close()


############################################
# Amazon QA Dataset                        #
############################################

@write_pairs(os.path.join(DATA_DIR, "formatted_lines_amazon.txt"))
def load_amazon_dataset():
    print("Loading Amazon dataset...")
    _, _, filenames = next(os.walk(data['amazon']))
    filepaths = [os.path.join(data['amazon'], f) for f in filenames]
    multiple_answers = filter(lambda f: 'multiple' in f, filepaths)
    single_answers = filter(lambda f: 'multiple' not in f, filepaths)
    ma_lines = load_files(*multiple_answers, open_func=gzip.open, line_eval_func=eval)
    sa_lines = load_files(*single_answers, open_func=gzip.open, line_eval_func=eval)
    ma_pairs = format_multiple_answer_amazon_data(ma_lines)
    sa_pairs = format_single_answer_amazon_data(sa_lines)
    return chain(ma_pairs, sa_pairs)


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

def load_convai_dataset():
    # TODO: Finish
    print("Loading Convai dataset...")
    _, _, filenames = next(os.walk(data['convai']))
    filepaths = [os.path.join(data['convai'], f) for f in filenames]
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

def load_squad_train_dataset():
    # FIXME
    print("Loading Squad Train dataset")
    _, _, filenames = next(os.walk(data['squad']))
    objs = load_files(*[os.path.join(data['squad'], f) for f in filenames])
    print(next(objs))


############################################
# Opensubtitles Dataset                    #
############################################

# @write_pairs(os.path.join(DATA_DIR, "formatted_lines_opensubtitles.txt"))
def load_opensubtitles_dataset():
    # TODO
    print("Loading Opensubtitles dataset...")
    _, _, filenames = next(os.walk(data['opensubtitles']))
    filepaths = [os.path.join(data['opensubtitles'], f) for f in filenames]
    datafiles = filter(lambda f: '.gz' not in f, filepaths)
    lines = load_files(*datafiles)
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

@write_pairs(os.path.join(DATA_DIR, "formatted_lines_cornell.txt"))
def load_cornell_dataset():
    print("Loading Cornell dataset...")
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
    lines = loadLines(os.path.join(data['cornell'], "movie_lines.txt"), MOVIE_LINES_FIELDS)
    conversations = loadConversations(os.path.join(data['cornell'], "movie_conversations.txt"),
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
# QA Dataset                               #
############################################

@write_pairs(os.path.join(DATA_DIR, "formatted_lines_qa.txt"))
def load_QA_dataset():
    print("Loading QA dataset...")
    _, dirs, _ = next(os.walk(data['qa']))
    datafiles = []
    for d in dirs:
        _, _, filenames = next(os.walk(os.path.join(data['qa'], d)))
        datafiles.extend([os.path.join(data['qa'], d, f) for f in filenames])
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

@write_pairs(os.path.join(DATA_DIR, "formatted_lines_twitter.txt"))
def load_twitter_dataset():
    print("Loading Twitter Customer Support dataset...")
    _, _, filenames = next(os.walk(data['twitter']))
    datafiles = [os.path.join(data['twitter'], file) for file in filenames]
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
# Reddit Dataset                           #
############################################

def load_reddit_dataset(path_to_datafiles):
    print("Loading Reddit dataset...")
    _, _, filenames = next(os.walk(path_to_datafiles))
    files = [os.path.join(path_to_datafiles, f) for f in filenames]
    datafiles = filter(lambda f: '.gz' in f, files)
    lines = load_files(*datafiles, open_func=gzip.open, line_eval_func=json.loads)
    for _ in range(2):
        try:
            line = next(lines)
        except StopIteration:
            None
        print(line.keys(), end="\n\n\n")


def load_reddit_txt(path_to_datafiles):
    print("Loading Reddit dataset...")
    datafile = os.path.join(path_to_datafiles, "RS_2011-01")
    lines = load_files(datafile, line_eval_func=json.loads)
    for _ in range(2):
        try:
            line = next(lines)
        except StopIteration:
            break
        print(line.keys(), end="\n\n\n")


############################################
# Testing Section                          #
############################################


############################################
# Export Functions                         #
############################################

load_funcs = {
    "amazon": load_amazon_dataset,
    "convai": load_convai_dataset,
    "twitter": load_twitter_dataset,
    "squad": load_squad_train_dataset,
    "opensubtitles": load_opensubtitles_dataset,
    "cornell": load_cornell_dataset,
    "qa": load_QA_dataset,
    "reddit": load_reddit_dataset
}
