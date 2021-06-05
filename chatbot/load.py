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
    """Loads in dataset files, given filepaths, and optional open and evaluation functions.

    Args:
        filepaths (str): relative filepaths to the datafiles to be loaded.
        open_func (func, optional): Function to open the file, should 'open' not be sufficient. Defaults to open.
        line_eval_func (func, optional): Function to further process the data loaded before it is yielded. Defaults to None.

    Yields:
        iterator: Iterator of the line loaded.
    """
    # open_func allows for different open funcitons, in case the built-in open() funciton is not enough
    # line_eval_func is optional, and allows some evaluation before return. Mostly used for JSON files, where json.loads() is needed
    for file in filepaths:
        print(f"    Loading {file.split('/')[-1]}...")
        with open_func(file) as f:
            for line in f:
                yield line if line_eval_func is None else line_eval_func(line)


def load_csv_files(*filepaths, delimiter=','):
    """Loads in csv files, given filepaths and an optional delimiter.

    Args:
        filepaths (str): relative filepaths to the datafiles to be loaded.
        delimiter (str, optional): Delimiter to use to load csv file. Defaults to ','.

    Yields:
        iterator: Iterator containing the lines
    """
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


def load_tsv_files(*filepaths, delimiter=','):
    """Simpler function for loading csv files, without as much protection against Unicode characters. Used for loading 'formatted_lines_*' files only.

    Args:
        delimiter (str, optional): Delimiter to be used to load csv files. Defaults to ','.

    Yields:
        iterator: Iterator containing the lines
    """
    for file in filepaths:
        with open(file) as f:
            read_csv = csv.reader(f, delimiter=delimiter)
            for line in read_csv:
                yield line


def write_pairs(datafile):
    """Decorator function that will take the data returned from a function, and write it to the specified file.

    Args:
        datafile (str): Relative path to the file to be written.
    """
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
                        pair = [s.strip().replace('\n', '').replace('\t', '') for s in pair]
                        writer.writerow(pair)
                    except UnicodeEncodeError:
                        continue  # Ignore Unicode characters
                    except StopIteration:
                        break  # Have reached end of iterator, stop.
        return wrapper
    return decorator


@write_pairs(os.path.join(DATA_DIR, "formatted_lines_combined.txt"))
def combine_datasets(*datafiles):
    """Simple function to combine two csv files together. Used to combine the output from multiple datasets into one file.

    Returns:
        iterator: Iterator of data to be written by write_pairs decorator.
    """
    print(f"Combining {', '.join([file.split('/')[-1] for file in datafiles])}...")
    return load_tsv_files(*datafiles, delimiter='\t')


def empty_file(filepath):
    """Empties the specified file, without deleting it.

    Args:
        filepath (str): Relative path to file to be emptied.
    """
    if os.path.exists(filepath):
        print(f"Emptying {filepath}")
        open(filepath, 'w').close()


############################################
# Amazon QA Dataset                        #
############################################

@write_pairs(os.path.join(DATA_DIR, "formatted_lines_amazon.txt"))
def load_amazon_dataset():
    """Function to load Amazon_QA dataset. Credit for the dataset:
    Henderson, M., Budzianowski, P., Casanueva, I., Coope, S., Gerz, D., Kumar,G., Mrkši ́c, N., Spithourakis, G., Su, P.-H., Vulic, I., & Wen, T.-H. (2019). A repository of conversational datasets [Data available at github.com/PolyAI-LDN/conversational-datasets]. Proceedings of the Workshop on NLP for Conversational AI. https://arxiv.org/abs/1904.06472. License: Apache License, Version 2.0.

    Returns:
        iterator: iterator of data to be written by write_pairs decorator.
    """
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
    """Function to format a single answer dictionary into a usable sentence pair.

    Args:
        line_it (iterator): Iterator containing dictionary objects of question and answer.

    Yields:
        iterator: Iterator of lists of sentence pairs.
    """
    while True:
        try:
            obj = next(line_it)
        except StopIteration:
            break
        yield [obj['question'], obj['answer']]


def format_multiple_answer_amazon_data(line_it):
    """Function to format a multiple answer dictionary into usable sentence pairs.

    Args:
        line_it (iterator): Iterator containing dictionary objects of questions and answers.
    Yields:
        iterator: Iterator of lists of sentence pairs.
    """
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

@write_pairs(os.path.join(DATA_DIR, "formatted_lines_convai.txt"))
def load_convai_dataset():
    """Function to load Convai dataset. Credit for the dataset:
    Aliannejadi, M., Kiseleva, J., Chuklin, A., Dalton, J., & Burtsev, M. (2020). Con-vAI3: Generating Clarifying Questions for Open-Domain Dialogue Systems (ClariQ). https://arxiv.org/abs/2009.11352.

    Yields:
        iterator: iterator of data to be written by write_pairs decorator.
    """
    print("Loading Convai dataset...")
    _, _, filenames = next(os.walk(data['convai']))
    filepaths = [os.path.join(data['convai'], f) for f in filenames]
    datafiles = filter(lambda f: 'data_' in f, filepaths)
    lines = load_files(*datafiles, line_eval_func=json.loads)
    while True:
        try:
            line = next(lines)
        except StopIteration:
            break
        previous = []
        for i in range(len(line)):
            previous = []  # Empty previous answers, to stop conflicts
            for j in range(len(line[i]['dialog'])):
                if previous != []:
                    if previous[0] == line[i]['dialog'][j]['sender']:
                        previous.append(f"{previous[1]} {line[i]['dialog'][j]['text']}")
                    else:
                        for msg in previous[1:-1]:
                            yield [msg, line[i]['dialog'][j]['text']]
                        previous = []
                else:
                    previous = [line[i]['dialog'][j]['sender'], line[i]['dialog'][j]['text']]


############################################
# Squad Train Dataset                      #
############################################

@write_pairs(os.path.join(DATA_DIR, "formatted_lines_squad.txt"))
def load_squad_train_dataset():
    """Function to load SQuAD dataset. Credit for the dataset:
    Rajpurkar, P., Jia, R., & Liang, P. (2018). Know What You Don’t Know: Unanswerable Questions for SQuAD. CoRR,abs/1806.03822. https://arxiv.org/abs/1806.03822.

    Yields:
        iterator: iterator of data to be written by write_pairs decorator.
    """
    print("Loading Squad Train dataset")
    _, _, filenames = next(os.walk(data['squad']))
    objs = load_files(*[os.path.join(data['squad'], f) for f in filenames], line_eval_func=json.loads)
    obj = next(objs)  # only one line, so only need to call this once.
    for dataobj in obj['data']:
        for paragraph in dataobj['paragraphs']:
            for qa in paragraph['qas']:
                for ans in qa['answers']:
                    yield [qa['question'], ans['text']]


############################################
# Opensubtitles Dataset                    #
############################################

@write_pairs(os.path.join(DATA_DIR, "formatted_lines_opensubtitles.txt"))
def load_opensubtitles_dataset():
    """Function to load OpenSubtitles dataset. Credit for the dataset:
    Henderson, M., Budzianowski, P., Casanueva, I., Coope, S., Gerz, D., Kumar,G., Mrkši ́c, N., Spithourakis, G., Su, P.-H., Vulic, I., & Wen, T.-H. (2019). A repository of conversational datasets [Data available at github.com/PolyAI-LDN/conversational-datasets]. Proceedings of the Workshop on NLP for Conversational AI. https://arxiv.org/abs/1904.06472. License: Apache License, Version 2.0.

    Yields:
        iterator: iterator of data to be written by write_pairs decorator.
    """
    print("Loading Opensubtitles dataset...")
    _, _, filenames = next(os.walk(data['opensubtitles']))
    filepaths = [os.path.join(data['opensubtitles'], f) for f in filenames]
    datafiles = filter(lambda f: '.gz' not in f, filepaths)
    lines = load_files(*datafiles)
    while True:
        try:
            line1 = next(lines)
            line2 = next(lines)
        except StopIteration:
            break
        yield [line1, line2]


############################################
# Cornell Dataset                          #
############################################

"""
Credit for the code for this section goes to: Inkawhich, M. (2017).Chatbot Tutorial – PyTorch Tutorials 1.8.1+cu102 doc-umentation. Retrieved December 3, 2020, from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html?highlight=chatbot.
"""


@write_pairs(os.path.join(DATA_DIR, "formatted_lines_cornell.txt"))
def load_cornell_dataset():
    """Function to load Cornell dataset. Credit for dataset:
    Danescu-Niculescu-Mizil, C., & Lee, L. (2011). Chameleons in imagined conversations: A new approach to understanding coordination of linguisticstyle in dialogs.Proceedings of the Workshop on Cognitive Modelingand Computational Linguistics, ACL 2011.

    Returns:
        list: list of lists of sentence pairs.
    """
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
    """Function to load QA dataset. Credit for the dataset:
    Smith, N. A., Heilman, M., & Hwa, R. (2008). Question generation as a competitive undergraduate course project. Proceedings of the NSF Workshopon the Question Generation Shared Task and Evaluation Challenge, 4–6.

    Yields:
        iterator: iterator of data to be written by write_pairs decorator.
    """
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
    """Function to load Twitter Customer Service dataset. Credit for the dataset:
    Axelbrooke, S. (2017).Customer Support on Twitter(Version 10). RetrievedJanuary 5, 2021, from https://www.kaggle.com/thoughtvector/customer-support-on-twitter/version/10. License: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

    Yields:
        iterator: iterator of data to be written by write_pairs decorator.
    """
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

def load_reddit_dataset():
    """UNFINISHED: Function to load Reddit dataset. Credit for the dataset:
    Henderson, M., Budzianowski, P., Casanueva, I., Coope, S., Gerz, D., Kumar,G., Mrkši ́c, N., Spithourakis, G., Su, P.-H., Vulic, I., & Wen, T.-H. (2019). A repository of conversational datasets [Data available at github.com/PolyAI-LDN/conversational-datasets]. Proceedings of the Workshop on NLP for Conversational AI. https://arxiv.org/abs/1904.06472. License: Apache License, Version 2.0.
    """
    print("Loading Reddit dataset...")
    _, _, filenames = next(os.walk(data['reddit']))
    files = [os.path.join(data['reddit'], f) for f in filenames]
    datafiles = filter(lambda f: '.gz' in f, files)
    lines = load_files(*datafiles, open_func=gzip.open, line_eval_func=json.loads)
    for _ in range(10):
        try:
            line = next(lines)
        except StopIteration:
            break
        print(line['title'])
        print('=========================')
        print(line['selftext'])
        print('-------------------------')
        # print(line.keys(), end="\n\n\n")


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
