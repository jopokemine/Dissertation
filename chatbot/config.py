import os

# MAX_LENGTH = 15
DATA_DIR = os.path.join("chatbot", "data.nosync")
# DATA_DIR = "data.nosync"
SAVE_DIR = os.path.join(DATA_DIR, "save")
data = {
    'amazon': os.path.join(DATA_DIR, 'amazon_qa'),
    'convai': os.path.join(DATA_DIR, 'convai_dataset'),
    'cornell': os.path.join(DATA_DIR, 'cornell movie-dialogs corpus'),
    'opensubtitles': os.path.join(DATA_DIR, 'opensubtitles'),
    'qa': os.path.join(DATA_DIR, 'Question_Answer_Dataset_v1.2'),
    'rsics': os.path.join(DATA_DIR, 'rsics_dataset'),
    'reddit': os.path.join(DATA_DIR, 'reddit_full_data'),
    'twitter': os.path.join(DATA_DIR, 'twitter_customer_support/twcs'),
    'ubuntu': os.path.join(DATA_DIR, 'ubuntu_dialogue_corpus/Ubuntu-dialogue-corpus')
}

################################
# Model Config                 #
################################
MODEL_NAME = 'cb_model'
ATTN_MODEL = 'dot'
# ATTN_MODEL = 'general'
# ATTN_MODEL = 'concat'
HIDDEN_SIZE = 1000
ENCODER_N_LAYERS = 2
DECODER_N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 64

################################
# Training Config              #
################################
CLIP = 50.0
TEACHER_FORCING_RATIO = 1.0
LEARNING_RATE = 0.0001
DECODER_LEARNING_RATIO = 5.0
N_ITERATION = 1000
PRINT_EVERY = 1
SAVE_EVERY = 500
