import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch import optim
from .load import load_funcs, combine_datasets
from .config import DATA_DIR, HIDDEN_SIZE, ENCODER_N_LAYERS, DROPOUT, ATTN_MODEL, DECODER_N_LAYERS, MODEL_NAME, LEARNING_RATE, DECODER_LEARNING_RATIO, SAVE_DIR, N_ITERATION, BATCH_SIZE, SAVE_EVERY, PRINT_EVERY, CLIP
from .bot import loadPrepareData, EncoderRNN, LuongAttnDecoderRNN, trainIters, evaluateInput, GreedySearchDecoder, Voc, evaluate, normalizeString


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', action="store_true", default=False, help='Train the model with corpus')
    parser.add_argument('-te', '--test', action="store_true", default=False, help='Test the saved model')
    dataset_arg = parser.add_argument('-d', '--dataset', default='cornell', help='Comma separated list of dataset(s) to use. Options (case-sensitive) are: amazon, cornell, convai, opensubtitles, QA, rsics, reddit, twitter, ubuntu, squad')

    args = parser.parse_args()

    datasets = args.dataset.split(',') if ',' in args.dataset else [args.dataset]
    datasets = [d.lower() for d in datasets]
    if not (set(datasets) <= set(load_funcs.keys())):
        diff = np.setdiff1d(datasets, load_funcs.keys())
        raise argparse.ArgumentError(dataset_arg, f"Invalid dataset(s): {diff}")

    if args.train and args.test:
        parser.error("Cannot have both -tr and -te")

    return args


def _load_datasets(datasets: list) -> None:
    for dataset in datasets:
        load_funcs[dataset]()

    formatted_files = [os.path.join(DATA_DIR, f"formatted_lines_{d}.txt") for d in datasets]
    combine_datasets(*formatted_files)
    for f in formatted_files:
        print(f"Removing {f}")
        os.remove(f)


def _build_encoder_decoder(voc: Voc, device, loadFilename=None):
    # Load model if a loadFilename is provided
    encoder_optimizer_sd = decoder_optimizer_sd = None

    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, HIDDEN_SIZE)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, DROPOUT)
    decoder = LuongAttnDecoderRNN(ATTN_MODEL, embedding, HIDDEN_SIZE, voc.num_words, DECODER_N_LAYERS, DROPOUT)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    return embedding, encoder, decoder, encoder_optimizer_sd, decoder_optimizer_sd


def init_chatbot(datasets: str, load_from_file: bool = False):
    USE_CUDA = torch.cuda.is_available()
    print(f"Device: {'cuda' if USE_CUDA else 'cpu'}")
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # Define path to data file
    datafile = os.path.join(DATA_DIR, "formatted_lines_combined.txt")

    SAVE_DIR = os.path.join(DATA_DIR, "save")
    voc, pairs = loadPrepareData(datafile, SAVE_DIR)

    loadFilename = None
    if load_from_file:
        dataset_dir = os.path.join(SAVE_DIR, MODEL_NAME, datasets,
                                   '{}-{}_{}'.format(ENCODER_N_LAYERS, DECODER_N_LAYERS, HIDDEN_SIZE))
        _, _, filenames = next(os.walk(dataset_dir))
        checkpoint = 0
        for f in filenames:
            checkpoint_num = f.split('_')[0]
            checkpoint_int = int(checkpoint_num)
            checkpoint = checkpoint_int if checkpoint_int > checkpoint else checkpoint
        loadFilename = os.path.join(dataset_dir, '{}_checkpoint.tar'.format(checkpoint))

    embedding, encoder, decoder, encoder_optimizer_sd, decoder_optimizer_sd = _build_encoder_decoder(voc, device, loadFilename)

    # Ensure DROPOUT layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE * DECODER_LEARNING_RATIO)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    return voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, loadFilename


def evaluate_sentence(encoder: EncoderRNN, decoder: LuongAttnDecoderRNN,
                      voc: Voc, sentence: str):
    encoder.eval()
    decoder.eval()
    searcher = GreedySearchDecoder(encoder, decoder)
    input_sentence = normalizeString(sentence)
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    return ' '.join(output_words)


def run(args: dict) -> None:
    datasets = args.dataset.split(',') if ',' in args.dataset else [args.dataset]
    datasets = [d.lower() for d in datasets]

    _load_datasets(datasets)
    datasets_str = '-'.join(datasets)

    if args.train or args.test:
        voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, loadFilename = init_chatbot(datasets_str, args.test)

    # Run training iterations
    if args.train:
        print("Starting Training!")
        trainIters(MODEL_NAME, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   embedding, ENCODER_N_LAYERS, DECODER_N_LAYERS, SAVE_DIR, N_ITERATION, BATCH_SIZE,
                   PRINT_EVERY, SAVE_EVERY, CLIP, datasets_str, loadFilename)

    ############################################
    # RUN EVALUATION ###########################
    ############################################

    if args.test:
        # Set DROPOUT layers to eval mode
        encoder.eval()
        decoder.eval()

        # Initialize search module
        searcher = GreedySearchDecoder(encoder, decoder)

        # Begin chatting (uncomment and run the following line to begin)
        evaluateInput(encoder, decoder, searcher, voc)


if __name__ == '__main__':
    args = parse()
    run(args)
