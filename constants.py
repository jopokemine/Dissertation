import os
import torch

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

save_dir = os.path.join("data", "save")

MAX_LENGTH = 10  # Maximum sentence length to consider
# Use CUDA if available. Otherwise, just use your machine's CPU.
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")