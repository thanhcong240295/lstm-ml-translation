"""
Constants used throughout the machine translation project.

This module defines all special tokens and their indices to ensure
consistency across the codebase.
"""

# Special token strings
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = {PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN}
