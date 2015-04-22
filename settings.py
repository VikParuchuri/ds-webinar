import os

BASE_DIR = os.path.dirname(__file__)

DATA_DIR = os.path.join(BASE_DIR, "data")
POS_SENT_DIRS = [os.path.join(DATA_DIR, "aclImdb", "train", "pos"), os.path.join(DATA_DIR, "aclImdb", "test", "pos")]
NEG_SENT_DIRS = [os.path.join(DATA_DIR, "aclImdb", "train", "neg"), os.path.join(DATA_DIR, "aclImdb", "test", "neg")]

AFINN_WORDLIST = os.path.join(BASE_DIR, "data", "afinn", "AFINN-111.txt")