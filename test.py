import json
import logging
import os

import tensorflow as tf

import utils
from model import Model
from utils import read_data

from flags import parse_args
FLAGS, unparsed = parse_args()

vocabulary = read_data(FLAGS.text)
print('Data size', len(vocabulary))

with open('./dictionary.json', encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8')

with open('./reverse_dictionary.json', encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf, encoding='utf-8')

for dl in utils.get_train_data(vocabulary, dictionary, batch_size=128, num_steps=32):
    #print(x)
    print('=-=-=-=-=-x=-=-=-=-=-=-', x)
    print('=-=-=-=-=-y=-=-=-=-=-=-', y)
