#!/usr/bin/env python

import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from utils_hv import create_input
import loader

from utils_hv import get_name, models_path, evaluate, test, eval_script#, eval_temp
from loader import word_mapping, char_mapping, tag_mapping, pos_mapping, dep_verb_mapping, dep_noun_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from model_hv import Model
import pdb
import cPickle
from os.path import basename
# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-o", "--outdir", default="./evaluation/results/res",
    help="Output directory of results"
)
optparser.add_option(
        "-d", "--dev", default="",
        help="Dev set location"
    )
optparser.add_option(
    "-m", "--models", default="",
    help="model location"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)

opts = optparser.parse_args()[0]
fn = opts.outdir
gcn_df = "../gcn_data/"

# Parse parameters

# Check parameters validity
assert os.path.isfile(opts.test)

# Check evaluation script / folders
eval_temp = basename(opts.outdir)
'''
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
'''
model = Model(load_path=opts.models)
parameters = model.parameters

if opts.word_bidirect != 1:
    parameters['word_bidirect'] = opts.word_bidirect

# word_to_id, char_to_id, tag_to_id, POS_to_id, N_to_id, V_to_id = [
#     {v: k for k, v in x.items()}
#     for x in [model.id_to_word, model.id_to_char, model.id_to_tag, model.id_to_POS, model.id_to_N, model.id_to_V]
# ]
word_to_id, char_to_id, tag_to_id, POS_to_id= [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag, model.id_to_POS]
]
V_to_id={'<UNK>':0}
N_to_id={'<UNK>':0}
id_to_tag = model.id_to_tag

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
test_sentences = loader.load_sentences(opts.test, lower, zeros)

# Use selected tagging scheme (IOB / IOBES)

test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, POS_to_id, lower)

# print "%i sentences in test." % (len(test_data))

# Save the mappings to disk

# Build the model
_, f_eval, f_eval_softmax, f_eval_hv = model.build(**parameters)

# Reload previous model values
# print 'Reloading previous model...'
model.reload()
#
# Train network
#
dico_tags = []


dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, POS_to_id, lower
    )
dx = gcn_df+'dx'
dy = gcn_df+'dy'
test(parameters, f_eval_hv, dev_sentences,
                      dev_data, tag_to_id, dx, dy)

train_sentences = loader.load_sentences(opts.train, lower, zeros)
train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, POS_to_id, lower
    )
x = gcn_df+'x'
y = gcn_df+'y'
test(parameters, f_eval_hv, train_sentences,
                      train_data, tag_to_id, x, y)
    
tx = gcn_df+'tx'
ty = gcn_df+'ty'
test(parameters, f_eval_hv, test_sentences,
                      test_data, tag_to_id, tx, ty)
