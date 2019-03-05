
import os
import cPickle
import loader
import codecs
from sklearn.metrics import classification_report

def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def reload_tag_mappings(fn):
    with open(fn,'rb') as f:
        mappings = cPickle.load(f)
    return mappings['id_to_tag']

f_mapping = './models/zeros=True,char_dim=25,char_lstm_dim=25,word_dim=200,word_lstm_dim=100,all_emb=False,cap_dim=4,dropout=0.5,lr_method=sgd-lr_.005,pos_dim=10,dep_dim=0,train=train,train_true=,reload=None,201801241144792/mappings.pkl'

id_to_tag = reload_tag_mappings(f_mapping)
print id_to_tag

# Data parameters
lower = 0
zeros = 0
tag_scheme = 'iobes'

# Load sentences
f_test = '../data/test'
test_sentences = loader.load_sentences(f_test, lower, zeros)

f_pred = '../GCN/gcn/test_predictions'
predictions = []
r_tag_all = []
p_tag_all = []
with open(f_pred, 'rb') as f:
    for sent in test_sentences:
        p_tags = []
        r_tags = []
        for word in sent:
            p_tag_id = f.readline().strip()
            p_tag = id_to_tag[int(p_tag_id)]
            p_tags.append(p_tag)
            r_tags.append(word[1])
        if tag_scheme == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        r_tag_all += [tag.split('-')[1] if '-' in tag else tag for tag in r_tags]
        p_tag_all += [tag.split('-')[1] if '-' in tag else tag for tag in p_tags]
        for word,tag in zip(sent, p_tags):
            newline = " ".join([word[0]] + [word[1]] + [tag])
            predictions.append(newline)
        predictions.append("")

eval_temp = './evaluation'
eval_script = os.path.join(eval_temp, "conlleval")
output_path = os.path.join(eval_temp, 'output') 
scores_path = os.path.join(eval_temp, 'scores')

with codecs.open(output_path, 'w', 'utf8') as f:
    f.write("\n".join(predictions))

os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

targets = ['O','Task','Process','Material']
report = classification_report(r_tag_all, p_tag_all, targets)
with open(scores_path, 'a') as f:
    f.write('Catergory results\n')
    f.write(report + '\n')

# CoNLL evaluation results
eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
for line in eval_lines:
    print line

# F1 on all entities
print float(eval_lines[1].strip().split()[-1])

            
