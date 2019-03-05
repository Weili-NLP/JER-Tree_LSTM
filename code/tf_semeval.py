
import tf_data_utils as utils
import os

import sys
import numpy as np
import tensorflow as tf
import random
import pickle

import tf_tree_lstm

DIR = './data'
GLOVE_DIR ='./'

import pdb
import time

tagnames = ['Other', 'Process', 'Task', 'Material']
num_to_tag = dict(enumerate(tagnames))
tag_to_num = {v:k for k,v in num_to_tag.items()}
n_classes = 4

class Config(object):
    num_emb=None
    emb_dim = 200
    hidden_dim = 150
    output_dim=4
    degree = 2

    num_epochs = 2
    early_stopping = 50
    dropout = 0.5
    lr = 0.05
    emb_lr = 0.1
    reg=0.001
    alpha=0.5 #balace factor in the obj function between eneity recognition and rel extraction

    batch_size = 16
    maxseqlen = None
    maxnodesize = None
    maxrelsize = None
    maxnonrelTsize = None
    maxnonrelPsize = None
    maxnonrelMsize = None
    trainable_embeddings=True

def print_confusion(confusion, num_to_tag):
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print()
    print(confusion)
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print('Tag: (%s) - P {%2.4f} / R {%2.4f}'%(tag, prec, recall))

def calculate_confusion(predicted_indices, y_indices):
    confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
    for i in range(len(y_indices)):
	if y_indices[i]!=-1:
            correct_label = y_indices[i]
            guessed_label = predicted_indices[i]
            confusion[correct_label, guessed_label] += 1
    return confusion

def train():
    config=Config()
    data,vocab = utils.load_tree_data(DIR)

    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print 'trainset: ', len(train_set)
    print 'devset: ', len(dev_set)
    print 'testset: ', len(test_set)

    num_emb = len(vocab)
    num_labels = 4
    '''
    for _, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set(xrange(num_labels)), set(labels)
    '''
    print 'num emb', num_emb
    print 'num labels', num_labels

    config.num_emb=num_emb
    config.output_dim = num_labels

    config.maxseqlen=utils.get_max_len_data(data)
    config.maxnodesize=utils.get_max_node_size(data)
    config.maxrelsize=utils.get_max_rel_size(data)
    config.maxnonrelTsize,config.maxnonrelPsize,config.maxnonrelMsize=utils.get_max_nonrel_size(data)

    print config.maxnodesize,config.maxseqlen, config.maxrelsize,config.maxnonrelTsize,config.maxnonrelPsize,config.maxnonrelMsize, " maxsize"
    random.seed()
    np.random.seed()

    with tf.Graph().as_default():

        model = tf_tree_lstm.tf_NarytreeLSTM(config)

        init=tf.global_variables_initializer()
        saver = tf.train.Saver()
        best_valid_score=0.0
        best_valid_epoch=0
        dev_score=0.0
        test_score=0.0
        with tf.Session() as sess:

            sess.run(init)
            start_time=time.time()

            for epoch in range(config.num_epochs):
                print 'epoch', epoch
                avg_loss=0.0
                avg_loss = train_epoch(model, train_set, sess)
                print 'avg loss', avg_loss

                dev_score, dev_rel_score, dev_pred_res, dev_label_res=evaluate(model,dev_set,sess)
                print 'dev-score', dev_score
		print 'dev-rel-score', dev_rel_score
		
		dev_pred_res=np.reshape(dev_pred_res, [-1])
                dev_label_res=np.reshape(dev_label_res, [-1])
	        dev_label_res=dev_label_res.astype(int)
	        confusion = calculate_confusion(dev_pred_res, dev_label_res)
        	cm = confusion.copy()
        	cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        	cm = cm[np.newaxis, :, :, np.newaxis].astype(np.float32)
        	cm_tf_image = tf.convert_to_tensor(cm)
        	cm_is = tf.image_summary("confusion_matrix", cm_tf_image)
        	cm_current_epoch = sess.run(cm_is)
        	print_confusion(confusion, num_to_tag)
                

                if dev_score > best_valid_score:
                    best_valid_score=dev_score
                    best_valid_epoch=epoch
                    saver.save(sess,'./ckpt/tree_rnn_weights')

                #if epoch-best_valid_epoch > config.early_stopping:
                    #break

                print "time per epochis {0}".format(time.time()-start_time)

            test_score,test_rel_score, test_pred_res, test_label_res = evaluate(model,test_set, sess)
            
	    
            #test_pred_res=np.reshape(test_pred_res, [-1])
            #test_label_res=np.reshape(test_label_res, [-1])
            test_pred_res=np.reshape(dev_pred_res, [-1])
            test_label_res=np.reshape(dev_label_res, [-1])
	    test_label_res=test_label_res.astype(int)
            
	    confusion = calculate_confusion(test_pred_res, test_label_res)
            cm = confusion.copy()
            cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
            cm = cm[np.newaxis, :, :, np.newaxis].astype(np.float32)
            cm_tf_image = tf.convert_to_tensor(cm)
            cm_is = tf.image_summary("confusion_matrix", cm_tf_image)
            cm_current_epoch = sess.run(cm_is)
            print_confusion(confusion, num_to_tag)    
            

            print test_score,'test_score'
	    print test_rel_score,'test_rel_score'

def train_epoch(model,data,sess):
    loss=model.train(data,sess)
    return loss

def evaluate(model,data,sess):
    acc, acc_rel, pred_r, label_r=model.evaluate(data,sess)
    return acc, acc_rel, pred_r, label_r

if __name__ == '__main__':
    train()

