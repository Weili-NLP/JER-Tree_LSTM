import numpy as np
import tensorflow as tf
import os
import sys

from tf_data_utils import extract_tree_data,extract_batch_tree_data,get_mask,numpy_fillna,genPair_list,isRel,getTotalrelnum

class tf_NarytreeLSTM(object):

    def __init__(self,config):
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.num_emb = config.num_emb
        self.output_dim = config.output_dim
        self.config=config
        self.batch_size=config.batch_size
        self.reg=self.config.reg
        self.degree=config.degree
        assert self.emb_dim > 1 and self.hidden_dim > 1

        self.add_placeholders()

        emb_leaves = self.add_embedding()

        self.add_model_variables()

        batch_loss, batch_lossr, self.corrent_num, self.total_num = self.compute_loss(emb_leaves)

        self.loss,self.total_loss=self.calc_batch_loss(batch_loss, batch_lossr)

        self.train_op1,self.train_op2 = self.add_training_op()

        #self.train_op = self.add_training_op_old()
        

    def add_embedding(self):

        #embed=np.load('glove{0}_uniform.npy'.format(self.emb_dim))
        with tf.variable_scope("Embed",regularizer=None):
            embedding=tf.get_variable('embedding',[self.num_emb,
                                                   self.emb_dim]
                        ,initializer=tf.random_uniform_initializer(-0.05,0.05),trainable=True,regularizer=None)
            ix=tf.to_int32(tf.not_equal(self.input,-1))*self.input
            emb_tree=tf.nn.embedding_lookup(embedding,ix)
            emb_tree=emb_tree*(tf.expand_dims(
                        tf.to_float(tf.not_equal(self.input,-1)),2))

            return emb_tree


    def add_placeholders(self):
        dim2=self.config.maxnodesize
        dim1=self.config.batch_size
        dim3=self.config.maxrelsize

        dim4=self.config.maxnonrelTsize
        dim5=self.config.maxnonrelPsize
        dim6=self.config.maxnonrelMsize

        self.input = tf.placeholder(tf.int32,[dim1,dim2],name='input')
        self.treestr = tf.placeholder(tf.int32,[dim1,dim2,2],name='tree')
        #self.relstr = tf.placeholder(tf.int32,[dim1,dim3,2],name='rel')

	self.relstrT = tf.placeholder(tf.int32,[dim1,dim3,2],name='relT')
	self.relstrP = tf.placeholder(tf.int32,[dim1,dim3,2],name='relP')
	self.relstrM = tf.placeholder(tf.int32,[dim1,dim3,2],name='relM')

        self.nonrelstrT = tf.placeholder(tf.int32,[dim1,dim4,2],name='nrelT')
        self.nonrelstrP = tf.placeholder(tf.int32,[dim1,dim5,2],name='nrelP')
        self.nonrelstrM = tf.placeholder(tf.int32,[dim1,dim6,2],name='nrelM')

        self.labels = tf.placeholder(tf.int32,[dim1,dim2],name='labels')
        self.dropout = tf.placeholder(tf.float32,name='dropout')

        self.n_inodes = tf.reduce_sum(tf.to_int32(tf.not_equal(self.treestr,-1)),[1,2])
        self.n_inodes = self.n_inodes/2

	#self.num_rels = tf.reduce_sum(tf.to_int32(tf.not_equal(self.relstr,-1)),[1,2])
	#self.num_rels = self.num_rels/2

	self.num_relsT = tf.reduce_sum(tf.to_int32(tf.not_equal(self.relstrT,-1)),[1,2])
	self.num_relsT = self.num_relsT/2

	self.num_relsP = tf.reduce_sum(tf.to_int32(tf.not_equal(self.relstrP,-1)),[1,2])
	self.num_relsP = self.num_relsP/2

	self.num_relsM = tf.reduce_sum(tf.to_int32(tf.not_equal(self.relstrM,-1)),[1,2])
	self.num_relsM = self.num_relsM/2

	self.num_nonrelsT = tf.reduce_sum(tf.to_int32(tf.not_equal(self.nonrelstrT,-1)),[1,2])
	self.num_nonrelsT = self.num_nonrelsT/2

	self.num_nonrelsP = tf.reduce_sum(tf.to_int32(tf.not_equal(self.nonrelstrP,-1)),[1,2])
	self.num_nonrelsP = self.num_nonrelsP/2

	self.num_nonrelsM = tf.reduce_sum(tf.to_int32(tf.not_equal(self.nonrelstrM,-1)),[1,2])
	self.num_nonrelsM = self.num_nonrelsM/2

        self.num_leaves = tf.reduce_sum(tf.to_int32(tf.not_equal(self.input,-1)),[1])
        self.batch_len = tf.placeholder(tf.int32,name="batch_len")

    def calc_wt_init(self,fan_in=300):
        eps=1.0/np.sqrt(fan_in)
        return eps

    def add_model_variables(self):

        with tf.variable_scope("Composition",
                                initializer=
                                tf.contrib.layers.xavier_initializer(),
                                regularizer=
                                tf.contrib.layers.l2_regularizer(self.config.reg
            )):

            cU = tf.get_variable("cU",[self.emb_dim,2*self.hidden_dim],initializer=tf.random_uniform_initializer(-self.calc_wt_init(),self.calc_wt_init()))
            cW = tf.get_variable("cW",[self.degree*self.hidden_dim,(self.degree+3)*self.hidden_dim],initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim)))
            cb = tf.get_variable("cb",[4*self.hidden_dim],initializer=tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))


        with tf.variable_scope("Projection",regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):

            U = tf.get_variable("U",[self.output_dim,self.hidden_dim],
                                initializer=tf.random_uniform_initializer(self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim))
                                    )
            bu = tf.get_variable("bu",[self.output_dim],initializer=
                                 tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))

	with tf.variable_scope("Transformation",regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):

            Tw = tf.get_variable("Tw",[self.hidden_dim,self.hidden_dim],
                                initializer=tf.random_uniform_initializer(self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim)))

	with tf.variable_scope("Classification",regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):

            Rt = tf.get_variable("Rt",[3*self.hidden_dim,2],
                                initializer=tf.random_uniform_initializer(self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim)))
	
	    Rp = tf.get_variable("Rp",[3*self.hidden_dim,2],
                                initializer=tf.random_uniform_initializer(self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim)))
	    
	    Rm = tf.get_variable("Rm",[3*self.hidden_dim,2],
                                initializer=tf.random_uniform_initializer(self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim)))
            
	    bt = tf.get_variable("bt",[2],initializer=
                                 tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))
	
	    bp = tf.get_variable("bp",[2],initializer=
                                 tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))

	    bm = tf.get_variable("bm",[2],initializer=
                                 tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))

    def process_leafs(self,emb):

        with tf.variable_scope("Composition",reuse=True):
            cU = tf.get_variable("cU",[self.emb_dim,2*self.hidden_dim])
            cb = tf.get_variable("cb",[4*self.hidden_dim])
            b = tf.slice(cb,[0],[2*self.hidden_dim])
            def _recurseleaf(x):

                concat_uo = tf.matmul(tf.expand_dims(x,0),cU) + b
                u,o = tf.split(1,2,concat_uo)
                o=tf.nn.sigmoid(o)
                u=tf.nn.tanh(u)

                c = u#tf.squeeze(u)
                h = o * tf.nn.tanh(c)


                hc = tf.concat(1,[h,c])
                hc=tf.squeeze(hc)
                return hc

        hc = tf.map_fn(_recurseleaf,emb)
        return hc


    def compute_loss(self,emb_batch,curr_batch_size=None):
        outloss_e=[]
        outloss_r=[]
        outloss_c=[]
	correct_num = 0
        total_data = 0
        prediction=[]
	prediction_relT=[]
	prediction_relP=[]
	prediction_relM=[]
        for idx_batch in range(self.config.batch_size):

            tree_states=self.compute_states(emb_batch,idx_batch)
            #print tree_states.get_shape().as_list()
            #tree_hat = tf.matmul(tree_states,Tw)

	    #loss_r=self.compute_maxmargin(tree_states, idx_batch)
            #outloss_r.append(loss_r)

	    loss_r=self.compute_crossentropy(tree_states, idx_batch)
            outloss_r.append(loss_r)

	    predT,predP,predM = self.rel_predict(tree_states, idx_batch)
	    prediction_relT.append(predT)
	    prediction_relP.append(predP)
	    prediction_relM.append(predM)
            
            logits = self.create_output(tree_states)
            labels1=tf.gather(self.labels,idx_batch)
            labels2=tf.reduce_sum(tf.to_int32(tf.not_equal(labels1,-1)))
            total_data += labels2
            labels=tf.gather(labels1,tf.range(labels2))
            loss_e = self.calc_loss(logits,labels)

            pred = tf.nn.softmax(logits)
            one_hot_pred = tf.argmax(pred,1)
	    
	    correct_pred_num = tf.reduce_sum(tf.cast(tf.equal(labels, tf.cast(one_hot_pred,"int32")),"int32"))
            correct_num += correct_pred_num

	    one_hot_pred = tf.reshape(one_hot_pred,[-1])
            prediction.append(one_hot_pred)
            outloss_e.append(loss_e)

        batch_loss_e=tf.pack(outloss_e)
        batch_loss_r=tf.pack(outloss_r)

        self.pred = prediction
	self.predrelT = prediction_relT
	self.predrelP = prediction_relP
	self.predrelM = prediction_relM
        return batch_loss_e, batch_loss_r, correct_num, total_data
    
    '''
    l2_loss_pairs = tf.reduce_sum(tf.square(output_left - output_right), 1)
    positive_loss = l2_loss_pairs
    negative_loss = tf.nn.relu(margin - l2_loss_pairs)
    final_loss = tf.mul(labels_node, positive_loss) + tf.mul(1. - labels_node, negative_loss)
    '''
    def rel_predict(self, tree_h, idx_batch):
	num_relsT = tf.gather(self.num_relsT,idx_batch)
        relstrT = tf.gather(tf.gather(self.relstrT,idx_batch),tf.range(num_relsT))
	num_relsP = tf.gather(self.num_relsP,idx_batch)
        relstrP = tf.gather(tf.gather(self.relstrP,idx_batch),tf.range(num_relsP))
	num_relsM = tf.gather(self.num_relsM,idx_batch)
        relstrM = tf.gather(tf.gather(self.relstrM,idx_batch),tf.range(num_relsM))
	with tf.variable_scope("Classification", reuse=True):
	    Rt = tf.get_variable("Rt",[3*self.hidden_dim,2])	
	    Rp = tf.get_variable("Rp",[3*self.hidden_dim,2])	    
	    Rm = tf.get_variable("Rm",[3*self.hidden_dim,2])            
	    bt = tf.get_variable("bt",[2])	
	    bp = tf.get_variable("bp",[2])
	    bm = tf.get_variable("bm",[2])
	    
	    indicesT = tf.reshape(relstrT, [-1])
            rel_h_T = tf.gather(tree_h,indicesT)
            rel_h_pair_T = tf.reshape(rel_h_T,[-1,2*self.hidden_dim])
            rel_ht_e1, rel_ht_e2 = tf.split(1,2,rel_h_pair_T)
            matT = tf.concat(1,[rel_ht_e1, rel_ht_e2, rel_ht_e1-rel_ht_e2])
	    
	    indicesP = tf.reshape(relstrP, [-1])
            rel_h_P = tf.gather(tree_h,indicesP)
            rel_h_pair_P = tf.reshape(rel_h_P,[-1,2*self.hidden_dim])
            rel_hp_e1, rel_hp_e2 = tf.split(1,2,rel_h_pair_P)
            matP = tf.concat(1,[rel_hp_e1, rel_hp_e2, rel_hp_e1-rel_hp_e2])
	    
	    indicesM = tf.reshape(relstrM, [-1])
            rel_h_M = tf.gather(tree_h,indicesM)
            rel_h_pair_M = tf.reshape(rel_h_M,[-1,2*self.hidden_dim])
            rel_hm_e1, rel_hm_e2 = tf.split(1,2,rel_h_pair_M)
            matM = tf.concat(1,[rel_hm_e1, rel_hm_e2, rel_hm_e1-rel_hm_e2])

	    yT_out = tf.matmul(matT, Rt) + bt
	    yP_out = tf.matmul(matP, Rp) + bp
            yM_out = tf.matmul(matM, Rm) + bm
	    
	    predT = tf.nn.softmax(yT_out)
	    rel_predT = tf.reshape(tf.argmax(predT,1),[-1])
	    predP = tf.nn.softmax(yP_out)
            rel_predP = tf.reshape(tf.argmax(predP,1),[-1])
	    predM = tf.nn.softmax(yM_out)
            rel_predM = tf.reshape(tf.argmax(predM,1),[-1])
	return rel_predT,rel_predP,rel_predM

    def compute_crossentropy(self,tree_h, idx_batch):
	num_relsT = tf.gather(self.num_relsT,idx_batch)
        relstrT = tf.gather(tf.gather(self.relstrT,idx_batch),tf.range(num_relsT))
	num_relsP = tf.gather(self.num_relsP,idx_batch)
        relstrP = tf.gather(tf.gather(self.relstrP,idx_batch),tf.range(num_relsP))
	num_relsM = tf.gather(self.num_relsM,idx_batch)
        relstrM = tf.gather(tf.gather(self.relstrM,idx_batch),tf.range(num_relsM))
	
	num_T = tf.gather(self.num_nonrelsT,idx_batch)
	num_P = tf.gather(self.num_nonrelsP,idx_batch)
	num_M = tf.gather(self.num_nonrelsM,idx_batch)
	nonrelstrT = tf.gather(tf.gather(self.nonrelstrT,idx_batch),tf.range(num_T))
        nonrelstrP = tf.gather(tf.gather(self.nonrelstrP,idx_batch),tf.range(num_P))
        nonrelstrM = tf.gather(tf.gather(self.nonrelstrM,idx_batch),tf.range(num_M))

	relT_labels = tf.ones([num_relsT],tf.int32)
	relP_labels = tf.ones([num_relsP],tf.int32)
	relM_labels = tf.ones([num_relsM],tf.int32)
        nonrelT_labels = tf.zeros([num_T],tf.int32)
	nonrelP_labels = tf.zeros([num_P],tf.int32)
	nonrelM_labels = tf.zeros([num_M],tf.int32)
	'''
        #V0: do not distinguish task process and material
	with tf.variable_scope("Classification", reuse=True):
	    merged = tf.concat(0,[relstrT,nonrelstrT,relstrP,nonrelstrP,relstrM,nonrelstrM])
	    indices = tf.reshape(merged, [-1])
            rel_h = tf.gather(tree_h,indices)
            rel_h_pair = tf.reshape(rel_h,[-1,2*self.hidden_dim])
            rel_h_e1, rel_h_e2 = tf.split(1,2,rel_h_pair)
            rel_mat = tf.concat(1,[rel_h_e1, rel_h_e2, rel_h_e1-rel_h_e2])
	    Rt = tf.get_variable("Rt",[3*self.hidden_dim,2])	    
	    bt = tf.get_variable("bt",[2])	
	    y_labels = tf.concat(0,[relT_labels,nonrelT_labels,relP_labels,nonrelP_labels,relM_labels,nonrelM_labels]) 
	    #y_labels = tf.reshape(y_labels,[-1])
	    y_out = tf.matmul(rel_mat, Rt) + bt
            #y_out = tf.cast(y_out,"float32")
	    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_out,y_labels)
	    loss = tf.reduce_sum(loss, [0])
        '''
        #V1: distinguish task process and material
        with tf.variable_scope("Classification", reuse=True):
	    Rt = tf.get_variable("Rt",[3*self.hidden_dim,2])	
	    Rp = tf.get_variable("Rp",[3*self.hidden_dim,2])	    
	    Rm = tf.get_variable("Rm",[3*self.hidden_dim,2])            
	    bt = tf.get_variable("bt",[2])	
	    bp = tf.get_variable("bp",[2])
	    bm = tf.get_variable("bm",[2])
	    #nT = tf.to_int32(num_relsT+num_T)
	    #nP = tf.to_int32(num_relsP+num_P)
	    #nM = tf.to_int32(num_relsM+num_M)
	    
	    #matT,matP,matM = tf.split(0,tf.convert_to_tensor([nT,nP,nM]),rel_mat)  split size should be known at graph construction.

            merged = tf.concat(0,[relstrT,nonrelstrT])
	    indices = tf.reshape(merged, [-1])
            rel_h = tf.gather(tree_h,indices)
            rel_h_pair = tf.reshape(rel_h,[-1,2*self.hidden_dim])
            rel_h_e1, rel_h_e2 = tf.split(1,2,rel_h_pair)
            matT = tf.concat(1,[rel_h_e1, rel_h_e2, rel_h_e1-rel_h_e2])

	    merged = tf.concat(0,[relstrP,nonrelstrP])
	    indices = tf.reshape(merged, [-1])
            rel_h = tf.gather(tree_h,indices)
            rel_h_pair = tf.reshape(rel_h,[-1,2*self.hidden_dim])
            rel_h_e1, rel_h_e2 = tf.split(1,2,rel_h_pair)
            matP = tf.concat(1,[rel_h_e1, rel_h_e2, rel_h_e1-rel_h_e2])

	    merged = tf.concat(0,[relstrM,nonrelstrM])
	    indices = tf.reshape(merged, [-1])
            rel_h = tf.gather(tree_h,indices)
            rel_h_pair = tf.reshape(rel_h,[-1,2*self.hidden_dim])
            rel_h_e1, rel_h_e2 = tf.split(1,2,rel_h_pair)
            matM = tf.concat(1,[rel_h_e1, rel_h_e2, rel_h_e1-rel_h_e2])

	    yT_out = tf.matmul(matT, Rt) + bt
	    yT_labels = tf.concat(0,[relT_labels,nonrelT_labels])
	    lossT = tf.nn.sparse_softmax_cross_entropy_with_logits(yT_out,yT_labels)
	    
	    yP_out = tf.matmul(matP, Rp) + bp
	    yP_labels = tf.concat(0,[relP_labels,nonrelP_labels])
	    lossP = tf.nn.sparse_softmax_cross_entropy_with_logits(yP_out,yP_labels)

            yM_out = tf.matmul(matM, Rm) + bm
	    yM_labels = tf.concat(0,[relM_labels,nonrelM_labels])
	    lossM = tf.nn.sparse_softmax_cross_entropy_with_logits(yM_out,yM_labels)

	    loss = tf.concat(0,[lossT,lossP,lossM])
	    loss = tf.reduce_sum(loss, [0])
	return loss

    def compute_maxmargin(self,tree_h, idx_batch):
        num_relsT = tf.gather(self.num_relsT,idx_batch)
        relstrT = tf.gather(tf.gather(self.relstrT,idx_batch),tf.range(num_relsT))
	num_relsP = tf.gather(self.num_relsP,idx_batch)
        relstrP = tf.gather(tf.gather(self.relstrP,idx_batch),tf.range(num_relsP))
	num_relsM = tf.gather(self.num_relsM,idx_batch)
        relstrM = tf.gather(tf.gather(self.relstrM,idx_batch),tf.range(num_relsM))

        relstr = tf.concat(0,[relstrT,relstrP,relstrM])
	num_rels = num_relsT+num_relsP+num_relsM

        loss = tf.convert_to_tensor(0.0)
	idx_var = tf.convert_to_tensor(0)
        margin = tf.constant(1.)   
	with tf.variable_scope("Transformation", reuse=True):
	    Tw = tf.get_variable("Tw",[self.hidden_dim,self.hidden_dim]) 
	    def _recurrence1(loss, idx_var):
	        tree_hat = tf.matmul(tree_h,Tw)

                rel_node_idx = tf.gather(relstr,idx_var)
	        rel_node_h = tf.gather(tree_h,rel_node_idx)
                _,rel_h2 = tf.split(0,2,rel_node_h) 
                rel_node_hat = tf.gather(tree_hat,rel_node_idx)  
                rel_h1_hat,_ = tf.split(0,2,rel_node_hat)

                pos_loss = tf.reduce_sum(tf.square(rel_h1_hat - rel_h2))
                rel_h1_hat_t = tf.tile(rel_h1_hat,[tf.shape(tree_hat)[0],1])
                rel_h2_t = tf.tile(rel_h2,[tf.shape(tree_hat)[0],1])
                merged = tf.concat(0, [rel_h1_hat_t - tree_h, tree_hat - rel_h2_t])
                neg_loss = tf.reduce_min(tf.reduce_sum(tf.square(merged),reduction_indices=1))
                loss_l2 = pos_loss + tf.nn.relu(margin - neg_loss)
                loss = loss+loss_l2
                idx_var = tf.add(idx_var,1)
                return loss,idx_var
            loop_cond = lambda a1,idx_var: tf.less(idx_var,num_rels)
            loop_vars=[loss, idx_var]
            loss,_=tf.while_loop(loop_cond, _recurrence1, loop_vars, parallel_iterations=10)
            return loss


    def compute_states(self,emb,idx_batch=0):

        num_leaves = tf.squeeze(tf.gather(self.num_leaves,idx_batch))
        n_inodes = tf.gather(self.n_inodes,idx_batch)
        embx=tf.gather(tf.gather(emb,idx_batch),tf.range(num_leaves))
        treestr=tf.gather(tf.gather(self.treestr,idx_batch),tf.range(n_inodes))
        leaf_hc = self.process_leafs(embx)
        leaf_h,leaf_c=tf.split(1,2,leaf_hc)

        node_h=tf.identity(leaf_h)
        node_c=tf.identity(leaf_c)

        idx_var=tf.convert_to_tensor(0)

        with tf.variable_scope("Composition",reuse=True):

            cW = tf.get_variable("cW",[self.degree*self.hidden_dim,(self.degree+3)*self.hidden_dim])
            cb = tf.get_variable("cb",[4*self.hidden_dim])
            bu,bo,bi,bf=tf.split(0,4,cb)

            def _recurrence(node_h,node_c,idx_var):
                node_info=tf.gather(treestr,idx_var)

                child_h=tf.gather(node_h,node_info)
                child_c=tf.gather(node_c,node_info)

                flat_ = tf.reshape(child_h,[-1])
                tmp=tf.matmul(tf.expand_dims(flat_,0),cW)
                u,o,i,fl,fr=tf.split(1,5,tmp)

                i=tf.nn.sigmoid(i+bi)
                o=tf.nn.sigmoid(o+bo)
                u=tf.nn.tanh(u+bu)
                fl=tf.nn.sigmoid(fl+bf)
                fr=tf.nn.sigmoid(fr+bf)

                f=tf.concat(0,[fl,fr])
                c = i * u + tf.reduce_sum(f*child_c,[0])
                h = o * tf.nn.tanh(c)

                node_h = tf.concat(0,[node_h,h])

                node_c = tf.concat(0,[node_c,c])

                idx_var=tf.add(idx_var,1)

                return node_h,node_c,idx_var
            loop_cond = lambda a1,b1,idx_var: tf.less(idx_var,n_inodes)

            loop_vars=[node_h,node_c,idx_var]
            node_h,node_c,idx_var=tf.while_loop(loop_cond, _recurrence,
                                                loop_vars,parallel_iterations=10)

            return node_h


    def create_output(self,tree_states):

        with tf.variable_scope("Projection",reuse=True):

            U = tf.get_variable("U",[self.output_dim,self.hidden_dim],
                                )
            bu = tf.get_variable("bu",[self.output_dim])

            h=tf.matmul(tree_states,U,transpose_b=True)+bu
            return h



    def calc_loss(self,logits,labels):

        l1=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,labels)
        loss=tf.reduce_sum(l1,[0])
        return loss

    def calc_batch_loss(self, batch_loss, batch_lossr):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regpart=tf.add_n(reg_losses)
        losse=tf.reduce_mean(batch_loss)
        lossr=tf.reduce_mean(batch_lossr)
        loss=losse+lossr
        total_loss=loss+0.5*regpart
        return loss,total_loss

    def add_training_op_old(self):

        opt = tf.train.AdagradOptimizer(self.config.lr)
        train_op = opt.minimize(self.total_loss)
        return train_op

    def add_training_op(self):
        loss=self.total_loss
        opt1=tf.train.AdagradOptimizer(self.config.lr)
        opt2=tf.train.AdagradOptimizer(self.config.emb_lr)

        ts=tf.trainable_variables()
        gs=tf.gradients(loss,ts)
        gs_ts=zip(gs,ts)

        gt_emb,gt_nn=[],[]
        for g,t in gs_ts:
            #print t.name,g.name
            if "Embed/embedding:0" in t.name:
                #g=tf.Print(g,[g.get_shape(),t.get_shape()])
                gt_emb.append((g,t))
                #print t.name
            else:
                gt_nn.append((g,t))
                #print t.name

        train_op1=opt1.apply_gradients(gt_nn)
        train_op2=opt2.apply_gradients(gt_emb)
        train_op=[train_op1,train_op2]

        return train_op



    def train(self,data,sess):
        from random import shuffle
        data_idxs=range(len(data))
        shuffle(data_idxs)
        losses=[]
        for i in range(0,len(data),self.batch_size):
            batch_size = min(i+self.batch_size,len(data))-i
            if batch_size < self.batch_size:break

            batch_idxs=data_idxs[i:i+batch_size]
            batch_data=[data[ix] for ix in batch_idxs] #[i:i+batch_size]

            input_b,treestr_b,labels_b,relstrT_b,relstrP_b,relstrM_b,nonrelstrT_b,nonrelstrP_b,nonrelstrM_b=extract_batch_tree_data(batch_data,self.config.maxnodesize, self.config.maxrelsize, self.config.maxnonrelTsize, self.config.maxnonrelPsize, self.config.maxnonrelMsize)

            feed={self.input:input_b, self.treestr:treestr_b, self.labels:labels_b, self.relstrT:relstrT_b, self.relstrP:relstrP_b, self.relstrM:relstrM_b, self.nonrelstrT:nonrelstrT_b, self.nonrelstrP:nonrelstrP_b, self.nonrelstrM:nonrelstrM_b, self.dropout:self.config.dropout, self.batch_len:len(input_b)}

            loss,_,_=sess.run([self.loss,self.train_op1,self.train_op2],feed_dict=feed)
            #loss,_=sess.run([self.loss,self.train_op],feed_dict=feed)

            losses.append(loss)
            avg_loss=np.mean(losses)
            sstr='avg loss %.2f at example %d of %d\r' % (avg_loss, i, len(data))
            sys.stdout.write(sstr)
            sys.stdout.flush()

            #if i>1000: break
        return np.mean(losses)

    def gen_predlst(self, pred_val):
	pred_list_T=[]
	pred_list_P=[]
	pred_list_M=[]
	for pred in pred_val:
	    eT=[]
	    eP=[]
	    eM=[]
	    for i in range(len(pred)):
		if pred[i]==1: 
		    eT.append(i)
		elif pred[i]==2:
		    eP.append(i)
		elif pred[i]==3:
		    eM.append(i)		
	    pred_list_T.append(genPair_list(eT))
	    pred_list_P.append(genPair_list(eP))
	    pred_list_M.append(genPair_list(eM))
	return pred_list_T,pred_list_P,pred_list_M

    def evaluate(self,data,sess):
        num_correct=0
	num_correct_rel=0
	total_rel=0
	total_rel_find=0
        total_data=0
        data_idxs=range(len(data))
        test_batch_size=self.config.batch_size
        pred_res=[]
        labels_res=[]

        for i in range(0,len(data),test_batch_size):
            batch_size = min(i+test_batch_size,len(data))-i
            if batch_size < test_batch_size:break
            batch_idxs=data_idxs[i:i+batch_size]
            batch_data=[data[ix] for ix in batch_idxs]#[i:i+batch_size]
            input_b,treestr_b,labels_b,relstrT_b,relstrP_b,relstrM_b,nonrelstrT_b,nonrelstrP_b,nonrelstrM_b=extract_batch_tree_data(batch_data,self.config.maxnodesize, self.config.maxrelsize, self.config.maxnonrelTsize, self.config.maxnonrelPsize, self.config.maxnonrelMsize)

            feed={self.input:input_b,self.treestr:treestr_b,self.labels:labels_b,self.relstrT:relstrT_b, self.relstrP:relstrP_b, self.relstrM:relstrM_b, self.nonrelstrT:nonrelstrT_b, self.nonrelstrP:nonrelstrP_b, self.nonrelstrM:nonrelstrM_b, self.dropout:1.0,self.batch_len:len(input_b)}

            corrent_N, total_N, pred_val = sess.run([self.corrent_num,self.total_num, self.pred], feed_dict=feed)

	    # construct relation classification testset for T,P,M seperately.
	    pred_list_T,pred_list_P,pred_list_M = self.gen_predlst(pred_val)

	    # compute relation labels for T,P,M seperately.
	    relT=relstrT_b.tolist() #real relations
	    relP=relstrP_b.tolist()
	    relM=relstrM_b.tolist()
	    labelsT=[]
	    labelsP=[]
            labelsM=[]
	    
	    for k, predlst_t in enumerate(pred_list_T):
                l=[]
		for pred_rel in predlst_t:
		    if isRel(pred_rel, relT[k]):
			l.append(1)
                    else:
			l.append(0)
		labelsT.append(l)
	    for k, predlst_p in enumerate(pred_list_P):
		l=[]
		for pred_rel in predlst_p:
		    if isRel(pred_rel, relP[k]):
			l.append(1)
                    else:
			l.append(0)
		labelsP.append(l)
	    for k, predlst_m in enumerate(pred_list_M):	
		l=[]	
		for pred_rel in predlst_m:
		    if isRel(pred_rel, relM[k]):
			l.append(1)
                    else:
			l.append(0)
		labelsM.append(l)

	    # change predlist to numpy array
	    dim1=len(batch_data)
	    dim3=self.config.maxrelsize
	    relstrT_arr = np.empty([dim1,dim3,2],dtype='int32')
            relstrT_arr.fill(-1)
    	    relstrP_arr = np.empty([dim1,dim3,2],dtype='int32')
            relstrP_arr.fill(-1)
            relstrM_arr = np.empty([dim1,dim3,2],dtype='int32')
    	    relstrM_arr.fill(-1)
	    for i in range(dim1):
		relstrT=np.array(pred_list_T[i],dtype='int32')
		if np.shape(relstrT)[0]!=0:    
            	    relstrT_arr[i,0:len(relstrT),0:2]=relstrT
		relstrP=np.array(pred_list_P[i],dtype='int32')
        	if np.shape(relstrP)[0]!=0:    
                    relstrP_arr[i,0:len(relstrP),0:2]=relstrP
		relstrM=np.array(pred_list_M[i],dtype='int32')
        	if np.shape(relstrM)[0]!=0:    
            	    relstrM_arr[i,0:len(relstrM),0:2]=relstrM

	    feed1={self.input:input_b,self.treestr:treestr_b,self.labels:labels_b,self.relstrT:relstrT_arr, self.relstrP:relstrP_arr, self.relstrM:relstrM_arr, self.nonrelstrT:nonrelstrT_b, self.nonrelstrP:nonrelstrP_b, self.nonrelstrM:nonrelstrM_b, self.dropout:1.0, self.batch_len:dim1}
	    rel_predT,rel_predP,rel_predM = sess.run([self.predrelT,self.predrelP,self.predrelM], feed_dict=feed1)

	    #compute correct numbers of relation
	    for k in range(len(rel_predT)):
		for j in range(len(rel_predT[k])):
		    if rel_predT[k][j]==1:
			total_rel_find += 1
			if labelsT[k][j]==1:
			    num_correct_rel += 1

	    for k in range(len(rel_predP)):
		for j in range(len(rel_predP[k])):
		    if rel_predP[k][j]==1:
			total_rel_find += 1
			if labelsP[k][j]==1:
			    num_correct_rel += 1

	    for k in range(len(rel_predM)):
		for j in range(len(rel_predM[k])):
		    if rel_predM[k][j]==1:
			total_rel_find += 1
			if labelsM[k][j]==1:
			    num_correct_rel += 1 

            mask=get_mask(pred_val,self.config.maxnodesize)
            pred_val=numpy_fillna(pred_val,mask)	    
	    pred_val=np.reshape(pred_val, [-1])
            #print pred_val
            #print '\n'

            pred_res.extend(pred_val)
	    labels_res.extend(np.reshape(labels_b, [-1]))
            num_correct += corrent_N
	    total_data += total_N

        total_rel = getTotalrelnum(data)
	acc=float(num_correct)/float(total_data)
	relacc=float(num_correct_rel)/float(total_rel)
        return acc, relacc, pred_res, labels_res

