from tf_treenode import tNode,processTree
import numpy as np
import os
import random

#flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
#flatten(a)

class Vocab(object):

    def __init__(self,path):
        self.words = []
        self.word2idx = {}
	self.idx2word={}
        self.unk_index = None
        self.unk_token = '<unk>'
	self.load(path)

    def load(self,path):
        with open(path,'r') as f:
            for line in f:
                w=line.strip()
                assert w not in self.words
                self.words.append(w)
		self.word2idx[w] = len(self.words)-1		
                self.idx2word[self.word2idx[w]]=w
	self.unk_index=self.word2idx[self.unk_token]

    def __len__(self):
        return len(self.words)

    def encode(self,word):
	if self.unk_index is None:
            assert word in self.word2idx
        return self.word2idx.get(word, self.unk_index)

    def decode(self,idx):
        assert idx >= len(self.words)
        return self.idx2word[idx]

    def size(self):
        return len(self.words)

    def writeVocab(self):
	with open('Vocab.txt','w') as f:
	    for w in self.words:
		f.write(w+'\n')

def load_tree_data(data_dir):
    voc=Vocab(os.path.join(data_dir,'Vocab.txt'))
    print 'Vocab: ', len(voc)

    split_paths={}
    for split in ['train','test','dev']:
        split_paths[split]=os.path.join(data_dir,split)

    fnlist=[tNode.encodetokens]
    arglist=[voc.encode]

    data={}
    for split,path in split_paths.iteritems():
        sentencepath=os.path.join(path,'sents.txt')
        treepath=os.path.join(path,'parents.txt')
        labelpath=os.path.join(path,'labels.txt')
	relpath=os.path.join(path,'rels.txt')
        nonrelpath=os.path.join(path,'nonrels.txt')
        trees=parse_trees(sentencepath,treepath,labelpath,relpath,nonrelpath)
        trees = [(processTree(tree,fnlist,arglist),rels,nrels) for tree,rels,nrels in trees]
        data[split]=trees

    return data,voc


def parse_trees(sentencepath, treepath, labelpath, relpath, nonrelpath):
    trees=[]
    with open(treepath,'r') as ft, open (labelpath) as fl, open(relpath,'r') as fr, open(nonrelpath,'r') as fn, open(
        sentencepath,'r') as f:
        while True:
            parentidxs = ft.readline()
            labels = fl.readline()
            sentence=f.readline()
	    rels=fr.readline()
	    nonrels=fn.readline()
            if not parentidxs or not labels or not sentence:
                break
            parentidxs=[int(p) for p in parentidxs.strip().split() ]
            labels=[int(l) if l != '#' else None for l in labels.strip().split()]
            rels=[int(r) for r in rels.strip().split()]
            relT=[]
            relP=[]
            relM=[]
	    for i in range(0,len(rels),2):
		if labels[rels[i]]==1:
		    relT.append(rels[i])
		    relT.append(rels[i+1])
		elif labels[rels[i]]==2:
		    relP.append(rels[i])
		    relP.append(rels[i+1])
		elif labels[rels[i]]==3:
		    relM.append(rels[i])
		    relM.append(rels[i+1])
		else:
		    print "Error!"
	    rels=[]
	    rels.append(relT)
	    rels.append(relP)
	    rels.append(relM)
            nonrels=nonrels.strip().split(',')
            nrels=[]
            for nrstr in nonrels:
                nrels.append([int(nri) for nri in nrstr.strip().split()])
            if len(nrels)==1:
		nrels=[[],[],[]]
            if len(parentidxs)>1:
                tree=parse_tree(sentence,parentidxs,labels)
                trees.append([tree, rels, nrels])
    return trees


def parse_tree(sentence,parents,labels):
    nodes = {}
    parents = [p - 1 for p in parents]  #change to zero based
    sentence=[w for w in sentence.strip().split()]
    for i in xrange(len(parents)):
        if i not in nodes:
            idx = i
            prev = None
            while True:
                node = tNode(idx)  
                if prev is not None:
                    assert prev.idx != node.idx
                    node.add_child(prev)

                node.label = labels[idx]
                nodes[idx] = node

                if idx < len(sentence):
                    node.word = sentence[idx].lower()

                parent = parents[idx]
                if parent in nodes:
                    assert len(nodes[parent].children) < 2
                    nodes[parent].add_child(node)
                    break
                elif parent == -1:
                    root = node
                    break

                prev = node
                idx = parent

    return root

def BFStree(root):
    from collections import deque
    node=root
    leaves=[]
    inodes=[]
    queue=deque([node])
    func=lambda node:node.children==[]

    while queue:
        node=queue.popleft()
        if func(node):
            leaves.append(node)
        else:
            inodes.append(node)
        if node.children:
            queue.extend(node.children)

    return leaves,inodes

def extract_tree_data(tree,rels,nonrels,max_degree=2,with_labels=False):
    leaves,inodes=BFStree(tree)
    labels=[]
    leaf_emb=[]
    tree_str=[]
    rel_strT = []
    rel_strP = []
    rel_strM = []
    nonrel_strT = []
    nonrel_strP = []
    nonrel_strM = []
    nrT=nonrels[0]
    nrP=nonrels[1]
    nrM=nonrels[2]
    rT=rels[0]
    rP=rels[1]
    rM=rels[2]
    index_dict={}
    i=0
    for leaf in reversed(leaves):
        index_dict[leaf.idx]=i
        leaf.idx = i
        i+=1
        labels.append(leaf.label)
        leaf_emb.append(leaf.word)
    for node in reversed(inodes):
        index_dict[node.idx]=i
        node.idx=i
        c=[child.idx for child in node.children]
        tree_str.append(c)
        labels.append(node.label)
        i+=1
    for ri in range(0,len(rT),2):
	r=[index_dict[rT[ri]],index_dict[rT[ri+1]]]
	rel_strT.append(r)
    for ri in range(0,len(rP),2):
	r=[index_dict[rP[ri]],index_dict[rP[ri+1]]]
	rel_strP.append(r)
    for ri in range(0,len(rM),2):
	r=[index_dict[rM[ri]],index_dict[rM[ri+1]]]
	rel_strM.append(r)
    for nri in range(0,len(nrT),2):
	nr=[index_dict[nrT[nri]],index_dict[nrT[nri+1]]]
	nonrel_strT.append(nr)
    for nri in range(0,len(nrP),2):
	nr=[index_dict[nrP[nri]],index_dict[nrP[nri+1]]]
	nonrel_strP.append(nr)
    for nri in range(0,len(nrM),2):
	nr=[index_dict[nrM[nri]],index_dict[nrM[nri+1]]]
	nonrel_strM.append(nr)
    if with_labels:
        labels_exist = [l is not None for l in labels]
        labels = [l or 0 for l in labels]
        return np.array(leaf_emb,dtype='int32'), np.array(tree_str,dtype='int32'), np.array(labels,dtype=float), np.array(rel_strT,dtype='int32'),np.array(rel_strP,dtype='int32'),np.array(rel_strM,dtype='int32'),np.array(nonrel_strT,dtype='int32'),np.array(nonrel_strP,dtype='int32'),np.array(nonrel_strM,dtype='int32')
    else:
        return np.array(leaf_emb,dtype='int32'), np.array(tree_str,dtype='int32'), np.array(rel_strT,dtype='int32'),np.array(rel_strP,dtype='int32'),np.array(rel_strM,dtype='int32'),np.array(nonrel_strT,dtype='int32'),np.array(nonrel_strP,dtype='int32'),np.array(nonrel_strM,dtype='int32')

def extract_batch_tree_data(batchdata,fillnum, relnum, nonrelTnum, nonrelPnum, nonrelMnum):

    dim1,dim2,dim3=len(batchdata),fillnum, relnum
    leaf_emb_arr = np.empty([dim1,dim2],dtype='int32')
    leaf_emb_arr.fill(-1)
    treestr_arr = np.empty([dim1,dim2,2],dtype='int32')
    treestr_arr.fill(-1)
    labels_arr = np.empty([dim1,dim2],dtype=float)
    labels_arr.fill(-1)
    relstrT_arr = np.empty([dim1,dim3,2],dtype='int32')
    relstrT_arr.fill(-1)
    relstrP_arr = np.empty([dim1,dim3,2],dtype='int32')
    relstrP_arr.fill(-1)
    relstrM_arr = np.empty([dim1,dim3,2],dtype='int32')
    relstrM_arr.fill(-1)
    nonrelstrT_arr = np.empty([dim1,nonrelTnum,2],dtype='int32')
    nonrelstrT_arr.fill(-1)
    nonrelstrP_arr = np.empty([dim1,nonrelPnum,2],dtype='int32')
    nonrelstrP_arr.fill(-1)
    nonrelstrM_arr = np.empty([dim1,nonrelMnum,2],dtype='int32')
    nonrelstrM_arr.fill(-1)
    for i,(tree,rel,nonrel) in enumerate(batchdata):
        input_,treestr,labels,relstrT,relstrP,relstrM,nonrelstrT,nonrelstrP,nonrelstrM = extract_tree_data(tree,rel,nonrel, max_degree=2, with_labels = True)
        leaf_emb_arr[i,0:len(input_)]=input_
        treestr_arr[i,0:len(treestr),0:2]=treestr
        labels_arr[i,0:len(labels)]=labels
        if np.shape(relstrT)[0]!=0:    #in case of the error: could not broadcast input array from shape(0) to shape(0,2)
            relstrT_arr[i,0:len(relstrT),0:2]=relstrT
        if np.shape(relstrP)[0]!=0:    
            relstrP_arr[i,0:len(relstrP),0:2]=relstrP
        if np.shape(relstrM)[0]!=0:    
            relstrM_arr[i,0:len(relstrM),0:2]=relstrM

        if np.shape(nonrelstrT)[0]!=0:    
            nonrelstrT_arr[i,0:len(nonrelstrT),0:2]=nonrelstrT
        if np.shape(nonrelstrP)[0]!=0:    
            nonrelstrP_arr[i,0:len(nonrelstrP),0:2]=nonrelstrP
        if np.shape(nonrelstrM)[0]!=0:    
            nonrelstrM_arr[i,0:len(nonrelstrM),0:2]=nonrelstrM

    return leaf_emb_arr,treestr_arr,labels_arr,relstrT_arr,relstrP_arr,relstrM_arr,nonrelstrT_arr,nonrelstrP_arr,nonrelstrM_arr

def extract_seq_data(data,numsamples=0,fillnum=100):
    seqdata=[]
    seqlabels=[]
    for tree,_ in data:
        seq,seqlbls=extract_seq_from_tree(tree,numsamples)
        seqdata.extend(seq)
        seqlabels.extend(seqlbls)

    seqlngths=[len(s) for s in seqdata]
    maxl=max(seqlngths)
    assert fillnum >=maxl
    if 1:
        seqarr=np.empty([len(seqdata),fillnum],dtype='int32')
        seqarr.fill(-1)
        for i,s in enumerate(seqdata):
            seqarr[i,0:len(s)]=np.array(s,dtype='int32')
        seqdata=seqarr
    return seqdata,seqlabels,seqlngths,maxl

def extract_seq_from_tree(tree,numsamples=0):

    if tree.span is None:
        tree.postOrder(tree,tree.get_spans)

    seq,lbl=[],[]
    s,l=tree.span,tree.label
    seq.append(s)
    lbl.append(l)

    if not numsamples:
        return seq,lbl


    num_nodes = tree.idx
    if numsamples==-1:
        numsamples=num_nodes
    #numsamples=min(numsamples,num_nodes)
    #sampled_idxs = random.sample(range(num_nodes),numsamples)
    #sampled_idxs=range(num_nodes)
    #print sampled_idxs,num_nodes

    subtrees={}
    #subtrees[tree.idx]=
    #func=lambda tr,su:su.update([(tr.idx,tr)])
    def func_(self,su):
        su.update([(self.idx,self)])

    tree.postOrder(tree,func_,subtrees)

    for j in xrange(numsamples):#sampled_idxs:
        i=random.randint(0,num_nodes)
        root = subtrees[i]
        s,l=root.span,root.label
        seq.append(s)
        lbl.append(l)

    return seq,lbl

def get_max_len_data(datadic):
    maxlen=0
    for data in datadic.values():
        for tree,_,_ in data:
            tree.postOrder(tree,tree.get_numleaves)
            #assert tree.num_leaves > 1
            if tree.num_leaves > maxlen:
                maxlen=tree.num_leaves

    return maxlen

def get_max_node_size(datadic):
    maxsize=0
    for data in datadic.values():
        for tree,_,_ in data:
            tree.postOrder(tree,tree.get_size)
            #assert tree.size > 1
            if tree.size > maxsize:
                maxsize=tree.size

    return maxsize

def get_max_rel_size(datadic):
    maxsize=0
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    for data in datadic.values():
        for _,rel,_ in data:
	    relc=flatten(rel)
            s=len(relc)
            #assert s > 1
            if s > maxsize:
                maxsize=s
    return maxsize

def getTotalrelnum(data):
    num=0
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    for _,rel,_ in data:
	relc = flatten(rel)
	num += len(relc)
    return num/2
	
def get_max_nonrel_size(datadic):
    maxsizeT=0
    maxsizeP=0
    maxsizeM=0
    for data in datadic.values():
        for _,_,nonrel in data:
            st=len(nonrel[0])
            sp=len(nonrel[1])
	    sm=len(nonrel[2])
            #assert s > 1
            if st > maxsizeT:
                maxsizeT=st
	    if sp > maxsizeP:
                maxsizeP=sp
	    if sm > maxsizeM:
                maxsizeM=sm
    return maxsizeT, maxsizeP, maxsizeM

def get_mask(data,fixlen):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])
    # Mask of valid places in each row
    mask = np.arange(fixlen) < lens[:,None]
    return mask


def numpy_fillna(data,mask):
    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=np.int32)
    out[mask] = np.concatenate(data)
    return out

def genPair_list(element_list):
    pairlist=[]
    for i in range(len(element_list)-1):
	for j in range(i+1,len(element_list),1):
	    pairlist.append([element_list[i],element_list[j]])
	    pairlist.append([element_list[j],element_list[i]])
    return pairlist

def isRel(pair, rels):
    for r in rels:
        if pair[0]==r[0] and pair[1]==r[1]:
            return True
	if rels[0]==-1:
            break
    return False


def test_fn():
    data_dir='./stanford_lstm/data/sst'
    fine_grained=0
    data,_=load_sentiment_treebank(data_dir,fine_grained)
    for d in data.itervalues():
        print len(d)

    d=data['dev']
    a,b,c,_=extract_seq_data(d[0:1],5)
    print a,b,c

    print get_max_len_data(data)
    return data
if __name__=='__main__':
    test_fn()
