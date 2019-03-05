from utils import iob_iobes
import pdb
def iob2obes(fn, outfn):
    sen = []
    sen_tokens = []
    data = []
    data_tokens = []
    for line in open(fn):
        if not line.rstrip():
            sen = iob_iobes(sen)
            data.append(sen)
            for i in range(len(sen)):
                sen_tokens[i][1] = sen[i]
            data_tokens.append(sen_tokens)
            sen = []
            sen_tokens = []
            continue
                                            
        tokens = line.rstrip().split()
        label  = tokens[1]
        sen.append(label)
        sen_tokens.append(tokens)

    fid = open(outfn,'w')
    for sen in data_tokens:
        for tokens in sen:
            fid.write(' '.join(tokens)+'\n')
        fid.write('\n')

fn = '/homes/luanyi/pubanal/project/code/tagger_add_feature/data/test.feats'
outfn = '/homes/luanyi/pubanal/project/code/tagger_add_feature/data/.iobes'
iob2obes(fn, outfn)
    
        
    
