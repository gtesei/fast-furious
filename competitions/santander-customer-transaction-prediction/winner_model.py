import fastai
from fastai.tabular import *
from fastai.text import *
import feather
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from fastai.callbacks import SaveModelCallback
import logging

#logger
def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
    
logger = get_logger()

def auroc_score(input, target):
    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()
    return roc_auc_score(target, input)

# Callback to calculate AUC at the end of each epoch
class AUROC(Callback):
    _order = -20 #Needs to run before the recorder

    def __init__(self, learn, **kwargs): self.learn = learn
    def on_train_begin(self, **kwargs): self.learn.recorder.add_metric_names(['AUROC'])
    def on_epoch_begin(self, **kwargs): self.output, self.target = [], []
    
    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            self.output.append(last_output)
            self.target.append(last_target)
                
    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output)
            target = torch.cat(self.target)
            preds = F.softmax(output, dim=1)
            metric = auroc_score(preds, target)
            return add_metrics(last_metrics, [metric])

# Callback that do the shuffle augmentation        
class AugShuffCallback(LearnerCallback):
    def __init__(self, learn:Learner):
        super().__init__(learn)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        if not train: return
        m_pos = last_target==1
        m_neg = last_target==0
        
        pos_cat = last_input[0][m_pos]
        pos_cont = last_input[1][m_pos]
        
        neg_cat = last_input[0][m_neg]
        neg_cont = last_input[1][m_neg]
        
        for f in range(200):
            shuffle_pos = torch.randperm(pos_cat.size(0)).to(last_input[0].device)
            pos_cat[:,f] = pos_cat[shuffle_pos,f]
            pos_cont[:,f] = pos_cont[shuffle_pos, f]
            pos_cont[:,f+200] = pos_cont[shuffle_pos, f+200]
            
            shuffle_neg = torch.randperm(neg_cat.size(0)).to(last_input[0].device)
            neg_cat[:,f] = neg_cat[shuffle_neg,f]
            neg_cont[:, f] = neg_cont[shuffle_neg, f]
            neg_cont[:,f+200] = neg_cont[shuffle_neg, f+200]
        
        new_input = [torch.cat([pos_cat, neg_cat]), torch.cat([pos_cont, neg_cont])]
        new_target = torch.cat([last_target[m_pos], last_target[m_neg]])
        
        return {'last_input': new_input, 'last_target': new_target}
        
# Just a longer version of the random sampler : each samples is given "mult" times.
class LongerRandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, mult=3):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.mult = mult

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source) * self.mult

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples*self.mult,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist()*self.mult)

    def __len__(self):
        return len(self.data_source)*self.mult
        
# This is the NN structure, starting from fast.ai TabularModel.
class my_TabularModel(nn.Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False, 
                 cont_emb=2, cont_emb_notu=2):
        
        super().__init__()
        # "Continuous embedding NN for raw features"
        self.cont_emb = cont_emb[1]
        self.cont_emb_l = torch.nn.Linear(1 + 2, cont_emb[0])
        self.cont_emb_l2 = torch.nn.Linear(cont_emb[0], cont_emb[1])
        
        # "Continuous embedding NN for "not unique" features". cf #1 solution post
        self.cont_emb_notu_l = torch.nn.Linear(1 + 2, cont_emb_notu[0])
        self.cont_emb_notu_l2 = torch.nn.Linear(cont_emb_notu[0], cont_emb_notu[1])
        self.cont_emb_notu = cont_emb_notu[1]
            
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        
        # Embedding for "has one" categorical features, cf #1 solution post
        self.embeds = embedding(emb_szs[0][0], emb_szs[0][1])
        
        # At first we included information about the variable being processed (to extract feature importance). 
        # It works better using a constant feat (kind of intercept)
        self.embeds_feat = embedding(201, 2)
        self.embeds_feat_w = embedding(201, 2)
        
        self.emb_drop = nn.Dropout(emb_drop)
        
        n_emb = self.embeds.embedding_dim
        n_emb_feat = self.embeds_feat.embedding_dim
        n_emb_feat_w = self.embeds_feat_w.embedding_dim
        
        self.n_emb, self.n_emb_feat, self.n_emb_feat_w, self.n_cont,self.y_range = n_emb, n_emb_feat, n_emb_feat_w, n_cont, y_range
        
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True)] * (len(sizes)-2) + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
            
        self.layers = nn.Sequential(*layers)
        self.seq = nn.Sequential()
        
        # Input size for the NN that predicts weights
        inp_w = self.n_emb + self.n_emb_feat_w + self.cont_emb + self.cont_emb_notu
        # Input size for the final NN that predicts output
        inp_x = self.n_emb + self.cont_emb + self.cont_emb_notu
        
        # NN that predicts the weights
        self.weight = nn.Linear(inp_w, 5)
        self.weight2 = nn.Linear(5,1)
        
        mom = 0.1
        self.bn_cat = nn.BatchNorm1d(200, momentum=mom)
        self.bn_feat_emb = nn.BatchNorm1d(200, momentum=mom)
        self.bn_feat_w = nn.BatchNorm1d(200, momentum=mom)
        self.bn_raw = nn.BatchNorm1d(200, momentum=mom)
        self.bn_notu = nn.BatchNorm1d(200, momentum=mom)
        self.bn_w = nn.BatchNorm1d(inp_w, momentum=mom)
        self.bn = nn.BatchNorm1d(inp_x, momentum=mom)
        
    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.cont_emb_notu + self.cont_emb] + layers + [out_sz]

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        b_size = x_cont.size(0)
        
        # embedding of has one feat
        x = [self.embeds(x_cat[:,i]) for i in range(200)]
        x = torch.stack(x, dim=1)
        
        # embedding of intercept. It was embedding of feature id before
        x_feat_emb = self.embeds_feat(x_cat[:,200])
        x_feat_emb = torch.stack([x_feat_emb]*200, 1)
        x_feat_emb = self.bn_feat_emb(x_feat_emb)
        x_feat_w = self.embeds_feat_w(x_cat[:,200])
        x_feat_w = torch.stack([x_feat_w]*200, 1)
        
        # "continuous embedding" of raw features
        x_cont_raw = x_cont[:,:200].contiguous().view(-1, 1)
        x_cont_raw = torch.cat([x_cont_raw, x_feat_emb.view(-1, self.n_emb_feat)], 1)
        x_cont_raw = F.relu(self.cont_emb_l(x_cont_raw))
        x_cont_raw = self.cont_emb_l2(x_cont_raw)
        x_cont_raw = x_cont_raw.view(b_size, 200, self.cont_emb)
        
        # "continuous embedding" of not unique features
        x_cont_notu = x_cont[:,200:].contiguous().view(-1, 1)
        x_cont_notu = torch.cat([x_cont_notu, x_feat_emb.view(-1,self.n_emb_feat)], 1)
        x_cont_notu = F.relu(self.cont_emb_notu_l(x_cont_notu))
        x_cont_notu = self.cont_emb_notu_l2(x_cont_notu)
        x_cont_notu = x_cont_notu.view(b_size, 200, self.cont_emb_notu)

        x_cont_notu = self.bn_notu(x_cont_notu)
        x = self.bn_cat(x)
        x_cont_raw = self.bn_raw(x_cont_raw)

        x = self.emb_drop(x)
        x_cont_raw = self.emb_drop(x_cont_raw)
        x_cont_notu = self.emb_drop(x_cont_notu)
        x_feat_w = self.bn_feat_w(x_feat_w)
        
        # Predict a weight for each of the previous embeddings
        x_w = torch.cat([x.view(-1,self.n_emb),
                         x_feat_w.view(-1,self.n_emb_feat_w),
                         x_cont_raw.view(-1, self.cont_emb), 
                         x_cont_notu.view(-1, self.cont_emb_notu)], 1)

        x_w = self.bn_w(x_w)

        w = F.relu(self.weight(x_w))
        w = self.weight2(w).view(b_size, -1)
        w = torch.nn.functional.softmax(w, dim=-1).unsqueeze(-1)

        # weighted average of the differents embeddings using weights given by NN
        x = (w * x).sum(dim=1)
        x_cont_raw = (w * x_cont_raw).sum(dim=1)
        x_cont_notu = (w * x_cont_notu).sum(dim=1)
        
        # Use NN on the weighted average to predict final output
        x = torch.cat([x, x_cont_raw, x_cont_notu], 1) if self.n_emb != 0 else x_cont
        x = self.bn(x)
            
        x = self.seq(x)
        x = self.layers(x)
        return x
    
def set_seed(seed=42):
    # python RNG
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np
    np.random.seed(seed)
    
ss = StandardScaler()

logger.info('Input data')

data = pd.read_feather('../input/create-data/921_data.fth')
data = data.set_index('ID_code')

etd = pd.read_feather('../input/create-data/921_etd.fth')
etd = etd.set_index('ID_code')

has_one = [f'var_{i}_has_one' for i in range(200)]
orig = [f'var_{i}' for i in range(200)]
not_u = [f'var_{i}_not_unique' for i in range(200)]

cont_vars = orig + not_u
cat_vars = has_one
target = 'target'
path = './'

logger.info('cat treatment')

for f in cat_vars:
    data[f] = data[f].astype('category').cat.as_ordered()
    etd[f] = pd.Categorical(etd[f], categories=data[f].cat.categories, ordered=True)

# constant feature to replace feature index information
feat = ['intercept']
data['intercept'] = 1
data['intercept'] = data['intercept'].astype('category')
etd['intercept'] = 1
etd['intercept'] = etd['intercept'].astype('category')
    
cat_vars += feat

ref = pd.concat([data[cont_vars + cat_vars + ['target']], etd[cont_vars + cat_vars]])
ref[cont_vars] = ss.fit_transform(ref[cont_vars].values)

data = ref.iloc[:200000]
etd = ref.iloc[200000:]

data[target] = data[target].astype('int')

del ref; gc.collect()

fold_seed = 42
ss = StratifiedKFold(n_splits=10, random_state=fold_seed, shuffle=True)

folds = []
for num, (train,test) in enumerate(ss.split(data[target], data[target])):
    folds.append([train, test])


layers=[32]
ps=0.2
emb_drop=0.08
cont_emb=(50,10)
cont_emb_notu=(50,10)
emb_szs = [[6,12]]
use_bn = True
joined=False
# Code modified to sub with one seed
seeds = [42] #, 1337, 666]

results = []
sub_preds = pd.DataFrame(columns=range(10), index=etd.index)
for num_fold, (train, test) in enumerate(folds):
    procs=[]
    df = (TabularList.from_df(data, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs)
                .split_by_idx(test)
                .label_from_df(cols=target)
            .add_test(TabularList.from_df(etd, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs))
            .databunch(num_workers=0, bs=1024))
            
    df.dls[0].dl = df.dls[0].new(sampler=LongerRandomSampler(data_source=df.train_ds, mult=2), shuffle=False).dl
    for num_seed, seed in enumerate(seeds):
        logger.info(f'Model {num_fold} seed {num_seed}')
        set_seed(seed)
        model = my_TabularModel(emb_szs, len(df.cont_names), out_sz=df.c, layers=layers, ps=ps, emb_drop=emb_drop,
                                 y_range=None, use_bn=use_bn, cont_emb=cont_emb, cont_emb_notu=cont_emb_notu)

        learn = Learner(df, model, metrics=None, callback_fns=AUROC, wd=0.1)
        learn.fit_one_cycle(15, max_lr=1e-2, callbacks=[SaveModelCallback(learn, every='improvement', monitor='AUROC', name=f'fold{fold_seed}_{num_fold}_seed_{seed}'), AugShuffCallback(learn)])
        pred, _ = learn.get_preds()
        pred = pred[:,1]
        
        pred_test, _ = learn.get_preds(DatasetType.Test)
        pred_test = pred_test[:,1]
        
        sub_preds.loc[:, num_fold] = pred_test
        results.append(np.max(learn.recorder.metrics))
        logger.info('result ' + str(results[-1]))
        
        np.save(f'oof_fold{fold_seed}_{num_fold}_seed_{seed}.npy', pred)
        np.save(f'test_fold{fold_seed}_{num_fold}_seed_{seed}.npy', pred_test)
        
        del learn, pred, model, pred_test; gc.collect()
    del df; gc.collect()
print(results)
print(np.mean(results))

sub_preds[target] = sub_preds.rank().mean(axis=1)
sub_preds[[target]].to_csv('submission_NN_wo_pseudo_seed42.csv', index_label='ID_code')