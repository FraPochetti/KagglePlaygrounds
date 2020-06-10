import pandas as pd
import numpy as np
import boto3
import s3fs
import json
import numpy as np
import os, time, shutil
import pickle as cpk
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import itertools
from scipy.stats import randint, uniform
from IPython.core.display import display, HTML
import pprint

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, make_scorer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon, image, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
try:
    import gluoncv as cv
except: 
    pass

#########################################################
# GENERAL FUNCTIONS
#########################################################

def missing_values(data):
    total = data.isnull().sum().sort_values(ascending = False) 
    percent = (data.isnull().sum() / data.isnull().count() * 100 ).sort_values(ascending = False) 
    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 
    return df[~(df['Total'] == 0)] 

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df.T)
        
def save_artifact(o, path):
    with open(path, "wb") as output_file:
        cpk.dump(o, output_file)
        
def load_artifact(path):
    with open(path, "rb") as input_file:
        o = cpk.load(input_file)
    return o

def split_data(df, n=3):
    X_train, X_valid, y_train, y_valid = train_test_split(df, df.adoptionspeed, test_size=0.33, random_state=42, stratify=df.adoptionspeed)

    speedy = X_train.loc[X_train.adoptionspeed==0]
    for _ in range(n): 
        X_train = X_train.append(speedy)
        y_train = y_train.append(speedy.adoptionspeed)  
        
    return X_train, X_valid, y_train, y_valid

#########################################################
# REPORTING MODEL PERFORMANCE
#########################################################

def kappa(y_true, y_pred): return cohen_kappa_score(y_true, y_pred, weights='quadratic')
def accuracy(y_true, y_pred): return accuracy_score(y_true, y_pred)

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix validation set',
                          cmap=plt.cm.Blues):
    """
    plot_confusion_matrix prints and plots the cm 
    confusion matrix received in input.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.grid(b=None)
    
def report(m, X, y_valid, preds=None, classes=[0, 1, 2, 3, 4], conf=True):
    p = m.predict(X) if preds is None else preds
    cm = confusion_matrix(y_valid, p)
    print(f"Accuracy: {np.round(accuracy(y_valid, p),5)}\nCohen's Kappa: {np.round(kappa(y_valid, p),5)}\nKaggle winning Cohen's Kappa: 0.45338\n")
    if conf: plot_confusion_matrix(cm, classes)
        
def read_catboost():
    with open("./catboost_info/catboost_training.json") as json_file:
        j = json.load(json_file)
    p = pd.DataFrame(j['iterations'])
    p1 = p.test.apply(pd.Series).rename(columns={0:'loss_valid', 1:'accuracy_valid'})
    p2 = p.learn.apply(pd.Series).rename(columns={0:'loss_train', 1:'accuracy_train'})
    p.drop(['test', 'learn'], axis=1, inplace=True)
    return p.join(p1).join(p2)

def get_feat_importance(m, df, html=False):
    fi = pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
    if html:
        display(HTML('<h3>Top 10 features</h3>'))
        display(fi.head(10))

        display(HTML('<h3>Bottom 10 features</h3>'))
        display(fi.tail(10))
    return fi

def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(9,int(0.3*len(fi))), legend=False, color='cornflowerblue')

#########################################################
# FIT ONE CYCLE
#########################################################

def annealing_linear(start, end, pct):
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start + pct * (end-start)

def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out

def plot_one_cycle_schedule(iterations, lr_schedule, mom_schedule):
    fig, ax = plt.subplots(1,2,figsize=(14, 4))
    ax[0].plot(iterations, lr_schedule)
    ax[0].set(xlabel="Iterations (epochs * train batches)", ylabel="Learning Rate", title="Learning Rate Schedule")
    ax[1].plot(iterations, mom_schedule)
    ax[1].set(xlabel="Iterations (epochs * train batches)", ylabel="Momentum",  title="Momentum Schedule")
    plt.show()

def calc_lr_mom_schedule(tot_epochs, len_train, lr_max, div_factor=25., pct_start=0.3, moms=(0.95, 0.85)):
    n = len_train * tot_epochs
    iterations = 1+np.arange(n)
    final_div = div_factor*1e4
    a1 = int(n * pct_start)
    a2 = n-a1

    low_lr = lr_max/div_factor
    final_lr = lr_max/final_div

    linear_lr = []
    linear_mom = []
    for i in iterations[:a1]:
        pct=i/a1
        lr = annealing_linear(low_lr, lr_max, pct)
        mom = annealing_linear(moms[0], moms[1], pct)
        linear_lr.append(lr)    
        linear_mom.append(mom)

    cos_lr = []
    cos_mom = []
    for i in iterations[a1:]:
        pct=(i-a1-1)/(a2-1)
        lr = annealing_cos(lr_max, final_lr, pct)
        mom = annealing_cos(moms[1], moms[0], pct)
        cos_lr.append(lr)
        cos_mom.append(mom)

    lr_schedule = np.array(linear_lr+cos_lr)
    mom_schedule = np.array(linear_mom+cos_mom)

    return iterations, lr_schedule, mom_schedule

#########################################################
# LR FINDER
#########################################################

def plot_lr(df_lrs, min_grad_lr=None, min_grad_loss=None, skip_start=10, skip_end=2):
    lrs = df_lrs[skip_start:-skip_end]
    ax = lrs.plot(x='lr', y='loss', logx=True)
    ax.set(xlabel="Learning Rate (log scale)", ylabel="Loss")
    if min_grad_lr is not None and min_grad_loss is not None: 
        print(f"Min numerical gradient @lr: {min_grad_lr:.2e}")
        ax.plot(min_grad_lr, min_grad_loss, markersize=10,marker='o',color='red')
    plt.show()  
    
def find_lr(net, dl, ctx, params, L=gluon.loss.SoftmaxCrossEntropyLoss(), app="cv", plot=True, init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(dl)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    trainer = gluon.Trainer(params, 'adam', {'learning_rate': lr})
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    lrs = []
       
    for i, batch in enumerate(dl):
        batch_num += 1
        X, y = get_batch(batch, ctx, app)
        
        with ag.record():
            outputs = forward_pass(net, X, app)
            loss = L(outputs, y).mean()

        l = loss.asscalar()
        avg_loss = beta * avg_loss + (1-beta) * l
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss: break
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1: best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        lrs.append(lr)
        
        loss.backward()
        trainer.step(1)
        
        lr *= mult
        trainer.set_learning_rate(lr)
                                  
    df_lrs = pd.DataFrame({"lr": lrs, "loss": losses})
    df = df_lrs[10:-5]
    losses = df.loss.values
    mg = np.gradient(losses).argmin()
    min_grad_loss = losses[mg]
    min_grad_lr = df.lr.values[mg]
    if plot: plot_lr(df_lrs, min_grad_lr=None, min_grad_loss=None)

    return df_lrs, min_grad_lr, min_grad_loss

#########################################################
# TABULAR
#########################################################
    
def categorify(df, catf, d=None):
    if d is None:
        d = {}
        for c in catf:
            df[c] = df[c].astype('category') #.as_ordered()
            d[c] = df[c].cat.categories
            df[c] = df[c].cat.codes+1
    else:
        for c in catf:
            df[c] = pd.Categorical(df[c], categories=d[c])
            df[c] = df[c].cat.codes+1
            df[c] = df[c].fillna(0)

    return df, d

def get_numf_scaler(train): return StandardScaler().fit(train)

def scale_numf(df, numf, scaler):
    cols = numf
    index = df.index
    scaled = scaler.transform(df[numf])
    scaled = pd.DataFrame(scaled, columns=cols, index=index)
    return pd.concat([scaled, df.drop(numf, axis=1)], axis=1)

def apply_cats(df, trn):
    for c in trn.columns:
        df[c] = pd.Categorical(df[c], categories=trn[c].cat.categories)

def split_cats(df, catf): return catf, [col for col in df.columns if col not in catf]

def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, int(round(1.6 * n_cat**0.56)))

def tuplify(X, y, catf, numf): return [((a, b), c) for a, b, c in zip(X[catf].values.tolist(), X[numf].values.tolist(), y.values.tolist())] 

def get_tab_ds(X_t, y_t, X_v, y_v, catf, numf): 
    scaler = get_numf_scaler(X_t[numf])
    X_t = scale_numf(X_t, numf, scaler)
    X_v = scale_numf(X_v, numf, scaler)
    
    return gluon.data.SimpleDataset(tuplify(X_t, y_t, catf, numf)), gluon.data.SimpleDataset(tuplify(X_v, y_v, catf, numf))

def get_tab_dl(t_ds, v_ds, bs=256): return gluon.data.DataLoader(t_ds, batch_size=bs, shuffle=True), gluon.data.DataLoader(v_ds, batch_size=bs)

class TabularModel(gluon.HybridBlock):
    def __init__(self, emb_szs, n_cont, szs, drops, emb_drop=0., use_bn=True, prefix=None, params=None):
        super(TabularModel, self).__init__(prefix=prefix, params=params)    
        with self.name_scope():
            self.embs = gluon.nn.HybridSequential()
            with self.embs.name_scope():
                [self.embs.add(gluon.nn.Embedding(c, s)) for c,s in emb_szs]
                    
            self.n_emb, self.n_cont = sum([e[1] for e in emb_szs]), n_cont
            szs = [self.n_emb+n_cont] + szs
            self.lins = gluon.nn.HybridSequential()
            with self.lins.name_scope():
                [self.lins.add(gluon.nn.Dense(i, activation='relu')) for i in szs[1:]]
            
            self.bns = gluon.nn.HybridSequential()
            with self.bns.name_scope():
                [self.bns.add(gluon.nn.BatchNorm(axis=0)) for i in szs[1:]]

            self.drops = gluon.nn.HybridSequential()
            with self.drops.name_scope():
                [self.drops.add(gluon.nn.Dropout(p)) for p in drops]
            
            self.emb_drop = gluon.nn.Dropout(emb_drop)
            self.bn = gluon.nn.BatchNorm(axis=0)
            self.use_bn=use_bn
            self.outp = gluon.nn.Dense(5)
        
    def hybrid_forward(self, F, x_cat, x_cont):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embs)]
        x = nd.concat(*x, dim=1)
        x = self.emb_drop(x)
        x2 = self.bn(x_cont)
        x = nd.concat(x, x2, dim=1)
        for l,d,b in zip(self.lins, self.drops, self.bns):
            x = l(x)
            if self.use_bn: x = b(x)
            x = d(x)
        out = self.outp(x) 
        return out
    
def get_tabular_model(ctx, emb_szs, n_cont, szs=[500,250,125], drops=[0.001,0.01,0.01], emb_drop=0., use_bn=True):
    net = TabularModel(emb_szs=emb_szs, szs=szs, drops=drops, n_cont=n_cont, emb_drop=emb_drop, use_bn=use_bn)
    net.initialize(mx.init.Xavier(), ctx=ctx)
    return net
    
#########################################################
# COMPUTER VISION
#########################################################
    
def get_cnn(which_net, ctx, classes):
    if which_net == 'resnet18':
        pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
        finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=classes)
    if which_net == 'resnet34':
        pretrained_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
        finetune_net = gluon.model_zoo.vision.resnet34_v2(classes=classes)
    if which_net == 'resnet50':
        pretrained_net = gluon.model_zoo.vision.resnet50_v2(pretrained=True)
        finetune_net = gluon.model_zoo.vision.resnet50_v2(classes=classes)
    
    finetune_net.features = pretrained_net.features
    finetune_net.output.initialize(mx.init.Xavier(), ctx=ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()
    return finetune_net

#########################################################
# NATURAL LANGUAGE PROCESSING
#########################################################

class MeanPoolingLayer(gluon.HybridBlock):
    """A block for mean pooling of encoder features"""
    def __init__(self, prefix=None, params=None):
        super(MeanPoolingLayer, self).__init__(prefix=prefix, params=params)
        
    def hybrid_forward(self, F, data, valid_length): # pylint: disable=arguments-differ
        """Forward logic"""
        # Data will have shape (T, N, C)
        masked_encoded = F.SequenceMask(data, sequence_length=valid_length,use_sequence_length=True)
        agg_state = F.broadcast_div(F.sum(masked_encoded, axis=0), F.expand_dims(valid_length, axis=1))
        return agg_state
    
class BaseNet(gluon.HybridBlock):
    """Network for sentiment analysis."""
    def __init__(self, dropout, classes, prefix=None, params=None):
        super(BaseNet, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = None # will set with lm embedding later
            self.encoder = None # will set with lm encoder later
            self.agg_layer = MeanPoolingLayer()
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(dropout))
                self.output.add(gluon.nn.Dense(classes, flatten=False))

    def hybrid_forward(self, F, data, valid_length): # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(data))  # Shape(T, N, C)
        agg_state = self.agg_layer(encoded, valid_length)
        out = self.output(agg_state)
        return out
    
def get_rnn(dropout, classes, ctx):
    lm_model, _ = get_lm(dropout, ctx)
    net = BaseNet(dropout=dropout, classes=classes)
    net.embedding = lm_model.embedding
    net.encoder = lm_model.encoder
    net.hybridize()
    net.output.initialize(mx.init.Xavier(), ctx=ctx)
    return net

def get_lm(dropout, ctx, language_model_name='standard_lstm_lm_200', pretrained=True):
    return nlp.model.get_model(name=language_model_name,
                               dataset_name='wikitext-2',
                               pretrained=pretrained,
                               ctx=ctx,
                               dropout=dropout)

# Helper function to preprocess a single data point
def preprocess(x, vocab, length_clip, tokenizer):
    data, label = x
    label = int(label)
    # A token index or a list of token indices is
    # returned according to the vocabulary.
    data = vocab[length_clip(tokenizer(data))]
    return data, label

# Helper function for getting the length
def get_length(x):
    return float(len(x[0]))

def preprocess_dataset(dataset, vocab, length_clip, tokenizer):
    with mp.Pool() as pool:
        # Each sample is processed in an asynchronous manner.
        dataset = gluon.data.SimpleDataset(pool.map(partial(preprocess, vocab=vocab, length_clip=length_clip, tokenizer=tokenizer), dataset))
        lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    return dataset, lengths

def get_nlp_dls(tr, val, vocab, length_clip, tokenizer, batch_size=256):
    train_dataset, train_data_lengths = preprocess_dataset(tr, vocab, length_clip, tokenizer)
    valid_dataset, valid_data_lengths = preprocess_dataset(val, vocab, length_clip, tokenizer)
    return get_nlp_dataloader(train_dataset, train_data_lengths, valid_dataset, batch_size=batch_size)

def get_nlp_dataloader(train_dataset, train_data_lengths, valid_dataset, batch_size=256, bucket_num=10, bucket_ratio=0.2):
    # Pad data, stack label and lengths
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=0, ret_length=True),
        nlp.data.batchify.Stack(dtype='float32'))
    
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        train_data_lengths,
        batch_size=batch_size,
        num_buckets=bucket_num,
        ratio=bucket_ratio,
        shuffle=True)

    # Construct a DataLoader object for both the training and test data
    train_dataloader = gluon.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)
    
    valid_dataloader = gluon.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        batchify_fn=batchify_fn)
    
    return train_dataloader, valid_dataloader

#########################################################
# MODEL TRAINING
#########################################################

def predict(net, 
            dl, 
            ctx, 
            app="cv"):
    
    outs = []
    for i, batch in enumerate(dl):
        X, y = get_batch(batch, ctx, app)
        outputs = forward_pass(net, X, app)
        outs.append(outputs)
        
    return nd.concat(*outs, dim=0).argmax(axis=1).asnumpy().astype(int)

def format_time(temp):
    hours = temp//3600
    temp = temp - 3600*hours
    minutes = temp//60
    seconds = temp - 60*minutes
    return f"{int(minutes)}m:{int(seconds)}s"

def get_batch(batch, 
              ctx, 
              app):
    if app=="cv": 
        X, y = batch
        return X.as_in_context(ctx), y.as_in_context(ctx)
    if app=="nlp":
        (X, leng), y = batch
        return (X.as_in_context(ctx), leng.as_in_context(ctx).astype(np.float32)), y.as_in_context(ctx)
    if app=="tab":
        (x_cat, x_cont), y = batch
        return (x_cat.as_in_context(ctx), x_cont.as_in_context(ctx).astype(np.float32)), y.as_in_context(ctx)       
    
def forward_pass(net, X, app):
    if app=="cv":
        return net(X)
    if app=="nlp":
        return net(X[0].T, X[1])
    if app=="tab":
        return net(X[0], X[1])
    
def evaluate(net, 
             valid_iter, 
             ctx, 
             L=gluon.loss.SoftmaxCrossEntropyLoss(),
             app="cv"):
    num_batch = len(valid_iter)
    valid_loss = 0
    metric = mx.metric.Accuracy()
    
    for i, batch in enumerate(valid_iter):
        X, y = get_batch(batch, ctx, app)
        outputs = forward_pass(net, X, app)
        valid_loss += L(outputs, y).mean().asscalar()
        metric.update(y, outputs)

    return metric.get()[1], valid_loss/num_batch
    
def train(train_iter, 
          valid_iter, 
          net, 
          trainer,
          ctx,
          epochs,
          L=gluon.loss.SoftmaxCrossEntropyLoss(), 
          app="cv", 
          hyper=None):
    
    num_batch = len(train_iter)
    metric = mx.metric.Accuracy()
    
    iteration = 0 
    for epoch in range(epochs):
        tic = time.time()
        train_loss = 0
        metric.reset()

        for i, batch in enumerate(train_iter):
            if hyper is not None:
                trainer.optimizer.beta1 = hyper['mom_schedule'][iteration]
                trainer.set_learning_rate(hyper['lr_schedule'][iteration])
                iteration+=1
            
            X, y = get_batch(batch, ctx, app)
            
            with ag.record():
                outputs = forward_pass(net, X, app)
                loss = L(outputs, y).mean()
            
            loss.backward()
            trainer.step(1)
            train_loss += loss.asscalar()
            metric.update(y, outputs)

        _, train_acc = metric.get()
        train_loss /= num_batch

        valid_acc, valid_loss = evaluate(net, valid_iter, ctx, L, app)
        print(f'[Epoch {epoch}] Train-acc: {np.round(train_acc, 3)}, loss: {np.round(train_loss, 3)} | Val-acc: {np.round(valid_acc, 3)}, loss: {np.round(valid_loss, 3)} | time: {format_time(time.time() - tic)}')
    
    return net

def fine_tune(dls, 
              net, 
              lr, 
              ctx, 
              L=gluon.loss.SoftmaxCrossEntropyLoss(),
              freeze=False, 
              epochs=5, 
              app="cv",
              hyper=None):
    
    train_iter, valid_iter = dls
    params = net.output.collect_params() if freeze else net.collect_params()
    trainer = gluon.Trainer(params, mx.optimizer.Adam(learning_rate=lr))
    
    return train(train_iter, 
                 valid_iter, 
                 net, 
                 trainer,
                 ctx,
                 epochs,
                 L=L, 
                 app=app, 
                 hyper=hyper)