
import numpy as np
import random
import pickle as pk
import numpy as np
import pandas as pd
import random
import sys
import codecs
import re
import string

import sklearn
from sklearn.manifold import TSNE

from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr, spearmanr
import math

import os
from os import listdir
from os.path import isfile, join

import json
import subprocess
from tqdm import tqdm_notebook as tqdm


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('colorblind')
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Verdana']
matplotlib.rcParams.update({'font.size': 12})
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


import Bio
from Bio import SeqIO
from Bio import AlignIO



import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, LSTM, Reshape, TimeDistributed
from tensorflow.keras.layers import  Concatenate, Softmax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input, GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Lambda, Add, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

def predpair():
    main_input = Input((None,5))

    cur_p1 = Conv1D(filters=64, kernel_size=10, padding='SAME')(main_input)
    cur_p2 = Conv1D(filters=64, kernel_size=10, padding='SAME')(main_input)
    cur1 = Attention()([cur_p1, main_input, cur_p2])

    cur = Concatenate()([cur1, main_input])
    cur = Bidirectional(LSTM(16, return_sequences=True))(cur)

    d = Dense(16, activation='relu')(cur)
    d = Dense(8, activation='relu') (d)
    d = TimeDistributed(Dense(1, activation='relu', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1))) (d)

    d = Reshape((1, -1))(d)

    main_output = Softmax()(d)
    model = Model(main_input, main_output)

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy', 'categorical_crossentropy'])    
    model.summary()
    
    return model

def is_good_seq(s):
    bases_list = ["A", "C", "G", "U"]
    for base in s:
        if base not in bases_list:
            return False    
    return True


def parse_data(dataset_file):
    seq2rf_id = {}
    rf_id2seq2data = {}
    with open(dataset_file) as f:
        f.readline()
        for line in f:
            s = line.strip().split("\t")
            rf_id = s[3].split("_")[0]
            if not rf_id in rf_id2seq2data:
                rf_id2seq2data[rf_id] = {}
            seq = s[0]
            if not is_good_seq(seq):
                continue
            seq2rf_id[seq] = rf_id
            if not seq in rf_id2seq2data[rf_id]:
                rf_id2seq2data[rf_id][seq] = []
            rf_id2seq2data[rf_id][seq].append({"seq":s[0], "mask":s[1], "ans":s[2], "full_id":s[3]})
    return rf_id2seq2data, seq2rf_id

def recode_data(s, mask, name):
    bases_dict     = {"A": 0, "C": 1, "G": 2, "U": 3}
    bases_list = ["A", "C", "G", "U"]

    seq_np = np.zeros((len(s), 5), dtype = np.float32) 
    for index in range(len(s)):
        base = s[index]
        channel = bases_dict[base]
        seq_np[index, channel] = 1.0

        if mask[index] == "1":
            seq_np[index, -1] = 2 ##############
        else:
            seq_np[index, -1] = 1
    return seq_np

def recode_ans(ans, l, name):
    answer = []
    ans_np = np.zeros(l, dtype = np.float32) 
    for index in range(l):
        if ans[index] == "1":
            ans_np[index] = 1
    ans_np = ans_np.reshape(ans_np.shape[0], 1)
    return ans_np

def get_data_batches(rf_set, rf_id2seq2data):
    items = []
    for rf in rf_set:
        for seq in rf_id2seq2data[rf]:
            items.append(rf_id2seq2data[rf][seq])
    return items

def recode_samples(in_set):
    set_q = [[recode_data(x["seq"], x["mask"], x["full_id"]) for x in batch] for batch in in_set]
    set_ans = [[recode_ans(x["ans"], len(x["seq"]), x["full_id"]) for x in batch] for batch in in_set]
    return set_q, set_ans

def prepare_data(rf_id2seq2data, train_file, val_file, test_file):
    
    train_rfs = pk.load(open(train_file, "rb"))
    val_rfs = pk.load(open(val_file, "rb"))
    test_rfs = pk.load(open(test_file, "rb"))
    
    train = get_data_batches(train_rfs, rf_id2seq2data)
    val = get_data_batches(val_rfs, rf_id2seq2data)
    test = get_data_batches(test_rfs, rf_id2seq2data)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    train_q, train_ans = recode_samples(train)
    val_q, val_ans = recode_samples(val)
    test_q, test_ans = recode_samples(test)
    
    return train_q, train_ans, val_q, val_ans, test_q, test_ans

def decode(qu):
    lets = []
    bases_list = ["A", "C", "G", "U"]
    for q in qu:
        for i in range(4):
            if q[i] == 1:
                lets.append(bases_list[i])
    assert len(lets) == len(qu)
    return lets

def count_pair_consistance(seq, model):
    anss = []
    batch = []
    for i in range(len(seq) + 1):
        if len(batch) == 32 or i == len(seq):
            batch = np.array(batch)
            pred = model.predict(batch.reshape(len(batch), inp.shape[0], 5), batch_size=32)
            anss.append(pred.reshape(len(batch), len(seq)))
            batch = []
        if i == len(seq):
            continue
        mask = "".join(["0" for j in range(i)] + ["1"] + ["0" for j in range(len(seq) - i - 1)])
        inp = recode_data(seq, mask, "")
        batch.append(inp)
    pairs = []
    for x in anss:
        for el in x:
            pairs.append(el)
    
    ind2max_value = {}
    for i in range(len(seq)):
        ind2max_value[i] = max(pairs[i])
    
    good_poss = []
    for i in range(len(seq)):
        for j in range(i):  
            if pairs[i][j] == ind2max_value[i] and pairs[j][i] == ind2max_value[j]:
                good_poss.append(i)
                good_poss.append(j)
    return sorted(list(set(good_poss)))

def get_paired_poss(data):
    paired = []
    for x in data:
        mask = x['mask']
        ans = x['ans']
        paired.append(mask.find("1"))
        paired.append(ans.find("1"))
    paired.sort()
    return paired

def calc_confusion_matrix(k2best_pairs_predicted, k2rfam_pairs, t):
    Tp, Fp, Fn = 0, 0, 0
    for k in k2rfam_pairs:
        if len(k2rfam_pairs[k]) == 0:
            continue
        pairs_pred = k2best_pairs_predicted[k]
        pairs_true = k2rfam_pairs[k]
        pairs_found = set()
        for p in pairs_pred:
            if p[2] <= t:
                continue
            if p[:2] in pairs_true:
                Tp += 1
                pairs_found.add(p[:2])
            else:
                Fp += 1
        for p in pairs_true:
            if not p in pairs_found:
                Fn += 1
    return Tp, Fp, Fn

def calc_precision_recall_tracks(k2best_pairs_predicted, k2rfam_pairs):
    pres, recs = [], []
    ps = []
    for k in k2best_pairs_predicted:
        for pair in k2best_pairs_predicted[k]:
            ps.append(pair[2])
    ps = random.sample(ps, 200)
    ps.sort()
    #print(len(ps))
    for t in tqdm(ps):
        Tp, Fp, Fn = calc_confusion_matrix(k2best_pairs_predicted, k2rfam_pairs, t)
        if Tp + Fp == 0:
            continue
        pr = Tp / (Tp + Fp)
        rec = Tp / (Tp + Fn)
        pres.append(pr)
        recs.append(rec)
    return pres, recs

def get_rfam_poss(pairs_rf):
    poss = set()
    for p in pairs_rf:
        poss.add(p[0])
        poss.add(p[1])
    return poss

def pair_acc(pairs_vi, pairs_rf, pos):
    true_ans = ""
    for pair in pairs_rf:
        if pos == pair[0]:
            true_ans = pair[1]
    for pair in pairs_vi:
        if pos == pair[0] and true_ans == pair[1]:
            return 1
    return 0    

def get_net_ans(seq):
    anss = []
    batch = []
    for i in range(len(seq) + 1):
        if len(batch) == 32 or i == len(seq):
            batch = np.array(batch)
            pred = model.predict(batch.reshape(len(batch), inp.shape[0], 5), batch_size=32)
            anss.append(pred.reshape(len(batch), len(seq)))
            batch = []
        if i == len(seq):
            continue
        mask = "".join(["0" for j in range(i)] + ["1"] + ["0" for j in range(len(seq) - i - 1)])
        inp = recode_data(seq, mask, "")
        batch.append(inp)
    res = []
    for x in anss:
        for el in x:
            res.append(el)
    return np.array(res)

def symmetrize_min(df):
    d2 = np.array(df)
    for i in range(len(d2)):
        for j in range(len(d2[i])):
            m = min([d2[i][j], d2[j][i]])
            d2[i][j] = m
            d2[j][i] = m
    return d2

def symmetrize_sum(df):
    d2 = np.array(df)
    for i in range(len(d2)):
        for j in range(len(d2[i])):
            m = (d2[i][j] + d2[j][i]) / 2
            d2[i][j] = m
            d2[j][i] = m
    return d2

def get_acc(d2, ans, q):
    #d2 = np.array(df)
    su = 0
    for i in range(len(q)):
        x = q[i]
        q_pos = get_q_pos(x)
        ans_pos = get_ans_pos(ans[i])
        if d2[q_pos][ans_pos] == max(d2[q_pos]):
            su += 1
    return su/len(q)



def get_pairs(seq):
    pairs = set()
    ans = symmetrize_min(get_net_ans(seq))
    for i in range(len(ans)):
        best_j, best_ans = 0, 0
        for j in range(len(ans)):
            if ans[i][j] > best_ans:
                best_ans = ans[i][j]
                best_j = j
        if ans[i][best_j] > 0:
            pairs.add((i, best_j, ans[i][best_j]))
            pairs.add((best_j, i, ans[i][best_j]))
    return pairs

def rand_seq(n):
    return "".join([random.choice("AUCG") for _ in range(n)])

def rand_mask(n):
    k = random.randint(0, n)
    ans = ""
    for i in range(n):
        if i == k:
            ans += "1"
        else:
            ans += "0"
    return ans
def count_pair_consistance_dms(seq, model):
    pairs = []
    for i in range(len(seq)):
        #if seq[i] == "A" or seq[i] == "C":
        mask = "".join(["0" for j in range(i)] + ["1"] + ["0" for j in range(len(seq) - i - 1)])
        inp = recode_data(seq, mask, "")
        pred = model.predict(inp.reshape(1, inp.shape[0], 5))
        pairs.append(pred.reshape(len(seq)))
    good_poss = []
    #th = -1.94 + -0.0015 * len(seq)
    for i in range(len(seq)):
        for j in range(i):  
            if pairs[i][j] == max(pairs[i]) and pairs[j][i] == max(pairs[j]):
                good_poss.append(i)
                good_poss.append(j)
    return sorted(list(set(good_poss)))

def plot_ans_and_dms(net_ans, qu, dms):
    seq = decode(qu)
    q_pos = get_q_pos(qu)
    fig, ax = plt.subplots(figsize=(len(qu)/50*12, 1))
    l = len(net_ans[0][0])
    plt.plot(list(range(l)), net_ans.reshape(l))
    x = np.arange(0, l, 1.0)
    plt.xticks(x)    
    ax.set_xticklabels(seq) 
    dms = [x*0.2 for x in dms]
    ax.stem(x, dms, linefmt='grey', markerfmt=' ', bottom=0, basefmt=' ')
    ax.arrow(q_pos, np.max(net_ans), 0, -np.max(net_ans)*0.9, head_width= 0.4, head_length=np.max(net_ans)/7, fc='k', ec='k')
    plt.show()

def get_struc_data(fname):
    data = np.zeros(GENOME_LENGTH) 
    with open(fname) as f:
        for _ in range(2):
            f.readline()
        for line in f:
            s = line.strip().split("\t")
            pos, val = int(s[0]), float(s[1])
            data[pos-1] = val
    return data

def get_gene_profile(gene, dms_for, dms_rev): 
    gene_dms = []
    if gene[2] == 1:
        gene_dms = dms_for[gene[0]:gene[1]]
    else:
        gene_dms = dms_rev[gene[0]:gene[1]][::-1] # не перевернуто
    return gene_dms

def gene_is_ok(gene_dms):
    avg = sum(gene_dms)/len(gene_dms)
    if avg > 15:
        return True
    return False

def add_noise(dms): #борьба с одинаковыми значениями
    dms_new = []
    for x in dms:
        dms_new.append(x + random.random()/10**2)
    return dms_new

def count_gene_quantile(gene_dms, pos):
    etalon = gene_dms[pos]
    bigger_count = 0
    for x in gene_dms:
        if x >= etalon:
            bigger_count += 1
    return bigger_count/len(gene_dms)

def get_seq(record, gene):
    s = record[gene[0]:gene[1]].seq

    if gene[2] == -1:
        s = s.reverse_complement()
    return str(s).replace("T", "U")

def get_ans_grad(seq, model):
    anss = []
    batch = []
    gradss = []
    for i in range(len(seq) + 1):
        if len(batch) == 32 or i == len(seq):
            batch = np.array(batch)
            inputs = tf.Variable(batch.reshape(len(batch), inp.shape[0], 5))
            pred = ""
            with tf.GradientTape() as tape:
                pred = model(inputs)#, batch_size=32)
            grads = tape.gradient(pred, inputs)
            anss.append(np.array(pred).reshape(len(batch), len(seq)))
            gradss.append(np.array(grads))
            batch = []
        if i == len(seq):
            continue
        mask = "".join(["0" for j in range(i)] + ["1"] + ["0" for j in range(len(seq) - i - 1)])
        inp = recode_data(seq, mask, "")
        batch.append(inp)
    
    res = []

    for x in anss:
        for el in x:
            res.append(el)
    
    importances = []

    for ex in gradss:
        for el in ex:
            importances.append(el)

    return np.array(res), np.array(importances)

def add_rectangles(df, mask, ax, lw=2.5):
    for i in range((len(df))):
        ans = df.iloc[i]
        for pos in range(len(ans)):
            if (i, pos) in mask:
                ax.add_patch(Rectangle((i, len(df)-pos-1), 1, 1, fill=False, edgecolor='black', lw=lw))
                
def get_importances(g):
    g_new = []
    for i in range(len(g)):
        tmp = []
        for j in range(len(g[i])):
            grads = g[i][j]
            imp = max([max(grads), -min(grads)])
            tmp.append(imp)
        g_new.append(tmp)
    return g_new

import string

def find_all_pairs(ss):
    open_symbols = "<([{"
    close_symbols = ">)]}"
    pk_open = string.ascii_uppercase
    pk_close = string.ascii_lowercase

    paired = {}
    for i in range(len(close_symbols)):
        paired[close_symbols[i]] = open_symbols[i]
    for i in range(len(pk_close)):
        paired[pk_close[i]] = pk_open[i]
    pairs = set()

    l2cos = {}
    for i, l in enumerate(ss):
        if l in open_symbols or l in pk_open:
            if not l in l2cos:
                l2cos[l] = []
            l2cos[l].append(i)
        if l in paired:
            j = l2cos[paired[l]].pop()
            pairs.add((j, i))
            pairs.add((i, j))
    return pairs

def get_prediction_plot(a, seq, ss=None):
    f, ax = plt.subplots(1, figsize=(5, 5))

    df = pd.DataFrame(data = a)
    
    df.sort_index(level=0, inplace=True, ascending=False)

    g = sns.heatmap(df, cmap="RdYlBu", annot=False, fmt='.2f',  
                    linewidths=0.01, ax=ax, cbar=True, 
                    cbar_kws = dict(use_gridspec=False,location="top"),
                                   xticklabels=seq, yticklabels=seq[::-1])
    g.set_xlabel('Query nucleotide')
    g.set_ylabel('Partner nucleotide')

    if ss != None:
        add_rectangles(df, ss, ax)
    plt.show()

def get_importance_plot(imp, seq, ss=None):
    f, ax = plt.subplots(1, figsize=(5, 5))
    df_imp = pd.DataFrame(data=imp)
    df_imp.sort_index(level=0, inplace=True, ascending=False)
    g2 = sns.heatmap(df_imp, cmap="Blues", annot=False, fmt='.2f', 
                    linewidths=0.01, ax=ax, cbar=True, 
                    cbar_kws = dict(use_gridspec=False,location="top"),
                                   xticklabels=seq, yticklabels=seq[::-1])
    g2.set_xlabel('Query nucleotide')
    g2.set_ylabel('Partner nucleotide')

    if ss != None:
        add_rectangles(df_imp, ss, ax, 1.5)

    plt.show()
    
def get_plot(a, imp, seq, ss=None):
    f, ax = plt.subplots(1, 2, figsize=(30, 11.5), dpi=340.0)

    df = pd.DataFrame(data = a)
    
    df.sort_index(level=0, inplace=True, ascending=False)

    g = sns.heatmap(df, cmap="RdYlBu", annot=False, fmt='.2f',  
                    linewidths=0.01, ax=ax[0], cbar=True, 
                    cbar_kws = dict(use_gridspec=False,location="right"),
                                   xticklabels=seq, yticklabels=seq[::-1])
    g.set_xlabel('\nQuery nucleotide', fontsize=23)
    g.set_ylabel('Partner nucleotide\n', fontsize=23)
    g.set_title('A\n', loc='left', fontsize=30)

    if ss != None:
        add_rectangles(df, ss, ax[0])

    df_imp = pd.DataFrame(data=imp)
    df_imp.sort_index(level=0, inplace=True, ascending=False)
    g2 = sns.heatmap(df_imp, cmap="Blues", annot=False, fmt='.2f', 
                    linewidths=0.01, ax=ax[1], cbar=True, 
                    cbar_kws = dict(use_gridspec=False,location="right"),
                                   xticklabels=seq, yticklabels=seq[::-1])
    g2.set_xlabel('\nFeature nucleotide', fontsize=23)
    g2.set_ylabel('Partner nucleotide\n', fontsize=23)
    g2.set_title('B\n', loc='left', fontsize=30)

    if ss != None:
        add_rectangles(df_imp, ss, ax[1], 1.5)

    plt.show()
    return f

def get_mean_qualities(c):
    seq = decode(test_q[c][0])
    #seq = gen_rand_seq(len(seq)) ##########################
    #print(seq2rf_id["".join(seq)], accs[c])
    ans, grad = get_net_ans(seq)
    ans = symmetrize_min(ans)
    imp = get_importances(grad)
    imp = symmetrize_min(imp)
    return np.mean(ans), np.mean(imp)

def get_pair_stat():
    imps_true, imps_rand = [], []
    ans_true, ans_rand = [], []

    for c in tqdm(range(len(test_q))):
        seq = decode(test_q[c][0])
        ans, grad = get_net_ans(seq)
        ans = symmetrize_min(ans)
        imp = get_importances(grad)
        imp = symmetrize_min(imp)
        m = get_mask_poss(test_q[c], test_ans[c])

        l = len(seq)
        for pair in m:
            i, j = pair
            imps_true.append(imp[i][j])
            ans_true.append(ans[i][j])

            r = random.randint(0, l-1)
            imps_rand.append(imp[i][r])
            ans_rand.append(ans[i][r])
    return ans_true, ans_rand, imps_true, imps_rand

def get_all_reciprocal_pairs(ans):
    pairs = []
    for i in range(len(ans)):
        j = np.argmax(ans[i]) 
        if np.argmax(ans[j]) == i:
            pairs.append((i, j))
    return pairs

def get_all_reciprocal_pairs_of_pairs(ans):
    pairs = []
    for i in range(len(ans)-1):
        j = np.argmax(ans[i])
        if j == 0:
            continue
        if j - 1 != np.argmax(ans[i+1]):
            continue
            
        if np.argmax(ans[j]) == i and np.argmax(ans[j-1]) == i+1:
            pairs.append(((i, i+1), (j-1, j)))
    return pairs

def get_lets_pair_stat():
    pair2count = {}
    ppair2count = {}
    for i, k in enumerate(k2seq):
        if i % 100 == 0:
            print(i)
        seq = k2seq[k]
        ans = get_net_ans(seq)
        ans = symmetrize_min(ans)
        pairs = get_all_reciprocal_pairs(ans)
        ppairs = get_all_reciprocal_pairs_of_pairs(ans)
        for p in pairs:
            ls = (seq[p[0]], seq[p[1]])
            if not ls in pair2count:
                pair2count[ls] = 0
            pair2count[ls] += 1
            
        for p in ppairs:
            ls = (seq[p[0][0]] + seq[p[0][1]], seq[p[1][0]] + seq[p[1][1]])
            if not ls in ppair2count:
                ppair2count[ls] = 0
            ppair2count[ls] += 1
    return pair2count, ppair2count

def count_paired_fraction(ss):
    count = sum([1 for l in ss if l in pairing_symbols])
    return count/len(ss)

def calc_fraction_distribution():
    frs = []
    for i in range(len(data_good)):
        fr = count_paired_fraction(data_good[i][-1])
        frs.append(fr)
    return frs

def filter_by_fr(threshold):
    alns_filtered, data_filtered = [], []
    for i in range(len(data_good)):
        fr = count_paired_fraction(data_good[i][-1])
        if fr >= threshold:
            alns_filtered.append(alns_good[i])
            data_filtered.append(data_good[i])
    print(len(alns_filtered))
    return alns_filtered, data_filtered

def filter_by_clan():
    data_filtered_new, alns_filtered_new = [], [] 
    clans_used = set()
    for i in range(len(data_filtered)):
        d = data_filtered[i]
        if len(d) == 3:
            if not d[1] in clans_used:
                clans_used.add(d[1])
                data_filtered_new.append(data_filtered[i])
                alns_filtered_new.append(alns_filtered[i])
        else:
            data_filtered_new.append(data_filtered[i])
            alns_filtered_new.append(alns_filtered[i]) 
    print(len(data_filtered_new))
    return data_filtered_new, alns_filtered_new

def delete_gaps(seq, ss):
    assert len(seq) == len(ss)
    seq_new, ss_new = "", ""
    for i in range(len(seq)):
        if seq[i] == "-":
            if ss[i] in pairing_symbols:
                #print("aaa!", seq, ss)
                return None, None
        else:
            seq_new += seq[i]
            ss_new += ss[i]
    return seq_new, ss_new
                
def gen_pre_dataset():
    seq_struc = []
    for i in range(len(alns_filtered)):
        aln = alns_filtered[i]
        d = data_filtered[i]
        records = [record.seq for record in aln]
        for rec in records:
            seq = str(rec)
            ss = d[-1]
            seq_new, ss_new = delete_gaps(seq, ss)
            if seq_new == None:
                continue
                
            if set(seq_new) != set("AUGC"):
                continue
            seq_struc.append([d[0], seq_new, ss_new])
    return seq_struc

def get_embedding(seq, model, intermediate_layer_model):
    anss = []
    batch = []
    seq = seq.replace('T', 'U')
    for i in range(len(seq) + 1):
        if len(batch) == 32 or i == len(seq):
            batch = np.array(batch)
            inputs = batch
            embs = intermediate_layer_model.predict(inputs)
            anss.append(np.array(embs))
            batch = []
        if i == len(seq):
            continue
        mask = "".join(["0" for j in range(i)] + ["1"] + ["0" for j in range(len(seq) - i - 1)])
        inp = recode_data(seq, mask, "")
        batch.append(inp)  
    return np.mean(np.concatenate(anss), axis=(0, 1))

def sample_data(cl2embs):
    x, y = [], []
    for cl in random.sample(list(cl2embs.keys()), len(list(cl2embs.keys()))):
        for p in cl2embs[cl]:
            x.append(p)
            y.append(cl)

    return np.array(x), np.array(y)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
