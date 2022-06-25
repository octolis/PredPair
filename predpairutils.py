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
    bases_dict = {"A": 0, "C": 1, "G": 2, "U": 3}
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

def calculate_freqs(test_q, seq2rf_id, rf_id2seq2data, predpair):
    
    paired_good_frs, unpaired_good_frs = [], [] 
    rf_used = set()

    for x in tqdm(test_q):
        
        seq = decode(x[0])
        seq = "".join(seq)
        rf = seq2rf_id[seq]
        if rf in rf_used:
            continue
        rf_used.add(rf)
        data = rf_id2seq2data[rf][seq]
        good_poss = count_pair_consistance(seq, predpair)
        paired_poss = get_paired_poss(data)
        unpaired_poss = [i for i in range(len(seq)) if not i in paired_poss]

        fr_paired = len(set(paired_poss) & set(good_poss))/len(paired_poss)
        fr_unpaired = len(set(unpaired_poss) & set(good_poss))/len(unpaired_poss)

        paired_good_frs.append(fr_paired)
        unpaired_good_frs.append(fr_unpaired)
        
    return paired_good_frs, unpaired_good_frs
    
def plot_reciprocity(paired_good_frs, unpaired_good_frs):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=192, gridspec_kw={'width_ratios': [2, 1.5]}) 
    ax1.hist([paired_good_frs, unpaired_good_frs], density=True, bins=10)
    ax1.legend(["Paired", "Not paired"])
    ax1.set_xlabel("Fraction of reciprocal nucleotide pairs")
    ax1.set_ylabel("Density")
    ax1.set_title('A\n', loc='left')

    ax2.plot(paired_good_frs, unpaired_good_frs, "o")
    ax2.plot([0,1], [0,1])
    ax2.set_xlabel("Fraction of reciprocal among paired")
    ax2.set_ylabel("Fraction of reciprocal among not paired")
    ax2.set_xlim((0, 1))
    ax2.set_ylim((0, 1))
    ax2.set_title('B\n', loc='left')
    plt.show()
    
def gen_data_for_pres_recall(rf_id2seq2data, path_for_seqs):
    seq = ""
    k = 0
    k2data = {}
    seq2k = {}
    path_for_seqs = "test_seqs/"
    script = open("script_for_rnaplfold.sh", "w")
    for r in tqdm(rf_id2seq2data):
        for s in rf_id2seq2data[r]:
            seq = s
            seq2k[seq] = k
            l = str(len(seq) + 1)
            with open(path_for_seqs + str(k) + ".fa", "w") as w:
                w.write(">" + str(k) + "\n")
                w.write(s)
            script.write("RNAplfold -o -W " + l + " -L " + l + "  −−cutoff=0.0  <" + path_for_seqs + str(k) + ".fa\n")
            k2data[k] = rf_id2seq2data[r][s]
            k += 1
    script.close()
    
    return k2data #, seq2k

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

def get_k2pairs_pred(mypath):

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    k2pairs_predicted = {}

    for f in onlyfiles:
        k = int(f.split("_")[0])
        k2pairs_predicted[k] = []
        with open(join(mypath, f)) as f:
            for line in f:
                s = line.strip().split()
                b, e = int(s[0])-1, int(s[1])-1
                p = float(s[2])
                k2pairs_predicted[k].append([b, e, p])
    return k2pairs_predicted

def get_plfold_pairs(k2pairs_predicted, k2data): 
    
    k2best_pairs_predicted = {}
    k2rfam_pairs = {}

    for k in k2pairs_predicted:
        k2best_pairs_predicted[k] = set()
        l = len(k2data[k][0]["seq"])
        best_pair = [[] for _ in range(l)]
        for pair in k2pairs_predicted[k]:
            b, e, p = pair
            if len(best_pair[b]) == 0 or best_pair[b][2] < p:
                s = best_pair[b][3] if len(best_pair[b]) > 0 else 0
                best_pair[b] = (b, e, p, p + s)
            if len(best_pair[e]) == 0 or best_pair[e][2] < p:
                s = best_pair[e][3] if len(best_pair[e]) > 0 else 0
                best_pair[e] = (e, b, p, p + s)
        for i in range(l):
            if len(best_pair[i]) > 0: #and best_pair[i][2] > 1 - best_pair[i][3]:
                k2best_pairs_predicted[k].add(best_pair[i][:3])
    rf_used = set()
    for k in k2best_pairs_predicted:
        k2rfam_pairs[k] = set()
        data = k2data[k]
        #print(data[0])
        rf = data[0]["full_id"]
        if rf in rf_used: #################
            continue
        rf_used.add(rf)
        for s in data:
            b = s["mask"].find("1")
            e = s["ans"].find("1")
            k2rfam_pairs[k].add((b, e))
            k2rfam_pairs[k].add((e, b))
        
    return k2best_pairs_predicted, k2rfam_pairs

def get_net_pairs(k2rfam_pairs, k2data, model):
    good_ks = [k for k in k2rfam_pairs if len(k2rfam_pairs[k]) > 0]
    k2net_best_pairs_predicted = {}

    for k in tqdm(good_ks):
        seq = k2data[k][0]['seq']
        pairs = get_pairs(seq, model)
        k2net_best_pairs_predicted[k] = pairs
        
    return k2net_best_pairs_predicted

def plot_prec_recall(recs, pres, recs_net, pres_net):
    f = plt.figure(figsize=(5,5), dpi = 340.0)

    plt.plot(recs, pres)
    plt.plot(recs_net, pres_net)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(["RNAplfold", "PredPair"])
    plt.show()

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

def get_net_ans(seq, model):
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

def gen_rand_dms():
    rand_qs = [recode_data(rand_seq(100), rand_mask(100), "rand") for _ in range(10)]
    rand_ans = [recode_ans(rand_mask(100), 100, "rand") for _ in range(10)]
    return rand_qs, rand_ans

def get_dms_genes(record, dms_for, dms_rev):
    all_genes = []
    good_genes = []
    used_coos = set()

    for feature in record.features:
        if not (feature.type == "gene" or feature.type == "CDS"):
            continue
        start = feature.location.start
        end = feature.location.end
        strand = feature.location.strand

        gene = feature.qualifiers["gene"][0]
        coos = str(start) + "-" + str(end)

        if not coos in used_coos:
            all_genes.append([int(start), int(end), strand, gene])
            used_coos.add(coos)

        if not feature.type == "CDS":
            continue

        if not (end-start) % 3 == 0:
            continue #pseudogenes
            
    for gene in all_genes:
        if gene[2] == 1:
            if gene[1] - gene[0] > 1000:
                continue
            prof = get_gene_profile(gene, dms_for, dms_rev)
            if gene_is_ok(prof):
                good_genes.append(gene)
    return good_genes

def get_dms_quantiles(good_genes, dms_for, dms_rev, record, model):
    qs = []
    rand_qs = []
    for j, gene in enumerate(good_genes):
        if j%20 == 0:
            print(j)
        if gene[1] - gene[0] > 500:
            continue
        seq = str(record[gene[0]:gene[1]].seq).replace("T", "U")
        dms = get_gene_profile(gene, dms_for, dms_rev)
        good_poss = count_pair_consistance_dms(seq, model)
        rand_poss = random.sample(list(range(len(seq))), len(good_poss))

        for pos in good_poss:
            q = count_gene_quantile(dms, pos)
            qs.append(q)
            #print(q)

        for pos2 in rand_poss:
            q2 = count_gene_quantile(dms, pos2)
            rand_qs.append(q2)
            
    return qs, rand_qs

def gen_mrnas_for_vienna(good_genes, record):
    path_for_seqs = "mrna_seqs/"
    script = open("mrna_script_for_rnaplfold", "w")
    mrna_k2data = {}
    k = 0
    for item in good_genes:

        seq = str(record.seq[item[0]:item[1]])
        mrna_k2data[k] = item
        l = str(len(seq) + 1)
        with open(path_for_seqs + str(k) + ".fa", "w") as w:
            w.write(">" + str(k) + "\n")
            w.write(seq)
        script.write("RNAplfold -o -W " + l + " -L " + l + "  −−cutoff=0.0  <" + path_for_seqs + str(k) + ".fa\n")
        k += 1
    script.close()
    
    return mrna_k2data

def get_mrna_pairs_vienna(k2pairs_predicted, k2data):
    k2best_pairs_predicted = {}

    for k in k2pairs_predicted:
        k2best_pairs_predicted[k] = set()
        l = k2data[k][1] - k2data[k][0]
        best_pair = [[] for _ in range(l)]
        for pair in k2pairs_predicted[k]:
            b, e, p = pair
            if len(best_pair[b]) == 0 or best_pair[b][2] < p:
                s = best_pair[b][3] if len(best_pair[b]) > 0 else 0
                best_pair[b] = (b, e, p, p + s)
            if len(best_pair[e]) == 0 or best_pair[e][2] < p:
                s = best_pair[e][3] if len(best_pair[e]) > 0 else 0
                best_pair[e] = (e, b, p, p + s)
        for i in range(l):
            if len(best_pair[i]) > 0: #and best_pair[i][2] > 1 - best_pair[i][3]:
                k2best_pairs_predicted[k].add(best_pair[i][:2])
    return k2best_pairs_predicted

def get_dms_quantiles_vienna(good_genes, record, dms_for, dms_rev, k2data, k2best_pairs_predicted):
    vienna_qs = []
    v_rand_qs = []
    for j, gene in enumerate(good_genes):
        if j%100 == 0:
            print(j)
        if gene[1] - gene[0] > 500:
            continue
        seq = str(record[gene[0]:gene[1]].seq).replace("T", "U")
        dms = get_gene_profile(gene, dms_for, dms_rev)

        index = [k for k,v in k2data.items() if v == gene][0]
        good_poss = list(set(sum(k2best_pairs_predicted[index],())))
        rand_poss = random.sample(list(range(len(seq))), len(good_poss))

        for pos in good_poss:
            q = count_gene_quantile(dms, pos)
            vienna_qs.append(q)
            #print(q)

        for pos2 in rand_poss:
            q2 = count_gene_quantile(dms, pos2)
            v_rand_qs.append(q2)
    return vienna_qs, v_rand_qs

def plot_dms(qs, rand_qs):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi = 340.0) 

    ax1.hist(qs, bins=10)
    f.text(0.5,0.04, "Quantile of the position in DMS-seq data", ha="center", va="center")
    ax1.set_ylabel("Density\n")
    ax1.set_title('A', loc='left')
    ax2.hist(rand_qs, bins=10)
    ax2.set_title('B', loc='left')

    plt.show()

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



def get_pairs(seq, model):
    pairs = set()
    ans = symmetrize_min(get_net_ans(seq, model))
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

def get_struc_data(fname, GENOME_LENGTH):
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


def plot_duf2693fd(seq, rfam_pic, known_structure, data, true_pairs): 
    fig = plt.figure(figsize=(21, 8.5), dpi=340)
    gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0, hspace=0)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title('A\n', loc='left')
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    ax2.set_title('B\n', loc='left')
    ax3 = fig.add_subplot(gs[:, 1])

    rfam_structure = mpimg.imread(rfam_pic)
    ax1.imshow(rfam_structure)

    rnafold_structure = mpimg.imread(known_structure)
    ax2.imshow(rnafold_structure)

    df = pd.DataFrame(data = data)
    df.sort_index(level=0, inplace=True, ascending=False)

    ax3 = sns.heatmap(df, cmap="RdYlBu", annot=False, fmt='.2f',  
                    linewidths=0.01, ax=ax3, cbar=True, 
                    cbar_kws = dict(use_gridspec=False,location="right"),
                                   xticklabels=seq, yticklabels=seq[::-1])
    ax3.set_xlabel('\nQuery nucleotide')
    ax3.set_ylabel('Partner nucleotide\n')
    ax3.set_title('C\n', loc='left')

    for i in range((len(df))):
            ans = df.iloc[i]
            for pos in range(len(ans)):
                if (i, pos) in true_pairs:
                    ax3.add_patch(Rectangle((i, len(df)-pos-1), 1, 1, fill=False, edgecolor='black', lw=2.5))

    plt.show()
    
def plot_pk(seq_pk, rfam_pic, a_pk, true_pairs_pk):
    fig = plt.figure(figsize=(13.5, 5), dpi=340)

    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.1, hspace=0)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title('A\n', loc='left')

    rfam_structure = mpimg.imread(rfam_pic)
    ax1.imshow(rfam_structure)

    df = pd.DataFrame(data = a_pk)

    df.sort_index(level=0, inplace=True, ascending=False)

    ax3 = fig.add_subplot(gs[0, 1])
    ax3 = sns.heatmap(df, cmap="RdYlBu", annot=False, fmt='.2f',  
                    linewidths=0.01, ax=ax3, cbar=True, 
                    cbar_kws = dict(use_gridspec=False, location="right"),
                                   xticklabels=seq_pk, yticklabels=seq_pk[::-1])
    ax3.set_xlabel('\nQuery nucleotide')
    ax3.set_ylabel('Partner nucleotide\n')
    ax3.set_title('B\n', loc='left')


    for i in range((len(df))):
            ans = df.iloc[i]
            for pos in range(len(ans)):
                if (i, pos) in true_pairs_pk:
                    ax3.add_patch(Rectangle((i, len(df)-pos-1), 1, 1, fill=False, edgecolor='black', lw=2.5))

    ax3.add_patch(Rectangle((7, 0), 1, 1, fill=False, edgecolor='blue', lw=2.5))
    ax3.add_patch(Rectangle((8, 1), 1, 1, fill=False, edgecolor='blue', lw=2.5))
    ax3.add_patch(Rectangle((9, 2), 1, 1, fill=False, edgecolor='blue', lw=2.5))
    ax3.add_patch(Rectangle((10, 3), 1, 1, fill=False, edgecolor='blue', lw=2.5))

    ax3.add_patch(Rectangle((len(seq_pk)- 1, len(seq_pk)-8), 1, 1, fill=False, edgecolor='blue', lw=2.5))
    ax3.add_patch(Rectangle(((len(seq_pk) - 2), (len(seq_pk) - 9)), 1, 1, fill=False, edgecolor='blue', lw=2.5))
    ax3.add_patch(Rectangle(((len(seq_pk) - 3), (len(seq_pk) - 10)), 1, 1, fill=False, edgecolor='blue', lw=2.5))
    ax3.add_patch(Rectangle(((len(seq_pk) - 4), (len(seq_pk) - 11)), 1, 1, fill=False, edgecolor='blue', lw=2.5))

    plt.show()


def get_mean_qualities(c):
    seq = decode(test_q[c][0])
    #seq = gen_rand_seq(len(seq)) ##########################
    #print(seq2rf_id["".join(seq)], accs[c])
    ans, grad = get_net_ans(seq)
    ans = symmetrize_min(ans)
    imp = get_importances(grad)
    imp = symmetrize_min(imp)
    return np.mean(ans), np.mean(imp)

def get_mask_poss(qs, ans):
    mask = set()
    for i in range(len(qs)):
        q, a = qs[i], ans[i]
        p1, p2 = get_q_pos(q), get_ans_pos(a)
        mask.add((p1, p2))
        mask.add((p2, p1))
    return mask

def get_q_pos(qu):
    for i in range(len(qu)):
        if qu[i][4] == 2: #############
            return i
        
def get_ans_pos(ans):
    for i in range(len(ans)):
        if ans[i] == 1:
            return i

def get_pair_stat(model, test_q, test_ans):
    imps_true, imps_rand = [], []
    ans_true, ans_rand = [], []

    for c in tqdm(range(len(test_q))):
        seq = decode(test_q[c][0])
        ans, grad = get_ans_grad(seq, model)
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

def plot_importances(at, ar, it2, ir2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi = 340.0) 

    ax1.hist([at, ar], density="True")
    ax1.set_xlabel("Confidence in the prediction")
    ax1.set_ylabel("Density")
    ax1.set_title('A', loc='left')
    ax1.legend(["Paired", "Random"])


    ax2.hist([it2, ir2], density="True")
    ax2.set_xlabel("Log of the importance of the position\n for the prediction")
    ax2.set_ylabel("Density")
    ax2.set_title('B', loc='left')
    ax2.legend(["Paired", "Random"])


    plt.show()

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

def get_lets_pair_stat(k2seq, model):
    pair2count = {}
    ppair2count = {}
    for i, k in enumerate(k2seq):
        if i % 100 == 0:
            print(i)
        seq = k2seq[k]
        ans = get_net_ans(seq, model)
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

def get_energy_data(pair2count, ppair2count):
    
    mat = [[-2.4, -3.3, -2.1, -1.4, -2.1, -2.1], \
       [-3.3, -3.4, -2.5, -1.5, -2.2, -2.4], \
       [-2.1, -2.5, 1.3, -0.5, -1.4, -1.3], \
       [-1.4, -1.5, -0.5, 0.3, -0.6, -1.0], \
       [-2.1, -2.2, -1.4, -0.6, -1.1, -0.9], \
       [-2.1, -2.4, -1.3, -1.0, -0.9, -1.3]]
    ls = ["CG", "GC", "GU", "UG", "AU", "UA"]

    pairs, counts = [], []

    for p in sorted(pair2count, key=lambda x: pair2count[x], reverse=True):
        if not "-".join(p[::-1]) in pairs:
            pairs.append("-".join(p))
            counts.append(pair2count[p]/sum(pair2count.values()))
            
    ppair2energy = {}
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            ppair2energy[(ls[i], ls[j])] = mat[i][j]
            
    ps, ens, counts_2 = [], [], []
    for p in ppair2energy:
        ps.append(p)
        ens.append(ppair2energy[p])
        counts_2.append(ppair2count[p])
        
    return pairs, ens, counts, counts_2

def plot_energy_data(pairs, energies, counts_of_pairs, counts_of_stacked):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi = 340.0) 

    ax1.bar(pairs, height=counts_of_pairs)
    ax1.set_xlabel("Pair")
    ax1.set_ylabel("Frequency")
    ax1.set_title('A\n', loc='left')

    ax2.plot(energies, counts_of_stacked, "o")
    ax2.set_xlabel("Stacking energy")
    ax2.set_ylabel("Frequency")
    ax2.set_title('B\n', loc='left')

    r = str(round(spearmanr(energies, counts_of_stacked)[0], 2))
    p = str(round(spearmanr(energies, counts_of_stacked)[1], 7))

    textstr = 'r = {0}\np = {1}'.format(r, p)
    # ax.hist(x, 50)
    # # these are matplotlib.patch.Patch properties
    # props = dict(boxstyle=None, facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax2.text(0.70, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
            verticalalignment='top')

    plt.show()

def count_paired_fraction(ss):
    pairing_symbols = "<>()[]{}" + string.ascii_letters
    count = sum([1 for l in ss if l in pairing_symbols])
    return count/len(ss)

def calc_fraction_distribution():
    frs = []
    for i in range(len(data_good)):
        fr = count_paired_fraction(data_good[i][-1])
        frs.append(fr)
    return frs

def clean_raw_rfam(RFAM_SEED):
    alns = AlignIO.parse(RFAM_SEED, "stockholm")
    alns = list(alns)
    
    data = []

    with open(RFAM_SEED) as f:
        d = []
        for line in f:
            if "#=GF DE" in line:
                s = line.strip().split()
                ids = '_'.join(s[2:])
                d.append(ids)

            if "#=GF CL" in line:
                s = line.strip().split()
                cl = s[2]
                d.append(cl)

            if "#=GC SS_cons" in line:
                s = line.strip().split()
                ss = s[2]
                d.append(ss)

            if "# STOCKHOLM 1.0" in line:
                if len(d) > 0:
                    data.append(d)
                    d = []

        if len(d) > 0:
            data.append(d)
    lens = {}
    bad_indexes = set()
    for i, d in enumerate(data):
        l = len(d)
        lens[l] = lens.get(l, 0) + 1
        if l == 4:
            bad_indexes.add(i)

    alns_good = [alns[i] for i in range(len(alns)) if i not in bad_indexes]
    data_good = [data[i] for i in range(len(data)) if i not in bad_indexes]

    return alns_good, data_good

def filter_by_fr(threshold, data_good, alns_good):
    alns_filtered, data_filtered = [], []
    for i in range(len(data_good)):
        fr = count_paired_fraction(data_good[i][-1])
        if fr >= threshold:
            alns_filtered.append(alns_good[i])
            data_filtered.append(data_good[i])
    print(len(alns_filtered))
    return alns_filtered, data_filtered

def filter_by_clan(alns_filtered, data_filtered):
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
    pairing_symbols = "<>()[]{}" + string.ascii_letters
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
                
def gen_pre_dataset(data_filtered, alns_filtered):
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

def make_class2seqs(ds):
    class2seqs = {}

    for x in ds:
        cl, seq = x[0], x[1]
        if not cl in class2seqs:
            class2seqs[cl] = set()
        class2seqs[cl].add(seq)

    bad_classes = []
    for cl in class2seqs:
        if len(class2seqs[cl]) < 100:
            bad_classes.append(cl)

    for cl in bad_classes:
        class2seqs.pop(cl)
    return class2seqs

def make_cl2embs(model, class2seqs, layer_name='bidirectional'):
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    cl2embs = {}

    for cl in tqdm(class2seqs):
        cl2embs[cl] = []
        seqs = random.sample(class2seqs[cl], len(class2seqs[cl]))
        for seq in seqs:
            emb = get_embedding(seq, model, intermediate_layer_model)
            cl2embs[cl].append(emb)
    return cl2embs

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

def make_rand_seqs2embs(RAND_SEQS, model, layer_name='bidirectional'):
    
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    with open(RAND_SEQS) as f: 
        rand_seqs = []
        for i in f:
            rand_seqs.append(i.strip().split('\t'))

    rand_seqs_dict = {}

    for x in rand_seqs:
        cl, seq = x[1], x[0]
        if not cl in rand_seqs_dict:
            rand_seqs_dict[cl] = set()
        rand_seqs_dict[cl].add(seq)

    rand_seqs2embs = {}

    for cl in tqdm(rand_seqs_dict):
        rand_seqs2embs[cl] = []
        seqs = random.sample(rand_seqs_dict[cl], 100)
        for seq in seqs:
            emb = get_embedding(seq, model, intermediate_layer_model)
            rand_seqs2embs[cl].append(emb)

    return rand_seqs2embs

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def plot_tsne(y, y_r, x_embedded, xr_embedded):
    sns.set_style(style='white')
    df = pd.DataFrame(x_embedded)
    df['RNA'] = y
    y_r_new = ["Class " + str(int(i) + 1) for i in y_r]

    style = []

    for _ in df.index:
        if _ <= len(df)/4:
            style.append('1')
        if _ > len(df)/4 and _ <= 2*(len(df)/4):
            style.append('2')
        if _ > 2*(len(df)/4) and _ <= 3*(len(df)/4):
            style.append('3')
        if _ > 3*(len(df)/4):
            style.append('4')

    df['Marker'] = style
    df['_'] = ['_' for _ in range(len(df))]
    df['Rfam family'] = df['Marker'] +  df['_']+ df['RNA']
    markers = {'1': "o", '2': "v", '3': "X", '4': 'D' }

    fig = plt.figure(figsize=(15.5, 7), dpi=340)

    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.1, hspace=0)

    g = fig.add_subplot(gs[0, 0])

    g = sns.scatterplot(df[0], df[1], hue=df['Rfam family'], style=df['Marker'], markers=markers, 
                          legend='full', palette='tab20')
    g.legend(loc='lower center', bbox_to_anchor=(0.5, -1.1), ncol=2, fontsize=8)
    g.grid(False)
    g.set_xlabel(None)
    g.set_ylabel(None)
    g.set_title('A\n', loc='left')


    g2 = fig.add_subplot(gs[0, 1])
    g2.set_title('B\n', loc='left')
    g2 = sns.scatterplot(xr_embedded[:,0], xr_embedded[:,1], hue=y_r_new, legend='full', palette='tab20')
    g2.grid(False)
    handles, labels = plt.gca().get_legend_handles_labels()
    g2.legend(handles, natural_sort(labels), loc='lower center', bbox_to_anchor=(0.5, -0.328), ncol=3)
    g2.set_xlabel(None)
    g2.set_ylabel(None)
    plt.show()
