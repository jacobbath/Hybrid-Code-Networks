import re
import copy
import pickle
import numpy as np
from collections import OrderedDict
import torch
from torch.autograd import Variable
import global_variables as g


def save_checkpoint(state, filename='./checkpoints/checkpoint.pth.tar'):
    print('save model!', filename)
    torch.save(state, filename)


def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f)


def load_pickle(path):
    print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)


def get_entities(fpath):
    entities = OrderedDict({'R_cuisine': [], 'R_location': [], 'R_price': [], 'R_number': []})
    with open(fpath, 'r') as file:
        lines = file.readlines()
        for l in lines:
            wds = l.rstrip().split(' ')[2].split('\t')
            slot_type = wds[0] # ex) R_price
            slot_val = wds[1] # ex) cheap
            # if slot_type not in entities:
            #     entities[slot_type] = []
            if slot_type in entities:
                if slot_val not in entities[slot_type]:
                    entities[slot_type].append(slot_val)
    return entities


def load_embd_weights(word2vec, vocab_size, embd_size, w2i):
    embedding_matrix = np.zeros((vocab_size, embd_size))
    print('embed_matrix.shape', embedding_matrix.shape)
    found_ct = 0
    for word, idx in w2i.items():
        # words not found in embedding index will be all-zeros.
        if word in word2vec.wv:
            embedding_matrix[idx] = word2vec.wv[word]
            found_ct += 1
    print(found_ct, 'words are found in word2vec. vocab_size is', vocab_size)
    return torch.from_numpy(embedding_matrix).type(torch.FloatTensor)


def preload(fpath, vocab, system_acts):
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            l = l.rstrip()
            if l != '':
                ls = l.split("\t")
                t_u = ls[0].split(' ', 1)
                # turn = t_u[0]
                uttr = t_u[1].split(' ')
                if len(ls) == 2: # includes user and system utterance
                    for w in uttr:
                        if w not in vocab:
                            vocab.append(w)
                if len(ls) == 2: # includes user and system utterance
                    sys_act = ls[1]
                    sys_act = re.sub(r'resto_\S+', '', sys_act)
                    if sys_act.startswith('api_call'): sys_act = 'api_call'
                    if sys_act not in system_acts: system_acts.append(sys_act)
    vocab = sorted(vocab)
    system_acts = sorted(system_acts)
    return vocab, system_acts


def load_data_from_file(fpath, entities, w2i, system_acts):
    '''
    store data as dialog (multi turns)
    '''
    data = []
    with open(fpath, 'r') as f:
        lines = f.readlines()
        # x: user uttr, y: sys act, c: context, b: BoW, p: previous sys act, f: action filter
        x, y, c, b, p, f = [], [], [], [], [], []
        context    = [0] * len(entities.keys())
        for idx, l in enumerate(lines):
            l = l.rstrip()
            if l == '':
                data.append((x, y, c, b, p, f))
                # reset
                x, y, c, b, p, f = [], [], [], [], [], []
                context = [0] * len(entities.keys())
            else:
                ls = l.split("\t")
                t_u = ls[0].split(' ', 1)
                # turn = t_u[0]
                uttr = t_u[1].split(' ')
                update_context(context, uttr, entities)
                act_filter = generate_act_filter(len(system_acts), context)
                bow = get_bow(uttr, w2i)
                sys_act = g.SILENT
                if len(ls) == 2: # includes user and system utterance
                    sys_act = ls[1]
                    sys_act = re.sub(r'resto_\S+', '', sys_act)
                    if sys_act.startswith('api_call'): sys_act = 'api_call'
                else:
                    continue # TODO

                x.append(uttr)
                if len(y) == 0:
                    p.append(g.SILENT)
                else:
                    p.append(y[-1])
                y.append(sys_act)
                c.append(copy.deepcopy(context))
                b.append(bow)
                f.append(act_filter)
    return data, system_acts


def load_data_from_string(uttr, entities, w2i, system_acts, context):
    '''
    store data as dialog (multi turns)
    '''
    data = []
    # x: user uttr, y: sys act, c: context, b: BoW, p: previous sys act, f: action filter
    x, y, c, b, p, f = [], [], [], [], [], []
    if not context:
        context = [0] * len(entities.keys())
    data.append((x, y, c, b, p, f))
    update_context(context, uttr.split(' '), entities)
    act_filter = generate_act_filter(len(system_acts), context)
    bow = get_bow(uttr, w2i)
    sys_act = g.SILENT
    x.append(uttr)

    if len(y) == 0:
        p.append(g.SILENT)
    else:
        p.append(y[-1])
    y.append(sys_act)
    c.append(copy.deepcopy(context))
    b.append(bow)
    f.append(act_filter)
    return data, system_acts, context


def update_context(context, sentence, entities):
    for idx, (ent_key, ent_vals) in enumerate(entities.items()):
        for w in sentence:
            if w in ent_vals:
                context[idx] = 1


def generate_act_filter(action_size, context):
    mask = [0] * action_size
    # TODO hard coding
    # 0  <SILENT>
    # 1  any preference on a type of cuisine
    # 2  api_call
    # 3  great let me do the reservation
    # 4  hello what can i help you with today
    # 5  here it is
    # 6  how many people would be in your party
    # 7  i'm on it
    # 8  is there anything i can help you with
    # 9  ok let me look into some options for you
    # 10 sure is there anything else to update
    # 11 sure let me find an other option for you
    # 12 what do you think of this option:
    # 13 where should it be
    # 14 which price range are looking for
    # 15 you're welcome
    # context: {'R_cuisine': [], 'R_location': [], 'R_price': [], 'R_number': []}
    mask[0] = 1
    mask[7] = 1
    mask[8] = 1
    if context == [0, 0, 0, 0]:
        mask[4] = 1

    if context == [1, 1, 1, 1]:
        mask[2] = 1
        mask[3] = 1
        mask[5] = 1
        mask[8] = 1
        mask[9] = 1
        mask[10] = 1
        mask[11] = 1
        mask[12] = 1
        mask[15] = 1

    if context[0] == 0: # R_cuisine
        mask[1] = 1
    if context[1] == 0: # R_location
        mask[13] = 1
    if context[2] == 0: # R_price
        mask[14] = 1
    if context[3] == 0: # R_number
        mask[6] = 1

    return mask


def get_bow(sentence, w2i):
    bow = [0] * len(w2i)
    for word in sentence:
        if word in w2i:
            bow[w2i[word]] += 1
    return bow


def add_padding(data, seq_len):
    pad_len = max(0, seq_len - len(data))
    data += [0] * pad_len
    data = data[:seq_len]
    return data


def make_word_vector(uttrs_list, w2i, dialog_maxlen, uttr_maxlen):
    dialog_list = []
    for uttrs in uttrs_list:
        dialog = []
        for sentence in uttrs:
            sent_vec = [w2i[w] if w in w2i else w2i[g.UNK] for w in sentence]
            sent_vec = add_padding(sent_vec, uttr_maxlen)
            dialog.append(sent_vec)
        for _ in range(dialog_maxlen - len(dialog)):
            dialog.append([0] * uttr_maxlen)
        dialog = torch.LongTensor(dialog[:dialog_maxlen])
        dialog_list.append(dialog)
    return to_var(torch.stack(dialog_list, 0))


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def padding(data, default_val, maxlen, pad_seq_len):
    for i, d in enumerate(data):
        pad_len = maxlen - len(d)
        for _ in range(pad_len):
            data[i].append([default_val] * pad_seq_len)
    return to_var(torch.FloatTensor(data))


def get_data_from_batch(batch, w2i, act2i, labels_included):
    uttrs_list = [d[0] for d in batch]
    dialog_maxlen = max([len(uttrs) for uttrs in uttrs_list])
    uttr_maxlen = max([len(u) for uttrs in uttrs_list for u in uttrs])
    uttr_var = make_word_vector(uttrs_list, w2i, dialog_maxlen, uttr_maxlen)

    labels_var = []
    if labels_included:
        batch_labels = [d[1] for d in batch]
        for labels in batch_labels:
            vec_labels = [act2i[l] for l in labels]
            pad_len = dialog_maxlen - len(labels)
            for _ in range(pad_len):
                vec_labels.append(act2i[g.SILENT])
            labels_var.append(torch.LongTensor(vec_labels))
        labels_var = to_var(torch.stack(labels_var, 0))

    batch_prev_acts = [d[4] for d in batch]
    prev_var = []
    for prev_acts in batch_prev_acts:
        vec_prev_acts = []
        for act in prev_acts:
            tmp = [0] * len(act2i)
            tmp[act2i[act]] = 1
            vec_prev_acts.append(tmp)
        pad_len = dialog_maxlen - len(prev_acts)
        for _ in range(pad_len):
            vec_prev_acts.append([0] * len(act2i))
        prev_var.append(torch.FloatTensor(vec_prev_acts))
    prev_var = to_var(torch.stack(prev_var, 0))

    context = copy.deepcopy([d[2] for d in batch])
    context = padding(context, 1, dialog_maxlen, len(context[0][0]))

    bow = copy.deepcopy([d[3] for d in batch])
    bow = padding(bow, 0, dialog_maxlen, len(bow[0][0]))

    act_filter = copy.deepcopy([d[5] for d in batch])
    act_filter = padding(act_filter, 0, dialog_maxlen, len(act_filter[0][0]))

    return uttr_var, labels_var, prev_var, context, bow, act_filter