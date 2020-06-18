import os
import random
import copy
import argparse
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import save_pickle, load_pickle, load_embd_weights, to_var, save_checkpoint
from utils import preload, load_data_from_file, get_entities, get_data_from_batch, load_data_from_string
from models import HybridCodeNetwork
#from gensim.models import KeyedVectors
import global_variables as g
from simulator import Simulator


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='n of dialogs. HCN uses one dialog for one minibatch')
parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs') # epochs originally defaulted 5
parser.add_argument('--embd_size', type=int, default=300, help='word embedding size')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size for LSTM')
parser.add_argument('--test', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--save_model', type=int, default=0, help='path saved params')
parser.add_argument('--task', type=int, default=5, help='5 for Task 5 and 6 for Task 6')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--resume', type=str, metavar='PATH', help='path saved params')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
#torch.manual_seed(args.seed)
#random.seed(args.seed)
# np.random.seed(args.seed)


def categorical_cross_entropy(preds, labels):
    loss = Variable(torch.zeros(1))
    for p, label in zip(preds, labels):
        loss -= torch.log(p[label] + 1.e-7).cpu()
    loss /= preds.size(0)
    return loss


def print_dialog(uttrs, preds, labels):
    preds_text = []
    for pred_dist in preds:
        idx = torch.argmax(pred_dist).item()
        preds_text.append(system_acts[idx])

    for pred_idx, uttr in enumerate(uttrs[0]):
        uttr_text = ''
        for idx in uttr:
            if idx == 0:
                continue
            else:
                uttr_text += i2w[idx.item()] + ' '
        #print(f'{uttr_text} | {preds_text[pred_idx]} | {system_acts[labels[pred_idx]]}')
        print(f"User: {uttr_text} \\\\\nBot: {preds_text[pred_idx]} \\\\")


def simulate_dialog(system_acts, is_test):
    dialog = '\n'
    episode_actions = torch.Tensor([])
    bot_says = '<BEGIN>'
    turn_count = 0
    episode_return = 0
    context = False
    #context_goal = [random.randint(0,1), 1, random.randint(0,1), 1]
    context_goal = [1, 1, 1, 1]
    user_happy = False  # if user has said <THANK YOU>
    api_call_done = False
    while turn_count < 50:
        if bot_says == 'what do you think of this option:':  # bad solution for some bug
            bot_says = 'what do you think of this option: '
        if bot_says == 'here it is':  # bad solution for some bug
            bot_says = 'here it is '
        user_says = user_simulator.respond(bot_says, context_goal, is_test)
        if user_says == '<THANK YOU>':
            user_happy = True
        data, system_acts, context = load_data_from_string(user_says, entities, w2i, system_acts, context)
        uttrs, labels, contexts, bows, prevs, act_fils = get_data_from_batch(data, w2i, act2i,
                                                                             labels_included=False)
        preds = model(uttrs, contexts, bows, prevs, act_fils)
        c = Categorical(preds)
        action = c.sample()
        episode_actions = torch.cat([episode_actions, c.log_prob(action).reshape(1)])
        bot_says = system_acts[action]
        if bot_says == 'api_call':
            api_call_done = True
        dialog += f"User: {user_says} \\\\\nBot: {bot_says} \\\\\n"
        turn_count += 1

        # episode return evalutation rules
        if bot_says == "<SILENT>":
            break
        if bot_says == "you're welcome":
            if context != context_goal: break  # checks that the bot has all information
            if not user_happy: break
            else:
                episode_return = .95**(turn_count-1)
                break

    return episode_actions, episode_return, dialog


def train(model, data, optimizer, w2i, act2i, n_epochs=5, batch_size=1):
    print('----Train---')
    data = copy.copy(data)
    for epoch in range(1, n_epochs + 1):
        print('Epoch', epoch, '---------')
        random.shuffle(data)
        correct, total = 0, 0
        pretrain_episodes = 0
        return_per_episode = []
        success = []
        for i in tqdm(range(2000)):
            #if len(return_per_episode) > 10 and np.mean(return_per_episode[-200:]) > 0.75 and pretrain_episodes == 0:
                #break
            REINFORCE = False if i < pretrain_episodes else True

            if REINFORCE:
                episode_actions, episode_return, dialog = simulate_dialog(system_acts, is_test=False)
                if return_per_episode == []:
                    baseline = 0
                else:
                    baseline = np.mean(return_per_episode[-min(len(return_per_episode), 100):])
                loss = torch.sum(episode_actions*(episode_return-baseline)).mul(-1)
                print_simulated_dialog = False
                if i % 1 == 0:
                    print('\n100 rolling mean return:', np.mean(return_per_episode[-100:]))
                    print('Dialog', i+1)
                    #print(dialog, 'loss', loss.item(), 'return', episode_return)
                    print_simulated_dialog = True
                return_per_episode.append(episode_return)
                test_return = simulate_test_dialogs(1, print_simulated_dialog)
                if test_return == 0.:
                    success.append(0)
                else:
                    success.append(1)
            else:
                batch_idx = random.randint(0, len(data)-batch_size)
                batch = data[batch_idx:batch_idx + batch_size]
                uttrs, labels, contexts, bows, prevs, act_fils = get_data_from_batch(batch, w2i, act2i,
                                                                                     labels_included=True)
                preds = model(uttrs, contexts, bows, prevs, act_fils)
                action_size = preds.size(-1)
                preds = preds.view(-1, action_size)
                labels = labels.view(-1)
                loss = categorical_cross_entropy(preds, labels)
                correct += torch.sum((labels == torch.max(preds, 1)[1]).long()).item()  # ByteTensor to LongTensor
                total += labels.size(0)
                if i % 100 == 0:
                    print()
                    print_dialog(uttrs, preds, labels)
                    print('Acc: {:.3f}% ({}/{})'.format(100 * correct / total, correct, total))
                    print('loss', loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        window = 50
        return_rolling_mean = pd.Series(return_per_episode).rolling(window).mean()
        success_rolling_mean = pd.Series(success).rolling(window).mean()
        with open('return_per_episode.txt', 'w') as f:
            for epi, ret in return_rolling_mean.fillna(.0).items():
                f.write(f'{epi+1}\t{ret}\n')
        with open('success_rate.txt', 'w') as f:
            for epi, suc in success_rolling_mean.fillna(.0).items():
                f.write(f'{epi+1}\t{suc}\n')

        plt.subplot(121)
        plt.plot(return_rolling_mean)
        plt.ylim((0, 1))
        plt.xlim((0, len(return_rolling_mean)))
        plt.ylabel('Return')

        plt.subplot(122)
        plt.plot(success_rolling_mean)
        plt.ylim((0, 1))
        plt.xlim((0, len(success_rolling_mean)))
        plt.ylabel('Success rate')
        plt.show()
        # save the model {{{
        if args.save_model == 1:
            filename = 'ckpts/HCN-Epoch-{}.model'.format(epoch)
            save_checkpoint({
                'epoch'      : epoch,
                'state_dict' : model.state_dict(),
                'optimizer'  : optimizer.state_dict()
            }, filename=filename)
        # }}}


def test(model, data, w2i, act2i, batch_size=1):
    print('----Test---')
    model.eval()
    correct, total = 0, 0
    for batch_idx in range(0, len(data)-batch_size, batch_size):
        batch = data[batch_idx:batch_idx+batch_size]
        uttrs, labels, contexts, bows, prevs, act_fils = get_data_from_batch(batch, w2i, act2i, labels_included=False)

        preds = model(uttrs, contexts, bows, prevs, act_fils)
        action_size = preds.size(-1)
        preds = preds.view(-1, action_size)
        labels = labels.view(-1)
        # loss = F.nll_loss(preds, labels)
        correct += torch.sum(labels == torch.max(preds, 1)[1]).item()
        total += labels.size(0)
    print('Test Acc: {:.3f}% ({}/{})'.format(100 * correct/total, correct, total))


def simulate_test_dialogs(how_many, print_simulated_dialog=False):
    #model.eval()
    with torch.no_grad():
        for i in range(how_many):
            episode_actions, episode_return, dialog = simulate_dialog(system_acts, is_test=True)
            if print_simulated_dialog:
                print(dialog, 'return', episode_return)
        return episode_return


entities = get_entities('dialog-bAbI-tasks/dialog-babi-kb-all.txt')
for idx, (ent_name, ent_vals) in enumerate(entities.items()):
    print('entities', idx, ent_name, ent_vals[0] )

assert args.task == 5 or args.task == 6, 'task must be 5 or 6'
if args.task == 5:
    fpath_train = 'dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt'
    #fpath_train = 'pretrain_dialogs.txt'
    fpath_test = 'dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-tst-OOV.txt'
elif args.task == 6: # this is not working yet
    fpath_train = 'dialog-bAbI-tasks/dialog-babi-task6-dstc2-trn.txt'
    fpath_test = 'dialog-bAbI-tasks/dialog-babi-task6-dstc2-tst.txt'

system_acts = [g.SILENT]

vocab = []
# only read training vocabs because OOV vocabrary should not be contained
vocab, system_acts = preload(fpath_train, vocab, system_acts)
vocab = [g.UNK] + vocab
w2i = dict((w, i) for i, w in enumerate(vocab))
i2w = dict((i, w) for i, w in enumerate(vocab))
train_data, system_acts = load_data_from_file(fpath_train, entities, w2i, system_acts)
test_data, system_acts = load_data_from_file(fpath_test, entities, w2i, system_acts)
print('vocab size:', len(vocab))
print('action size:', len(system_acts))

max_turn_train = max([len(d[0]) for d in train_data])
max_turn_test = max([len(d[0]) for d in test_data])
max_turn = max(max_turn_train, max_turn_test)
print('max turn:', max_turn)
act2i = dict((act, i) for i, act in enumerate(system_acts))
print('action_size:', len(system_acts))
for act, i in act2i.items():
    print('act', i, act)

# use saved pickle since loading word2vec is slow.
#print('loading a word2vec binary...')
#model_path = './data/GoogleNews-vectors-negative300.bin'
#word2vec = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
#print('done')
#pre_embd_w = load_embd_weights(word2vec, len(vocab), args.embd_size, w2i)
#save_pickle(pre_embd_w, 'pre_embd_w.pickle')
#save_pickle(system_acts, 'system_acts.pickle')
pre_embd_w = load_pickle('pre_embd_w.pickle')

opts = {'use_ctx': True, 'use_embd': True, 'use_prev': True, 'use_mask': False}
model = HybridCodeNetwork(len(vocab), args.embd_size, args.hidden_size, len(system_acts), pre_embd_w, **opts)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))

if args.resume is not None and os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    start_epoch = ckpt['epoch'] + 1 if 'epoch' in ckpt else args.start_epoch
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
else:
    print("=> no checkpoint found")

user_source = 'example_phrases_dict.pickle'
#user_source = 'simulator_uttrs.pickle'

user_simulator = Simulator(user_source, entities)
if args.test != 1:
    train(model, train_data, optimizer, w2i, act2i, args.n_epochs, args.batch_size)
#simulate_test_dialogs(10, print_simulated_dialog=True)
test(model, test_data, w2i, act2i)
