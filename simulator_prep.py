from utils import load_pickle, save_pickle
import re

file = open('dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt', 'r')
text = file.readlines()
file.close()
system_acts = load_pickle('system_acts.pickle')

def print_dict():
    for key in uttr_dict:
        print(key)
        print(uttr_dict[key])
        print()

uttr_dict = {'<BEGIN>': set()}
for act in system_acts:
    uttr_dict[act] = set()

prev_uttr = '<BEGIN>'
for uttr in text:
    if uttr == '\n':
        prev_uttr = '<BEGIN>'
    for act in system_acts:
        if prev_uttr == '':
            prev_uttr = act
            continue
        if act in uttr:
            user_uttr = re.sub(r'\d+', '', uttr.split(act)[0]).strip()
            uttr_dict[prev_uttr].add(user_uttr)
            prev_uttr = act

save_pickle(uttr_dict, 'simulator_uttrs.pickle')