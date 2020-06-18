from utils import load_pickle, save_pickle
import re
import string

ALPH = string.ascii_uppercase


def process_babi_dataset(save, print_dict=False):
    file = open('dialog-bAbI-tasks/dialog-babi-task5-full-dialogs-trn.txt', 'r')
    text = file.readlines()
    file.close()
    system_acts = load_pickle('system_acts.pickle')

    def print_dict():
        for key in uttr_dict:
            print(key)
            print(uttr_dict[key])
            print()

    uttr_dict = {'<BEGIN>': [set()]}
    for act in system_acts:
        uttr_dict[act] = [set()]

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
                uttr_dict[prev_uttr][0].add(user_uttr)
                prev_uttr = act

    if save:
        save_pickle(uttr_dict, 'simulator_uttrs.pickle')
    if print_dict:
        for k, v in uttr_dict.items():
            print(k, v, '\n')



def process_example_phrases(save, print_dict=False):
    from openpyxl import load_workbook
    def cell(row, col):
        return sh[ALPH[col-1]+str(row)].value

    uttr_dict = {}
    uttr_dict['<SILENT>'] = [set()]
    uttr_dict['any preference on a type of cuisine'] = [set(), set()]
    uttr_dict['api_call'] = [set()]
    uttr_dict['great let me do the reservation'] = [set()]
    uttr_dict['hello what can i help you with today'] = [set()]
    uttr_dict['here it is '] = [set(), set()]
    uttr_dict['how many people would be in your party'] = [set()]
    uttr_dict["i'm on it"] = [set()]
    uttr_dict['is there anything i can help you with'] = [set()]
    uttr_dict['ok let me look into some options for you'] = [set()]
    uttr_dict['sure is there anything else to update'] = [set()]
    uttr_dict['sure let me find an other option for you'] = [set()]
    uttr_dict['what do you think of this option: '] = [set()]
    uttr_dict['where should it be'] = [set()]
    uttr_dict['which price range are looking for'] = [set(), set()]
    uttr_dict["you're welcome"] = [set()]
    uttr_dict['<BEGIN>'] = [set()]

    wb = load_workbook(filename='user_simulator_phrases.xlsx')
    sh = wb['Phrases']

    col = 0
    for phrase, list in uttr_dict.items():
        col += 1
        row = 2
        while True:
            phrase = cell(row, col)
            if phrase == None:
                break
            else:
                list[0].add(phrase)
                row += 1
                if ' ' not in phrase:
                    pass

        if len(list) > 1:  # only necessary when context_vector can be != [1, 1, 1, 1]
            row = 2
            col += 1
            while True:
                if cell(row, col) == None:
                    break
                else:
                    list[1].add(cell(row, col))
                    row += 1
    if save:
        save_pickle(uttr_dict, 'example_phrases_dict.pickle')
    if print_dict:
        for k, v in uttr_dict.items():
            print(k, v, '\n')


#process_babi_dataset(save=True, print_dict=True)
process_example_phrases(save=True, print_dict=True)
