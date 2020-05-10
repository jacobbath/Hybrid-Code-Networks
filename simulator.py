import random
from utils import load_pickle


class Simulator():

    def __init__(self, source, entities):
        data = load_pickle(source)
        self.train_uttrs = {}
        self.test_uttrs = {}
        for bot_uttr, resp_list in data.items():
            self.train_uttrs[bot_uttr] = []
            self.test_uttrs[bot_uttr] = []
            for resp_set in resp_list:
                if len(resp_set) == 0:
                    self.train_uttrs[bot_uttr].append(set())
                    self.test_uttrs[bot_uttr].append(set())
                else:
                    train_set = set(random.sample(resp_set, round(len(resp_set)*0.5)))
                    test_set = resp_set-train_set
                    if len(test_set) == 0:
                        test_set = train_set
                    self.train_uttrs[bot_uttr].append(train_set)
                    self.test_uttrs[bot_uttr].append(test_set)
        self.entities = entities

    def respond(self, uttr, context_goal, is_test):
        if is_test:
            uttr_dict = self.test_uttrs
        else:
            uttr_dict = self.train_uttrs

        if uttr == 'great let me do the reservation': resp = '<THANK YOU>'
        elif len(uttr_dict[uttr]) > 1:
            if uttr == 'any preference on a type of cuisine':
                if context_goal[0]:
                    resp = random.sample(uttr_dict[uttr][0], 1)[0]
                else:
                    resp = random.sample(uttr_dict[uttr][1], 1)[0]
            elif uttr == 'which price range are looking for':
                if context_goal[2]:
                    resp = random.sample(uttr_dict[uttr][0], 1)[0]
                else:
                    resp = random.sample(uttr_dict[uttr][1], 1)[0]
            else:  # case when bot says 'here it is'
                r = random.randint(0, len(uttr_dict[uttr])-1)
                if r == 0: resp = '<THANK YOU>'
                else: resp = random.sample(uttr_dict[uttr][r], 1)[0]
        else:
            resp = random.sample(uttr_dict[uttr][0], 1)[0]
        if '<cuisine>' in resp:
            resp = resp.replace('<cuisine>', random.sample(self.entities['R_cuisine'], 1)[0])
        if '<location>' in resp:
            resp = resp.replace('<location>', random.sample(self.entities['R_location'], 1)[0])
        if '<price>' in resp:
            resp = resp.replace('<price>', random.sample(self.entities['R_price'], 1)[0])
        if '<number>' in resp:
            resp = resp.replace('<number>', random.sample(self.entities['R_number'], 1)[0])

        return resp
