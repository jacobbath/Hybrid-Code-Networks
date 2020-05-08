import random
from utils import load_pickle


class Simulator():

    def __init__(self, source):
        self.uttr_dict = load_pickle(source)

    def respond(self, uttr):
        if len(self.uttr_dict[uttr]) > 1:
            r = random.randint(0, len(self.uttr_dict[uttr])-1)
            resp = random.sample(self.uttr_dict[uttr][r], 1)[0]
        else:
            resp = random.sample(self.uttr_dict[uttr][0], 1)[0]
        if '<cuisine>' in resp:
            cuisines = ['japanese', 'chinese']
            resp = resp.replace('<cuisine>', random.sample(cuisines, 1)[0])

        return resp
