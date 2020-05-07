import random
from utils import load_pickle
data = load_pickle('simulator_uttrs.pickle')


class Simulator():

    def __init__(self, uttr_dict):
        self.uttr_dict = uttr_dict

    def respond(self, uttr):
        return random.sample(self.uttr_dict[uttr], 1)[0]
