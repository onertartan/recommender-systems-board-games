from abc import ABC, abstractmethod
import numpy as np


class BaseRecommender(ABC):
    df_id2name = {"game": None, "user": None, "content": None}
    df_name2id = {"game": None, "user": None, "content": None}

    def __init__(self):
        self.ids = None
        self.name = ""

    @abstractmethod
    def get_recommendations(self, target_id, num_recommendations=10):
        pass

    def get_target_id(self, entered_id):
        if not entered_id:
            target_id = np.random.choice(self.ids)
        else:
            try:
                target_id = int(entered_id)
                if not (target_id in self.ids):
                    target_id = np.random.choice(self.ids)
            except:  # if user enters user name (str)
                entered_id = self.df_name2id[self.name].loc[entered_id].item()

                if entered_id in self.ids:
                    return entered_id
                else:
                    target_id = np.random.choice(self.ids)
        return target_id
