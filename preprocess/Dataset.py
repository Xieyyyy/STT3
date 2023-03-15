import pickle

import numpy as np
import torch
import torch.utils.data
from transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, file, domain):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        print("load time...")
        with open(file + domain + "/" + domain + "_time.pkl", "rb") as f:
            self.time = pickle.load(f)
        print("load time gap...")
        with open(file + domain + "/" + domain + "_time_gap.pkl", "rb") as f:
            self.time_gap = pickle.load(f)
        print("load event type...")
        with open(file + domain + "/" + domain + "_event_type.pkl", "rb") as f:
            self.event_type = pickle.load(f)
        print("load weather info...")
        with open(file + domain + "/" + domain + "_weather_info.pkl", "rb") as f:
            self.weather_info = pickle.load(f)
        # print("load weather info next...")
        # with open(file + domain + "/" + domain + "_weather_info_next.pkl", "rb") as f:
        #     self.weather_info_next = pickle.load(f)
        # print("load relative time 1...")
        # with open(file + domain + "/" + domain + "_relative_time1.pkl", "rb") as f:
        #     self.relative_time1 = pickle.load(f)
        # print("load relative time 2...")
        # with open(file + domain + "/" + domain + "_relative_time2.pkl", "rb") as f:
        #     self.relative_time2 = pickle.load(f)

        # self.relative_time(self.time)

        self.length = len(self.time)

    def relative_time(self, time):
        '''transfer the time for a event seq. each time-zero is the first event of a seq'''
        for inst in time:
            time_zero = inst[0]
            inst[0] = 0.1
            for i in range(len(inst) - 1):
                inst[i + 1] -= time_zero - 0.1

    def norm_time(self, time):
        time_min = time[0][0]
        time_max = time[len(time) - 1][-1]
        diff = time_max - time_min
        for inst in time:
            for i in range(len(inst) - 1):
                inst[i] = 1000 * (inst[i] - time_min) / diff
        time[0][0] = 1e-4

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.weather_info[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(max([len(region_inst) for region_inst in inst]) for inst in insts)

    batch_seq = np.array([[
        region_inst + [Constants.PAD] * (max_len - len(region_inst))
        for region_inst in inst] for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_weather(insts, weather_types=6):
    max_len = max(max([len(region_inst) for region_inst in inst]) for inst in insts)

    batch_seq = np.array([[region_inst +
                           [[Constants.PAD for _ in range(weather_types)] for _ in
                            range(max_len - len(region_inst))] for
                           region_inst in inst] for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(max([len(region_inst) for region_inst in inst]) for inst in insts)

    batch_seq = np.array([[
        region_inst + [Constants.PAD] * (max_len - len(region_inst))
        for region_inst in inst] for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type, weather_info = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    weather_info = pad_weather(weather_info)
    # weather_info_next = pad_weather(weather_info_next)
    # relative_time1 = pad_time(relative_time1)
    # relative_time2 = pad_time(relative_time2)
    return time, time_gap, event_type, weather_info


def get_dataloader(file, batch_size, shuffle=True, domain="train"):
    """ Prepare dataloader. """

    ds = EventData(file, domain)
    dl = torch.utils.data.DataLoader(
        ds,
        # num_workers=1,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
