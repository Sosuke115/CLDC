import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.RANDOM_STATE = kwargs["RANDOM_STATE"]
        self.WV_MATRIX = kwargs["WV_MATRIX"]
        self.GPU = kwargs["GPU"]
        self.IN_CHANNEL = 1

        torch.manual_seed(self.RANDOM_STATE)
        torch.cuda.manual_seed(self.RANDOM_STATE)

        assert len(self.FILTERS) == len(self.FILTER_NUM)

        self.embedding = nn.Embedding(
            len(kwargs["WV_MATRIX"]), self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1
        )
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))

        self.embedding.weight.requires_grad = False

        seed_arr = range(len(self.FILTERS))
        seed_arr = shuffle(seed_arr, random_state=self.RANDOM_STATE)
        torch.manual_seed(self.RANDOM_STATE)
        torch.cuda.manual_seed(self.RANDOM_STATE)

        for i in range(len(self.FILTERS)):
            torch.manual_seed(seed_arr[i])
            torch.cuda.manual_seed(seed_arr[i])
            conv = nn.Conv1d(
                self.IN_CHANNEL,
                self.FILTER_NUM[i],
                self.WORD_DIM * self.FILTERS[i],
                stride=self.WORD_DIM,
            )
            setattr(self, f"conv_{i}", conv)
        torch.manual_seed(self.RANDOM_STATE)
        torch.cuda.manual_seed(self.RANDOM_STATE)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f"conv_{i}")

    def forward(self, inp):

        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(
                F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1
            ).view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))
        ]

        x = torch.cat(conv_results, 1)
        torch.manual_seed(self.RANDOM_STATE)
        torch.cuda.manual_seed(self.RANDOM_STATE)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)

        return x
