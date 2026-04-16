import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.linear = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.linear(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super().__init__()
        assert imgH % 16 == 0, "imgH must be multiple of 16"

        # 🔴 NOTE: ConvNet wrapper EXISTS (this fixes your error)
        self.FeatureExtraction = nn.Module()
        self.FeatureExtraction.ConvNet = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(True)
        )

        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh)
        )

        self.Prediction = nn.Linear(nh, nclass)

    def forward(self, x):
        conv = self.FeatureExtraction.ConvNet(x)
        b, c, h, w = conv.size()
        assert h == 1, "Feature map height must be 1"

        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        contextual_feature = self.SequenceModeling(conv)
        output = self.Prediction(contextual_feature)
        return output
