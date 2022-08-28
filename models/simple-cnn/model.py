"""
torchserve를 활용한 모델 서빙에 활용할 아주 간단한 LSTM 모델입니다.
"""
from torch import nn


class SimpleLstm(nn.Module):
    """
    This is simple LSTM model
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM()

    def forward(self, input_vector):
        """
        Forward Method

        :param input_vector:
        :return:
        """
        return self.lstm(input_vector)
