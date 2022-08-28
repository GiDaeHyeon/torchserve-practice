import os
from glob import glob


class IMDB:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        if mode == 'train':
            self.dataset = sorted(glob(f'./data/IMDB/{mode}/*/*.txt'))
        elif mode == 'test':
            self.dataset = sorted(glob(f'./data/IMDB/{mode}/pos/*.txt'))[:2500] \
                           + sorted(glob(f'./data/IMDB/{mode}/neg/*.txt'))[:2500]
        else:
            raise ValueError(f'mode 인자는 train 또는 test이어야 합니다.\n 현재 mode 인자 값은 {mode}입니다.')

        if len(self.dataset) == 0:
            raise ValueError(f'경로 재지정 요망')

    def get_data(self) -> tuple:
        datas = []
        for data in self.dataset:
            with open(data, 'r', encoding='utf-8') as txt:
                datas.append(txt.read())
        labels = [1 if data.split('/')[-2] == 'pos' else 0 for data in self.dataset]
        return datas, labels
