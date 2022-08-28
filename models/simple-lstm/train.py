from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from train_module import SimpleTrainModule
from data_module import SimpleDataModule
from model import SimpleLstm


BATCH_SIZE = 64

data_module = SimpleDataModule(batch_size=BATCH_SIZE)
model = SimpleLstm(num_embeddings=data_module.num_embeddings)
train_module = SimpleTrainModule(model=model)


logger = TensorBoardLogger(save_dir='./logs/model-training/')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=True, mode='min')

trainer = Trainer(max_epochs=500, callbacks=[early_stopping], logger=logger)


if __name__ == '__main__':
    trainer.fit(train_module, data_module)
