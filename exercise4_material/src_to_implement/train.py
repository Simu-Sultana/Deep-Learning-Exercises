import torch as t
import pandas as pd
from sklearn.model_selection import train_test_split
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model

def main():
    # 1) load CSV (semicolon separated)
    tab = pd.read_csv('data.csv', sep=';')

    # 2) split
    train_df, val_df = train_test_split(tab, test_size=0.2, random_state=42, shuffle=True)

    # 3) datasets & loaders
    # keep workers >0 for speed; Windows needs the main-guard (this file has it)
    num_workers = 2
    pin = t.cuda.is_available()
    train_ds = ChallengeDataset(train_df, 'train')
    val_ds   = ChallengeDataset(val_df,   'val')
    train_dl = t.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_dl   = t.utils.data.DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin)

    # 4) model, loss, optim, trainer
    net = model.ResNet()
    crit = t.nn.BCELoss()                    # model outputs probabilities (sigmoid)
    opt  = t.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    use_cuda = t.cuda.is_available()
    trainer = Trainer(net, crit, optim=opt, train_dl=train_dl, val_test_dl=val_dl,
                      cuda=use_cuda, early_stopping_patience=7)

    # 5) train
    train_losses, val_losses = trainer.fit(epochs=10)

    # 6) plot
    plt.figure()
    plt.plot(np.arange(len(train_losses)), train_losses, label='train loss')
    plt.plot(np.arange(len(val_losses)),   val_losses,   label='val loss')
    plt.yscale('log'); plt.legend()
    plt.savefig('losses.png')
    print("Saved losses.png")

if __name__ == "__main__":
    
    main()
