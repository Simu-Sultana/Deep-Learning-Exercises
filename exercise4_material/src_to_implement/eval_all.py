import torch as t, pandas as pd, glob, re
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from data import ChallengeDataset
import model

tab = pd.read_csv('data.csv', sep=';')
val = tab.sample(frac=0.2, random_state=42)
val_dl = DataLoader(ChallengeDataset(val, 'val'), batch_size=64, shuffle=False)

ckps = sorted(glob.glob('checkpoints/checkpoint_*.ckp'))
assert ckps, "No checkpoints found"

def f1_for(ckp):
    net = model.ResNet().eval()
    net.load_state_dict(t.load(ckp, map_location='cpu')['state_dict'])
    Y,P=[],[]
    with t.no_grad():
        for x,y in val_dl:
            P.append(net(x))
            Y.append(y)
    P=t.cat(P).numpy(); Y=t.cat(Y).numpy()
    return f1_score(Y, (P>=0.5).astype('int32'), average='micro')

scores = []
for c in ckps:
    epoch = int(re.search(r'checkpoint_(\d+)\.ckp', c).group(1))
    scores.append((epoch, f1_for(c)))
scores.sort()
for ep, s in scores:
    print(f"epoch {ep:02d}  micro-F1 {s:.4f}")

best_ep, best_f1 = max(scores, key=lambda x: x[1])
print(f"\nBEST by F1 -> epoch {best_ep:02d}  (micro-F1={best_f1:.4f})")
