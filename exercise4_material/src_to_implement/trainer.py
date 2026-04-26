import os
import torch as t
from sklearn.metrics import f1_score

class Trainer:
    def __init__(self, model, crit, optim=None, train_dl=None, val_test_dl=None,
                 cuda=True, early_stopping_patience=-1):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        os.makedirs('checkpoints', exist_ok=True)
        t.save({'state_dict': self._model.state_dict()},
               f'checkpoints/checkpoint_{epoch:03d}.ckp')

    def restore_checkpoint(self, epoch_n):
        device = 'cuda' if self._cuda else t.device('cpu')
        ckp = t.load(f'checkpoints/checkpoint_{epoch_n:03d}.ckp', device)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        t.onnx.export(
            m, x, fn, export_params=True, opset_version=10, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

    def train_step(self, x, y):
        self._optim.zero_grad(set_to_none=True)
        pred = self._model(x)
        loss = self._crit(pred, y)
        loss.backward()
        self._optim.step()
        return float(loss.item())

    @t.no_grad()
    def val_test_step(self, x, y):
        pred = self._model(x)
        loss = self._crit(pred, y)
        return float(loss.item()), pred

    def train_epoch(self):
        self._model.train()
        losses = []
        for xb, yb in self._train_dl:
            if self._cuda:
                xb = xb.cuda(non_blocking=True); yb = yb.cuda(non_blocking=True)
            losses.append(self.train_step(xb, yb))
        return sum(losses) / max(1, len(losses))

    @t.no_grad()
    def val_test(self):
        self._model.eval()
        losses, all_y, all_p = [], [], []
        for xb, yb in self._val_test_dl:
            if self._cuda:
                xb = xb.cuda(non_blocking=True); yb = yb.cuda(non_blocking=True)
            l, pred = self.val_test_step(xb, yb)
            losses.append(l)
            all_y.append(yb.detach().cpu())
            all_p.append(pred.detach().cpu())
        if all_y:
            Y = t.cat(all_y).numpy()
            P = t.cat(all_p).numpy()
            f1 = f1_score(Y, (P >= 0.5).astype('int32'), average='micro')
        else:
            f1 = 0.0
        return (sum(losses) / max(1, len(losses))), f1

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        train_losses, val_losses = [], []
        best_val = float('inf'); wait = 0
        epoch = 0
        while True:
            epoch += 1
            if epochs > 0 and epoch > epochs:
                break
            tr = self.train_epoch()
            va, _ = self.val_test()
            train_losses.append(tr); val_losses.append(va)

            if va < best_val - 1e-6:
                best_val = va; wait = 0
                self.save_checkpoint(epoch)
            else:
                wait += 1
                if self._early_stopping_patience > 0 and wait >= self._early_stopping_patience:
                    break
        return train_losses, val_losses
