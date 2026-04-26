import torch as t
from trainer import Trainer
import sys
from model import ResNet

epoch = int(sys.argv[1])

model = ResNet()
crit = t.nn.BCELoss()
trainer = Trainer(model, crit, cuda=False)
trainer.restore_checkpoint(epoch)
trainer.save_onnx(f'checkpoint_{epoch:03d}.onnx')
print(f"Exported ONNX: checkpoint_{epoch:03d}.onnx")
