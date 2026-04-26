from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import torchvision.transforms as T

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std  = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, mode: str):
        assert mode in {"train", "val"}
        self.df = data.reset_index(drop=True)
        self.mode = mode
        self.path_col = self._infer_path_col(self.df)
        self.label_cols = self._infer_label_cols(self.df)

       
        self._bases = self._make_bases()

        if mode == "train":
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((300, 300)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(train_mean, train_std),
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((300, 300)),
                T.ToTensor(),
                T.Normalize(train_mean, train_std),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row[self.path_col])
        img = imread(str(img_path))              # grayscale H×W or RGB H×W×C
        if getattr(img, "ndim", 2) == 2:
            img = gray2rgb(img)                  # → H×W×3
        x = self.transform(img)                  # → 3×300×300

        y = torch.tensor([
            float(row[self.label_cols[0]]),
            float(row[self.label_cols[1]])
        ], dtype=torch.float32)
        return x, y

    
    def _infer_path_col(self, df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lname = {c: str(c).lower() for c in cols}
        for k in ("path","img_path","image_path","filename","image","file","filepath","file_path"):
            for c in cols:
                if lname[c] == k:
                    return c
        # heuristic: first object column that looks like a file path
        for c in cols:
            if df[c].dtype == object:
                v = str(df[c].iloc[0])
                if any(ext in v.lower() for ext in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]):
                    return c
        return cols[0]

    def _infer_label_cols(self, df: pd.DataFrame):
        if all(k in df.columns for k in ["crack","inactive"]):
            return ["crack","inactive"]
        others = [c for c in df.columns if c != self.path_col]
        crack = next((c for c in others if "crack" in str(c).lower()), None)
        inact = next((c for c in others if "inactive" in str(c).lower() or "inact" in str(c).lower()), None)
        if crack and inact:
            return [crack, inact]
        num_cols = [c for c in others if pd.api.types.is_numeric_dtype(df[c])]
        assert len(num_cols) >= 2, "Could not infer two label columns from data.csv"
        return num_cols[:2]

    def _make_bases(self):
        bases = []
        cwd = Path.cwd().resolve()
        bases.append(cwd)
        for _ in range(3):
            cwd = cwd.parent
            bases.append(cwd)
        here = Path(__file__).resolve()
        bases.append(here)
        bases.append(here.parent)

        # de-duplicate while preserving order
        seen, uniq = set(), []
        for b in bases:
            if b not in seen:
                seen.add(b)
                uniq.append(b)
        return uniq

    def _resolve_path(self, p):
        p = Path(str(p))
        if p.is_absolute() and p.exists():
            return p
        # Try candidate bases directly
        for base in self._bases:
            cand = (base / p)
            if cand.exists():
                return cand
        
        name = p.name
        for base in self._bases:
            try:
                for cand in base.rglob(name):
                    return cand
            except Exception:
                pass
        
        return p
