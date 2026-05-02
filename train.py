# train.py
# -*- coding: utf-8 -*-

import os, random, math, argparse
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder  

from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm  

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rand_bbox(W,H,lam):
    cut_rat=math.sqrt(1.-lam)
    cw=int(W*cut_rat); ch=int(H*cut_rat)
    cx=np.random.randint(W); cy=np.random.randint(H)
    x1=np.clip(cx-cw//2,0,W); y1=np.clip(cy-ch//2,0,H)
    x2=np.clip(cx+cw//2,0,W); y2=np.clip(cy+ch//2,0,H)
    return x1,y1,x2,y2

def apply_mixup(inputs, targets, alpha=0.2, num_classes=11, device="cuda"):
    lam=np.random.beta(alpha,alpha)
    b=inputs.size(0); idx=torch.randperm(b).to(device)
    mixed = lam*inputs + (1-lam)*inputs[idx]
    ya = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    yb = torch.nn.functional.one_hot(targets[idx], num_classes=num_classes).float()
    y = lam*ya + (1-lam)*yb
    return mixed, y

def apply_cutmix(inputs, targets, alpha=1.0, num_classes=11, device="cuda"):
    lam=np.random.beta(alpha,alpha)
    b,_,H,W=inputs.size(); idx=torch.randperm(b).to(device)
    x1,y1,x2,y2=rand_bbox(W,H,lam)
    out=inputs.clone(); out[:,:,y1:y2,x1:x2]=inputs[idx,:,y1:y2,x1:x2]
    lam = 1 - ((x2-x1)*(y2-y1)/(W*H))
    ya=torch.nn.functional.one_hot(targets,num_classes=num_classes).float()
    yb=torch.nn.functional.one_hot(targets[idx],num_classes=num_classes).float()
    y = lam*ya + (1-lam)*yb
    return out, y

def build_transforms(img_size=224):
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_t = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_t, eval_t

def build_model(num_classes, pretrained=True, label_smoothing=0.0):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, num_classes))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return model, criterion

def train_one_epoch(model, loader, optimizer, device, criterion,
                    scaler=None, num_classes=11,
                    use_mixup=False, mixup_alpha=0.2,
                    use_cutmix=False, cutmix_alpha=1.0):
    model.train()
    total=0; correct=0; run_loss=0.0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):  
        imgs=imgs.to(device); labels=labels.to(device)
        soft=None
        if use_mixup:
            imgs, soft = apply_mixup(imgs, labels, mixup_alpha, num_classes, device)
        elif use_cutmix:
            imgs, soft = apply_cutmix(imgs, labels, cutmix_alpha, num_classes, device)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits=model(imgs)
                loss = (-(soft*torch.log_softmax(logits,1)).sum(1).mean()) if soft is not None else criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits=model(imgs)
            loss = (-(soft*torch.log_softmax(logits,1)).sum(1).mean()) if soft is not None else criterion(logits, labels)
            loss.backward(); optimizer.step()
        run_loss += loss.item()*imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)
    return run_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total=0; correct=0; run_loss=0.0
    ys=[]; ps=[]
    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):  
        imgs=imgs.to(device); labels=labels.to(device)
        logits=model(imgs); loss=criterion(logits, labels)
        run_loss += loss.item()*imgs.size(0)
        pred=logits.argmax(1)
        correct += (pred==labels).sum().item()
        total += labels.size(0)
        ys.append(labels.cpu().numpy()); ps.append(pred.cpu().numpy())
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    return (run_loss/total if total>0 else 0.0), (correct/total if total>0 else 0.0), y, p

def plot_curves(epochs, train_loss, val_loss, train_acc, val_acc, val_f1, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # 1) Loss
    plt.figure(figsize=(7,5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend()
    p1=os.path.join(save_dir,"curve_loss.png"); plt.savefig(p1, dpi=180, bbox_inches="tight"); plt.close()

    # 2) Accuracy
    plt.figure(figsize=(7,5))
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves"); plt.legend()
    p2=os.path.join(save_dir,"curve_accuracy.png"); plt.savefig(p2, dpi=180, bbox_inches="tight"); plt.close()

    # 3) F1 (macro on val)
    plt.figure(figsize=(7,5))
    plt.plot(epochs, val_f1, label="val_f1 (macro)")
    plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title("Validation F1 (macro)"); plt.legend()
    p3=os.path.join(save_dir,"curve_f1.png"); plt.savefig(p3, dpi=180, bbox_inches="tight"); plt.close()
    print(f"[INFO] Saved curves:\n  {p1}\n  {p2}\n  {p3}")

def plot_confusion_matrix(cm, classes, normalize=True, title="Confusion Matrix", cmap="Blues", save_path=None):
    if normalize: cm = cm.astype("float")/cm.sum(axis=1, keepdims=True).clip(min=1e-12)
    plt.figure(figsize=(10,8)); plt.imshow(cm, interpolation="nearest", cmap=cmap); plt.title(title)
    plt.colorbar(); ticks=np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right"); plt.yticks(ticks, classes)
    fmt=".2f" if normalize else "d"; thr=cm.max()/2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,format(cm[i,j],fmt),ha="center",va="center",
                     color="white" if cm[i,j]>thr else "black", fontsize=9)
    plt.tight_layout(); plt.ylabel("True"); plt.xlabel("Pred")
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)  
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--use_mixup", action="store_true")
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--use_cutmix", action="store_true")
    ap.add_argument("--cutmix_alpha", type=float, default=1.0)
    ap.add_argument("--save_dir", type=str, default="./outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    if device=="cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")

    # ===== 使用 ImageFolder 載入資料 =====
    t_train, t_eval = build_transforms(args.img_size)

    train_dir = os.path.join(args.data_root, 'train')
    val_dir   = os.path.join(args.data_root, 'val')
    test_dir  = os.path.join(args.data_root, 'test')

    train_ds = ImageFolder(train_dir, transform=t_train)
    val_ds   = ImageFolder(val_dir,   transform=t_eval)
    test_ds  = ImageFolder(test_dir,  transform=t_eval)

    classes = train_ds.classes
    num_classes = len(classes)
    print(f"[INFO] Classes: {classes}")
    print(f"[INFO] Images => Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader=DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device=="cuda"))
    val_loader  =DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device=="cuda"))
    test_loader =DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device=="cuda"))

    model, criterion = build_model(num_classes, pretrained=args.pretrained, label_smoothing=args.label_smoothing)
    model=model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device=="cuda") else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_val=0.0; best_ckpt=os.path.join(args.save_dir,"best_resnet18.pth")

    # ===== 這裡開始紀錄曲線 =====
    ep_idx=[]; tr_losses=[]; tr_accs=[]; va_losses=[]; va_accs=[]; va_f1s=[]

    for epoch in range(1, args.epochs+1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, device, criterion,
                                 scaler=scaler, num_classes=num_classes,
                                 use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
                                 use_cutmix=args.use_cutmix, cutmix_alpha=args.cutmix_alpha)
        vl, va, y_true, y_pred = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        f1 = f1_score(y_true, y_pred, average="macro") if len(y_true)>0 else 0.0

        ep_idx.append(epoch); tr_losses.append(tl); tr_accs.append(ta); va_losses.append(vl); va_accs.append(va); va_f1s.append(f1)

        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] "
              f"train_loss={tl:.4f} train_acc={ta:.4f} | "
              f"val_loss={vl:.4f} val_acc={va:.4f} val_f1(macro)={f1:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if va > best_val:
            best_val = va
            torch.save({"model_state": model.state_dict(), "classes": classes, "args": vars(args)}, best_ckpt)
            print(f"  -> [BEST] Saved {best_ckpt} (val_acc={va:.4f})")

    plot_curves(ep_idx, tr_losses, va_losses, tr_accs, va_accs, va_f1s, args.save_dir)

    # 最後再存 mymodel.pth
    torch.save(model.state_dict(), os.path.join(args.save_dir, "mymodel.pth"))
    print(f"[INFO] Saved final model to {os.path.join(args.save_dir, 'mymodel.pth')}")

    if os.path.exists(best_ckpt):
        ckpt=torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] Loaded best checkpoint from {best_ckpt}")

    tl, ta, y_t, y_p = evaluate(model, test_loader, device, criterion)
    print(f"[TEST] loss={tl:.4f} acc={ta:.4f}")
    cm = confusion_matrix(y_t, y_p, labels=list(range(num_classes)))

    print("\nClassification Report (TEST):")
    print(classification_report(y_t, y_p, target_names=classes, digits=4))

    cm1=os.path.join(args.save_dir,"confusion_matrix.png")
    plot_confusion_matrix(cm, classes, normalize=True, title="Normalized Confusion Matrix", save_path=cm1)
    cm2=os.path.join(args.save_dir,"confusion_matrix_raw.png")
    plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix (Counts)", save_path=cm2)
    print(f"[INFO] Saved confusion matrices:\n  {cm1}\n  {cm2}\n[DONE]")

if __name__=="__main__":
    main()