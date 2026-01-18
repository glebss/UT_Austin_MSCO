import torch
torch.set_warn_always(True)
import numpy as np
from torch.optim import lr_scheduler

from models import FCN, save_model
from utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
try:
    from . import dense_transforms
except:
    import dense_transforms

import torch.utils.tensorboard as tb

from tqdm import tqdm


def train(args):
    from os import path
    model = FCN()
    if torch.cuda.is_available():
        model = model.to('cuda')
    train_logger, valid_logger = None, None
    weights_loss = torch.tensor([1.0 / w for w in DENSE_CLASS_DISTRIBUTION])
    if torch.cuda.is_available():
        weights_loss = weights_loss.to('cuda')
    loss = torch.nn.CrossEntropyLoss(weight=weights_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    optimizer.zero_grad()
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.25)
    train_transforms = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # dense_transforms.RandomResizedCrop((96, 128)),
        dense_transforms.ToTensor()
    ])

    train_dataset = load_dense_data(args.train_dataset, num_workers=6, batch_size=args.bs, transform=train_transforms)
    val_dataset = load_dense_data(args.val_dataset, num_workers=6, batch_size=args.bs)
    best_global_acc = None
    best_iou = 0.0
    for epoch in tqdm(range(args.num_epochs)):
        # print(f"epoch {epoch}")
        conf_matrix = ConfusionMatrix()
        for n_step, (imgs, labels) in enumerate(train_dataset):
            imgs, labels = imgs.to('cuda'), labels.to('cuda')
            out = model(imgs)
            labels = labels.flatten()
            out = out.permute((0, 2, 3, 1))
            out = out.flatten(end_dim=2)
            pred_labels = torch.softmax(out, dim=-1).argmax(-1)
            l = loss(out, labels)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            conf_matrix.add(pred_labels, labels)
            train_logger.add_scalar("loss", l.item(), global_step=epoch * len(train_dataset) + n_step)
        
        train_logger.add_scalar("iou", conf_matrix.iou.item(), global_step=epoch * len(train_dataset) + n_step)
        train_logger.add_scalar("global_accuracy", conf_matrix.global_accuracy.item(), global_step=epoch * len(train_dataset) + n_step)
        train_logger.add_scalar("average_accuracy", conf_matrix.average_accuracy.item(), global_step=epoch * len(train_dataset) + n_step)
        for nc in range(5):
            train_logger.add_scalar(f"class_iou_{nc}", conf_matrix.class_iou[nc].item(), global_step=epoch * len(train_dataset) + n_step)
            train_logger.add_scalar(f"class_accuracy_{nc}", conf_matrix.class_accuracy[nc].item(), global_step=epoch * len(train_dataset) + n_step)
        
        # validation
        model.eval()
        conf_matrix_val = ConfusionMatrix()
        print("Validation...")
        losses_val = []
        for n_step, (imgs, labels) in enumerate(val_dataset):
            imgs, labels = imgs.to('cuda'), labels.to('cuda')
            out = model(imgs)
            labels = labels.flatten()
            out = out.permute((0, 2, 3, 1))
            out = out.flatten(end_dim=2)
            l = loss(out, labels)
            l = l.detach().cpu().item()
            losses_val.append(l)
            pred_labels = torch.softmax(out, dim=-1).argmax(-1)
            conf_matrix_val.add(pred_labels.detach().cpu(), labels.detach().cpu())
        scheduler.step()

        valid_logger.add_scalar("loss", np.mean(losses_val), global_step=epoch * len(train_dataset) + n_step)
        valid_logger.add_scalar("iou", conf_matrix_val.iou.item(), global_step=epoch * len(train_dataset) + n_step)
        valid_logger.add_scalar("global_accuracy", conf_matrix_val.global_accuracy.item(), global_step=epoch * len(train_dataset) + n_step)
        valid_logger.add_scalar("average_accuracy", conf_matrix_val.average_accuracy.item(), global_step=epoch * len(train_dataset) + n_step)
        for nc in range(5):
            valid_logger.add_scalar(f"class_iou_{nc}", conf_matrix_val.class_iou[nc].item(), global_step=epoch * len(train_dataset) + n_step)
            valid_logger.add_scalar(f"class_accuracy_{nc}", conf_matrix_val.class_accuracy[nc].item(), global_step=epoch * len(train_dataset) + n_step)
        model.train()

        if best_global_acc is None or conf_matrix_val.global_accuracy.item() > best_global_acc:
            best_global_acc = conf_matrix_val.global_accuracy.item()
            best_iou = conf_matrix_val.iou.item()
            save_model(model)
    print(f"best acc {best_global_acc:.4f}, best iou {best_iou:.4f}")

def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default="log")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--train_dataset", type=str, default="dense_data/train", help="path to the training data")
    parser.add_argument("--val_dataset", type=str, default="dense_data/valid", help="path to the validation data")
    parser.add_argument("--bs", type=int, default=16, help="training batch size")
    args = parser.parse_args()
    train(args)
