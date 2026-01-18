# from .models import CNNClassifier, save_model
# from .utils import accuracy, load_data
from models import CNNClassifier, save_model
from utils import accuracy, load_data
import torch
import numpy as np
import torch.utils.tensorboard as tb

from tqdm import tqdm


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    train_dataset = load_data(args.train_dataset, num_workers=1, batch_size=args.bs)
    val_dataset = load_data(args.val_dataset, num_workers=1, batch_size=args.bs)
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    for epoch in range(args.num_epochs):

        print(f"epoch {epoch}")
        train_accuracies = []
        for n_step, (imgs, labels) in enumerate(tqdm(train_dataset)):
            out = model(imgs)
            pred_labels = torch.softmax(out, dim=-1).argmax(-1)
            training_acc = (pred_labels == labels).sum() / len(pred_labels)
            train_accuracies.append(training_acc)
            l = loss(out, labels)
            train_logger.add_scalar("train/loss", l.item(), global_step=epoch * len(train_dataset) + n_step)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_acc = np.mean(train_accuracies) * 100
        train_logger.add_scalar("train/accuracy", train_acc, global_step=epoch * len(train_dataset) + n_step)
        
        # validation step
        model.eval()
        val_accs = []
        for (imgs, labels) in tqdm(val_dataset):
            out = model(imgs)
            pred_labels = torch.softmax(out, dim=-1).argmax(-1)
            valid_acc = (pred_labels == labels).sum() / len(pred_labels)
            val_accs.append(valid_acc)
        val_acc = np.mean(val_accs) * 100
        valid_logger.add_scalar("valid/accuracy", val_acc, global_step=epoch * len(train_dataset) + n_step)
        print(f"epoch {epoch}, train_acc {train_acc:.2f}, val acc {val_acc:.2f}")
        model.train()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default="log")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--train_dataset", type=str, default="data/train", help="path to the training data")
    parser.add_argument("--val_dataset", type=str, default="data/valid", help="path to the validation data")
    parser.add_argument("--bs", type=int, default=32, help="training batch size")
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
