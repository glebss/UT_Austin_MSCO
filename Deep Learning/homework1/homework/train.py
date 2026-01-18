from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data

import torch


def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    loss = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    train_dataset = load_data(args.train_dataset, num_workers=1, batch_size=args.bs)
    val_dataset = load_data(args.val_dataset, num_workers=1, batch_size=args.bs)
    model.train()

    for epoch in range(args.num_epochs):

        print(f"epoch {epoch}")
        for imgs, labels in train_dataset:
            out = model(imgs)
            l = loss(out, labels)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        
        # validation step
        # model.eval()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument("--lr", type=float, default=0.02, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--train_dataset", type=str, help="path to the training data")
    parser.add_argument("--val_dataset", type=str, help="path to the validation data")
    parser.add_argument("--bs", type=int, default=16, help="training batch size")

    args = parser.parse_args()
    train(args)
