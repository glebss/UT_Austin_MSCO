from models import CNNClassifier, save_model
from utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
from torch.optim import lr_scheduler
import torch.utils.tensorboard as tb


from tqdm import tqdm


def train(args):
    from os import path
    model = CNNClassifier()
    if torch.cuda.is_available():
        model = model.to('cuda')
    train_logger, valid_logger = None, None
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0)
    optimizer.zero_grad()
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1400], gamma=0.25)
    train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(64),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[87.81784583, 84.4062965, 82.47563152], std=[44.56715667, 42.31447801, 46.4486642])
            ])
    val_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[87.81784583, 84.4062965, 82.47563152], std=[44.56715667, 42.31447801, 46.4486642])
        ]
    )
    train_dataset = load_data(args.train_dataset, num_workers=4, batch_size=args.bs, transforms=train_transforms)
    val_dataset = load_data(args.val_dataset, num_workers=4, batch_size=args.bs, transforms=val_transforms)
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    best_acc = None
    for epoch in range(args.num_epochs):

        print(f"epoch {epoch}")
        train_accuracies = []
        for n_step, (imgs, labels) in enumerate(tqdm(train_dataset)):
            imgs, labels = imgs.to('cuda'), labels.to('cuda')
            out = model(imgs)
            pred_labels = torch.softmax(out, dim=-1).argmax(-1)
            training_acc = (pred_labels == labels).sum() / len(pred_labels)
            train_accuracies.append(training_acc)
            l = loss(out, labels)
            train_logger.add_scalar("loss", l.item(), global_step=epoch * len(train_dataset) + n_step)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_acc = torch.mean(torch.tensor(train_accuracies)).item() * 100
        train_logger.add_scalar("accuracy", train_acc, global_step=epoch * len(train_dataset) + n_step)
        
        # validation step
        model.eval()
        val_accs = []
        for (imgs, labels) in tqdm(val_dataset):
            imgs, labels = imgs.to('cuda'), labels.to('cuda')
            out = model(imgs)
            pred_labels = torch.softmax(out, dim=-1).argmax(-1)
            valid_acc = (pred_labels == labels).sum() / len(pred_labels)
            val_accs.append(valid_acc)
        val_acc = torch.mean(torch.tensor(val_accs)).item() * 100
        valid_logger.add_scalar("accuracy", val_acc, global_step=epoch * len(train_dataset) + n_step)
        scheduler.step()
        print(f"epoch {epoch}, train_acc {train_acc:.2f}, val acc {val_acc:.2f}")
        model.train()
        if best_acc is None or val_acc > best_acc:
            best_acc = val_acc
            save_model(model)
    print(best_acc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default="log")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--train_dataset", type=str, default="data/train", help="path to the training data")
    parser.add_argument("--val_dataset", type=str, default="data/valid", help="path to the validation data")
    parser.add_argument("--bs", type=int, default=64, help="training batch size")

    args = parser.parse_args()
    train(args)
