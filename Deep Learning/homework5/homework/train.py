from .planner import Planner, save_model
# try:
#     from .planner import Planner, save_model
# except:
#     from planner import Planner, save_model 
import torch
from torch.optim import lr_scheduler
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
# try:
#     from .utils import load_data
#     from . import dense_transforms
# except:
#     from utils import load_data
#     import dense_transforms

# from utils import PyTux
# from controller import control

from tqdm import tqdm

def train(args):
    from os import path
    model = Planner()
    if torch.cuda.is_available():
        model = model.to('cuda')
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    optimizer.zero_grad()
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[120, 140], gamma=0.1)
    # loss_mse = torch.nn.MSELoss(reduction="sum")
    loss_mse = torch.nn.HuberLoss(reduction="sum", delta=1)

    train_transforms = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(),
        # dense_transforms.RandomVerticalFlip(),
        dense_transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        dense_transforms.RandomShift(p=0.25),
        dense_transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
        dense_transforms.RandomInvert(p=0),
        dense_transforms.ToTensor(),
        # dense_transforms.RandomErasing(p=0.25)
    ])

    train_dataset = load_data(args.train_dataset, num_workers=6, batch_size=args.bs, transform=train_transforms)
    val_dataset = load_data(args.val_dataset, num_workers=6, batch_size=args.bs)

    # pytux = PyTux()
    # max_far = -1
    for epoch in tqdm(range(args.num_epochs)):
        for n_step, (imgs, aim_points) in enumerate(train_dataset):
            imgs, aim_points = imgs.to('cuda'), aim_points.to('cuda')
            out = model(imgs)
            loss = loss_mse(out, aim_points)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_logger.add_scalar("loss", loss.item(), global_step=epoch * len(train_dataset) + n_step)
            scheduler.step(epoch + n_step / len(train_dataset))
        
        # validation
        # if epoch % args.val_freq == 0:
        #     model = model.to('cpu')
        #     model.eval()
        #     steps, how_far = pytux.rollout("cocoa_temple", control, planner=model, max_frames=1000, verbose=False)
        #     print(f"Epoch {epoch} : {how_far}")
        #     if how_far > max_far:
        #         save_model(model)
        #         max_far = how_far        
        #     # losses_val = []
        #     # print("Validation...")
        #     # for n_step_val, (imgs, aim_points) in enumerate(val_dataset):
        #     #     imgs, aim_points = imgs.to('cuda'), aim_points.to('cuda')
        #     #     out = model(imgs)
        #     #     loss = loss_mse(out, aim_points)
        #     #     losses_val.append(loss.item())
        #     # valid_logger.add_scalar("loss", np.mean(losses_val), global_step=epoch * len(train_dataset) + n_step)
        #     # print(f'Loss valid {np.mean(losses_val):.6f}')
        #     model = model.to('cuda')
        #     model.train()
        
        # scheduler.step()

        # save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default="log")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--train_dataset", type=str, default="drive_data", help="path to the training data")
    parser.add_argument("--val_dataset", type=str, default="drive_data_new/valid", help="path to the validation data")
    parser.add_argument("--bs", type=int, default=16, help="training batch size")
    parser.add_argument("--val_freq", type=int, default=5, help="Validation frequency")
    args = parser.parse_args()
    train(args)
