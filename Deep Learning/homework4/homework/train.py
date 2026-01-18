import torch
import numpy as np
# import cv2
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision.ops.ciou_loss import complete_box_iou_loss
from torchvision.transforms.functional import to_pil_image

from .models import Detector, save_model, extract_peak
from .utils import load_detection_data
from . import dense_transforms

# try:
#     from .models import Detector, save_model, extract_peak
#     from .utils import load_detection_data
#     from . import dense_transforms
# except:
#     from models import Detector, save_model, extract_peak
#     from utils import load_detection_data
#     import dense_transforms
import torch.utils.tensorboard as tb

from tqdm import tqdm

COLORS = {0: (0, 0, 200), 1: (200, 0, 0), 2: (0, 200, 0)}

def train(args):
    from os import path
    model = Detector()
    if torch.cuda.is_available():
        model = model.to('cuda')
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    loss_mse = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.1)
    optimizer.zero_grad()
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400], gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000)

    train_transforms = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        dense_transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        dense_transforms.RandomCropResize(p=0.5),
        dense_transforms.ToTensor()
    ])
    train_dataset = load_detection_data(args.train_dataset, num_workers=6, batch_size=args.bs, transform=train_transforms, train=True)
    val_dataset = load_detection_data(args.val_dataset, num_workers=6, batch_size=args.bs, train=True)

    for epoch in tqdm(range(args.num_epochs)):
        for n_step, (imgs, hms, sizes) in enumerate(train_dataset):
            imgs, hms, sizes = imgs.to('cuda'), hms.to('cuda'), sizes.to('cuda')
            out = model(imgs)
            out_hms, out_sizes = torch.split(out, [3, 2], dim=1)
            mask = (hms[:, 0, ...] == 1) | (hms[:, 1, ...] == 1) | (hms[:, 2, ...] == 1)
            l_bbox = F.smooth_l1_loss(out_sizes, sizes, reduction='none')
            l_bbox = (l_bbox[:, 0, ...][mask] + l_bbox[:, 1, ...][mask]).sum()
            l_ce = sigmoid_focal_loss(out_hms, hms, reduction="sum", gamma=args.gamma, alpha=args.alpha)
            l_mse = loss_mse(torch.sigmoid(out_hms), hms)
            l = args.ce_weight * l_ce + args.mse_weight * l_mse + args.bbox_weight * l_bbox
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_logger.add_scalar("loss", l.item(), global_step=epoch * len(train_dataset) + n_step)
            scheduler.step(epoch + n_step / len(train_dataset))

        # validation
        if epoch % args.val_freq == 0:
            model.eval()
            losses_val = []
            print("Validation...")
            for n_step_val, (imgs, hms, sizes) in enumerate(val_dataset):
                imgs, hms, sizes = imgs.to('cuda'), hms.to('cuda'), sizes.to('cuda')
                out = model(imgs)
                out_hms, out_sizes = torch.split(out, [3, 2], dim=1)
                mask = (hms[:, 0, ...] == 1) | (hms[:, 1, ...] == 1) | (hms[:, 2, ...] == 1)
                l_bbox = F.smooth_l1_loss(out_sizes, sizes, reduction='none')
                l_bbox = (l_bbox[:, 0, ...][mask] + l_bbox[:, 1, ...][mask]).sum()
                l_ce = sigmoid_focal_loss(out_hms, hms, reduction="sum", gamma=args.gamma, alpha=args.alpha)
                l_mse = loss_mse(torch.sigmoid(out_hms), hms)
                l = args.ce_weight * l_ce + args.mse_weight * l_mse + args.bbox_weight * l_bbox
                losses_val.append(l.item())
                out_hms = torch.sigmoid(out_hms)
                # # visualization
                # if n_step_val == 0:
                #     images = draw_detections(imgs, out_hms)
                #     images = torch.permute(torch.tensor(images), (0, 3, 1, 2)) / 255.0
                #     valid_logger.add_images('detections_pred', images, epoch * len(train_dataset) + n_step)
                #     images = draw_detections(imgs, hms)
                #     images = torch.permute(torch.tensor(images), (0, 3, 1, 2)) / 255.0
                #     valid_logger.add_images('detections_gt', images, epoch * len(train_dataset) + n_step)
                #     valid_logger.add_images('heatmaps_pred', out_hms, epoch * len(train_dataset) + n_step)
                #     valid_logger.add_images('heatmaps_gt', hms, epoch * len(train_dataset) + n_step)
            valid_logger.add_scalar("loss", np.mean(losses_val), global_step=epoch * len(train_dataset) + n_step)
            print(f'Loss valid {np.mean(losses_val):.2f}')
            model.train()
        
        
        
        save_model(model)

# def draw_detections(imgs, heatmaps):
#     out = []
#     for k, img in enumerate(imgs):
#         img = img.detach().cpu()
#         img = to_pil_image(img)
#         img = np.array(img)
#         hm = heatmaps[k]
#         for nc in range(3):
#             hm_c = hm[nc]
#             peaks = extract_peak(hm_c, min_score=0.4, max_det=30)
#             for p in peaks:
#                 y, x = p[1:3]
#                 img = cv2.rectangle(img, (y-1, x-1), (y+1, x+1), COLORS[nc], -1)
#         out.append(img[None, ...])
#     return np.concatenate(out)



def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default="log")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--train_dataset", type=str, default="dense_data/train", help="path to the training data")
    parser.add_argument("--val_dataset", type=str, default="dense_data/valid", help="path to the validation data")
    parser.add_argument("--bs", type=int, default=16, help="training batch size")
    parser.add_argument("--mse_weight", type=float, default=0.1, help="MSE loss weight")
    parser.add_argument("--ce_weight", type=float, default=1.0, help="CE loss weight")
    parser.add_argument("--bbox_weight", type=float, default=2.0, help="bbox loss weight")
    parser.add_argument("--alpha", type=float, default=0.99, help="focal loss alpha")
    parser.add_argument("--gamma", type=float, default=2, help="focal loss gamma")
    parser.add_argument("--val_freq", type=int, default=5, help="Validation frequency")
    args = parser.parse_args()
    train(args)
