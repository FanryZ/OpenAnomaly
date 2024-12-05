import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import logging

import open_clip
from dataset import VisaDataset, MVTecDataset
from model import LinearLayer, UNet, LUNet
from loss import FocalLoss, BinaryDiceLoss, F1Loss
from prompt_ensemble import encode_text_with_prompt_ensemble
from segment_anything import SamPredictor, sam_model_registry
from sam_reg import sam_mask_regression
import cv2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor2image(img):
    img = img.cpu().numpy()
    img = np.transpose(img, [1, 2, 0]) * np.array([[[0.26862954, 0.26130258, 0.27577711]]]) + \
            np.array([[[0.48145466, 0.4578275, 0.40821073]]])
    img = cv2.cvtColor((img * 255.).astype("uint8"), cv2.COLOR_RGB2BGR)
    return img

def train(args):
    # configs
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log

    # model configs
    features_list = args.features_list
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    # clip model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    # datasets
    if args.dataset == 'mvtec':
        train_data = MVTecDataset(root=args.train_data_path, transform=preprocess, target_transform=transform,
                                  aug_rate=args.aug_rate)
    else:
        train_data = VisaDataset(root=args.train_data_path, transform=preprocess, target_transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # linear layer
    trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                                  len(args.features_list), args.model).to(device)
    if args.pretrained_linear:
        state_dict = torch.load(args.pretrained_linear)["trainable_linearlayer"]
        trainable_layer.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(list(trainable_layer.parameters()), lr=learning_rate, betas=(0.5, 0.999))
        
    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_ce = FocalLoss(gamma=1)
    loss_f1 = F1Loss(beta=2)

    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        obj_list = train_data.get_cls_names()
        text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)

    for epoch in range(epochs):
        print("Training epoch {}.".format(epoch))
        loss_list = []
        idx = 0
        pbar = tqdm(train_dataloader)
        for items in pbar:
            idx += 1
            image = items['img'].to(device)
            cls_name = items['cls_name']
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    image_features, patch_tokens = model.encode_image(image, features_list)
                    text_features = []
                    for cls in cls_name:
                        text_features.append(text_prompts[cls])
                    text_features = torch.stack(text_features, dim=0)

                # pixel level
                patch_tokens = trainable_layer(patch_tokens)
                anomaly_maps = []
                anomaly_map_fusion = torch.zeros_like(image[:, :1, :, :], device=device)
                for layer in range(len(patch_tokens)):
                    patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * patch_tokens[layer] @ text_features)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=image_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)

                visual_prompts = torch.tensor(np.concatenate(visual_prompts, axis=0), device=device)
                input_fusion = torch.cat([anomaly_map_fusion, visual_prompts], dim=1).detach().float() if args.freeze_linear \
                    else torch.cat([anomaly_map_fusion, visual_prompts], dim=1).float()
                # input_fusion = torch.cat([anomaly_map_fusion, visual_prompts], dim=1).detach()
                fused_pred = fusion_net(input_fusion)

            # losses
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0
            loss, loss1 = 0, 0
            loss2 = torch.tensor(0.0, device=device)
            # if not args.freeze_linear:
            for num in range(len(anomaly_maps)):
                loss1 += loss_focal(anomaly_maps[num], gt)
                loss1 += loss_dice(anomaly_maps[num][:, 1, :, :], gt)
            if args.freeze_linear:
                loss1 = loss1.detach()
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            pbar.set_description("loss: {:.3f}; loss1: {:.3f}; loss2: {:.3f}.".format(loss.item(), loss1.item() / 4, loss2.item()))

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({'trainable_linearlayer': trainable_layer.state_dict()}, ckp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./exps/vit_large_14_518', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=200, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="image size")
    parser.add_argument("--print_freq", type=int, default=30, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=3, help="save frequency")
    
    parser.add_argument("--train_fusion", action="store_true")
    parser.add_argument("--freeze_linear", action="store_true")
    parser.add_argument("--light_network", action="store_true")
    parser.add_argument("--pretrained_linear", type=str, default="")
    args = parser.parse_args()

    setup_seed(111)
    train(args)

