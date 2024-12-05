import os
import cv2
import json
import torch
import logging
import argparse
import numpy as np
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_curve, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise

import open_clip
from few_shot import memory
from model import LinearLayer, UNet, LUNet
from dataset import VisaDataset, MVTecDataset
from prompt_ensemble import encode_text_with_prompt_ensemble_cls, encode_text_with_prompt_ensemble_seg
from segment_anything import SamPredictor, sam_model_registry
from sam_reg import sam_mask_regression, visual_saliency_calculation, single_object_similarity, mask_merge, mask_refine
from open_scene import OpenDataset
from utils import *

from tqdm import tqdm
from multiprocessing import Pool


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    few_shot_features = args.few_shot_features
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt_path = os.path.join(save_path, 'log.txt')

    # clip
    # if args.object_localization:
    #     grdino_model = load_grdino_model(args.grdino_config, args.grdino_checkpoint).to(device)
    #     _input_image = _input_image.to(device)
    #     boxes_filt, pred_phrases = get_grounding_output(grdino_model, _input_image, "dail", box_threshold=0.3, text_threshold=0.25, token_spans=None)
    #     _input_image = F.interpolate(_input_image.unsqueeze(0), (518, 518))
    
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    
    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
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
        if args.mode == 'zero_shot' and (arg == 'k_shot' or arg == 'few_shot_features'):
            continue
        logger.info(f'{arg}: {getattr(args, arg)}')

    # seg
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
    linearlayer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                    len(features_list), args.model).to(device)
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])

    # dataset
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    if dataset_name == 'mvtec':
        test_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                        aug_rate=-1, mode='test')
    elif dataset_name == 'visa':
        test_data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='test')
    elif dataset_name == 'open':
        test_data = OpenDataset(root=dataset_dir, transform=preprocess, target_transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.get_cls_names()

    # few shot
    if args.mode == 'few_shot':
        mem_features = memory(args.model, model, obj_list, dataset_dir, save_path, preprocess, transform,
                        args.k_shot, few_shot_features, dataset_name, device)

    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_prompts_seg = encode_text_with_prompt_ensemble_seg(model, obj_list, tokenizer, device)
        text_prompts_cls, text_prompts_cls_list = \
            encode_text_with_prompt_ensemble_cls(model, obj_list, tokenizer, device)

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    
    if args.sam_prompt:
        sam = sam_model_registry["vit_l"](checkpoint=args.sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
    test_set = set()
    
    for items in tqdm(test_dataloader):
        tag = (items["cls_name"][0], items["img_path"][0].split("/")[-2])
        if args.quick_test and tag in test_set:
            continue
        test_set.add(tag)
                
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].append(items['anomaly'].item())
            
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens_clip = model.encode_image(image, features_list)  
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features, text_features_image, text_features_cls = [], [], []
            # probability of anormaly, calculated by the sim of text-image encoding
            for cls in cls_name:
                text_features.append(text_prompts_seg[cls])
                text_features_cls.append(text_prompts_cls[cls])
            text_features = torch.stack(text_features, dim=0)            
            text_features_cls = torch.stack(text_features_cls, dim=0)
            text_probs = (100 * image_features @ text_features_cls[0]).softmax(dim=-1)
            # sample
            patch_tokens = linearlayer(patch_tokens_clip)
            pa_maps = []
            for idf, features in enumerate(patch_tokens):
                features /= features.norm(dim=-1, keepdim=True)
                patch_size = model_configs["vision_cfg"]["patch_size"]
                patch_h, patch_w = args.image_size // patch_size, args.image_size // patch_size
                pa_map = single_object_similarity([patch_h, patch_w], features)
                pa_maps.append(pa_map)
            pa_maps = np.array(pa_maps).astype("float32")
            pa_maps_sum = np.sum(pa_maps, axis=0)
            
            results['pr_sp'].append(args.alpha_class * np.mean(pa_maps_sum) + text_probs[0][1].cpu().item())
            
            if args.image_only:
                results['anomaly_maps'].append(np.zeros([1, args.image_size, args.image_size]))
                continue
            # pixel
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            anomaly_map = np.sum(anomaly_maps, axis=0)
            
            # visualization
            path = items['img_path']
            cls = path[0].split('/')[-2]
            filename = path[0].split('/')[-1]
            vis = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
            mask = normalize(anomaly_map[0])
            vis = apply_ad_scoremap(vis, mask)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
            save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls)
            if not os.path.exists(save_vis):
                os.makedirs(save_vis)
            cv2.imwrite(os.path.join(save_vis, filename), vis)

            if args.sam_prompt:
                input_image = tensor2image(image)
                predictor.set_image(input_image)
                masks, centers, boxes = sam_mask_regression(
                    predictor, 
                    anomaly_map, 
                    args.mask_type, 
                    use_boxes=args.use_boxes, 
                    max_masks=args.max_masks, 
                    area_threshold_ratio=-1,
                    region_mask_threshold=args.region_mask_threshold, 
                    anomaly_threshold=args.anomaly_threshold
                )
                
                if masks.size > 0:
                    n, h, w = masks.shape
                    mask_anomaly_scores = 0
                    sam_mask = mask_merge(masks)
                else:
                    sam_mask = np.zeros_like(anomaly_map[0])
                if len(boxes) > 0:
                    pa_map = cv2.resize(pa_maps_sum, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR) 
                    pa_map = pa_map / pa_map.max() - args.beta_shift
                    sam_mask = mask_refine(sam_mask, pa_map, beta=args.beta)
                    anomaly_map = anomaly_map + args.alpha * sam_mask[np.newaxis, :, :] + args.alpha_cross * sam_mask[np.newaxis, :, :] * anomaly_map

                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
                vis_sos_map = 255. * (pa_maps_sum / pa_maps_sum.max())
                vis_sos_map = cv2.applyColorMap(vis_sos_map.astype("uint8"), cv2.COLORMAP_BONE)
                # cv2.imwrite(os.path.join(save_vis, "pa_" + filename), vis_sos_map)
                
                if len(centers) > 0:
                    prompt = np.array(centers)
                    input_image[prompt[:, 1], prompt[:, 0]] = 255
                # cv2.imwrite(os.path.join(save_vis, "prompt_" + filename), input_image)

                image_boxes = tensor2image(image)
                for box in boxes:
                    x, y, w, h = box
                    cv2.rectangle(image_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # cv2.imwrite(os.path.join(save_vis, "box_" + filename), image_boxes)
                
                vis = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
                mask = normalize(anomaly_map[0])
                vis = apply_ad_scoremap(vis, mask)
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
                save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls)
                # cv2.imwrite(os.path.join(save_vis, "sam_" + filename), vis)
                
            results['anomaly_maps'].append(anomaly_map)
            np.save(os.path.join(save_vis, filename.split(".")[0] + ".npy"), anomaly_map[0])
            
            
    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    print("Evaluating metrics: ")

    obj_results = []
    for obj in tqdm(obj_list):
    # def cal_metric(obj):
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        pr_sp_tmp = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
                pr_sp_tmp.append(np.max(results['anomaly_maps'][idxes]))
                gt_sp.append(results['gt_sp'][idxes])
                pr_sp.append(results['pr_sp'][idxes])
        obj_results.append([gt_px, pr_px, pr_sp_tmp, gt_sp, pr_sp, args.image_only])
    
    pool = Pool()
    results = pool.map(cal_metric, obj_results) 
    for obj, result in zip(obj_list, results):
        table, auroc_sp, auroc_px, f1_sp, f1_px, aupro, ap_sp, ap_px = result
        table = [obj] + table
        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_ls.append(aupro)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)

    # logger
    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_px_ls) * 100, decimals=1)), str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(aupro_ls) * 100, decimals=1)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_sp_ls) * 100, decimals=1)), str(np.round(np.mean(ap_sp_ls) * 100, decimals=1))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
                                'f1_sp', 'ap_sp'], tablefmt="pipe")
    logger.info("\n%s", results)
    print("{}\n{}".format(args.checkpoint_path, results))


def cal_metric(inputs):
    gt_px, pr_px, pr_sp_tmp, gt_sp, pr_sp, image_only = inputs
    gt_px = np.array(gt_px)
    gt_sp = np.array(gt_sp)
    pr_px = np.array(pr_px)
    pr_sp = np.array(pr_sp)
    if args.mode == 'few_shot':
        pr_sp_tmp = np.array(pr_sp_tmp)
        pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min())
        pr_sp = 0.5 * (pr_sp + pr_sp_tmp)

    ap_sp = average_precision_score(gt_sp, pr_sp)
    auroc_sp = roc_auc_score(gt_sp, pr_sp)
    # f1_sp
    precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
    
    if image_only:
        auroc_px, ap_px, f1_px, aupro = 0.0, 0.0, 0.0, 0.0 
    else:
        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro = cal_pro_score(gt_px, pr_px)

    table = []
    table.append(str(np.round(auroc_px * 100, decimals=1)))
    table.append(str(np.round(f1_px * 100, decimals=1)))
    table.append(str(np.round(ap_px * 100, decimals=1)))
    table.append(str(np.round(aupro * 100, decimals=1)))
    table.append(str(np.round(auroc_sp * 100, decimals=1)))
    table.append(str(np.round(f1_sp * 100, decimals=1)))
    table.append(str(np.round(ap_sp * 100, decimals=1)))
    return table, auroc_sp, auroc_px, f1_sp, f1_px, aupro, ap_sp, ap_px


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/tiaoshi', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9], help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")
    # few shot
    parser.add_argument("--k_shot", type=int, default=10, help="e.g., 10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    
    # SAM Settings
    parser.add_argument("--sam_prompt", action="store_true")
    parser.add_argument("--use_boxes", action="store_true")
    parser.add_argument("--mask_type", type=str, default="softmax", choices=["softmax", "max", "intersect"])

    parser.add_argument("--anomaly_fuse", action="store_true")
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--light_weight", action="store_true")
    parser.add_argument("--max_masks", type=int, default=-1)
    parser.add_argument("--region_mask_threshold", type=float, default=-1.0)
    
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha_cross", type=float, default=0.0)
    parser.add_argument("--alpha_class", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--beta_shift", type=float, default=0.0)
    parser.add_argument("--anomaly_threshold", type=float, default=0.0)
    
    parser.add_argument("--eval_results", type=str, default="")
    parser.add_argument("--image_only", action="store_true")
    
    parser.add_argument("--object_localization", action="store_true")
    parser.add_argument("--grdino_config", type=str, default="./checkpoints/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grdino_checkpoint", type=str, default="./checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_checkpoint", type=str, default="./checkpoints/sam_vit_l_0b3195.pth")

    parser.add_argument("--methods", type=str, default="open")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
