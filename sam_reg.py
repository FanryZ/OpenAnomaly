import torch
import numpy as np
import cv2
from prompt_ensemble import encode_text_with_prompt_ensemble
from segment_anything import SamPredictor, sam_model_registry


def gaussian_kernel(center_x, center_y, height, width, sigma=50):
    xv, yv = np.meshgrid(range(height), range(width), indexing="xy")
    distances = np.sqrt((center_x - xv)**2 + (center_y - yv)**2)
    # gaussian_kernel = np.exp(-distances**2 / (2 * sigma**2))
    gaussian_kernel = 1 / (1 + distances**2)
    return gaussian_kernel
    # normed_kernel = (masked_result - np.min(masked_result)) / (np.max(masked_result) - np.min(masked_result))
    

def box_intersect(box1, box2):
    point_in_box = lambda x, y, box: x >= box[0] and x <= box[0] + box[2] and y >= box[1] and y <= box[1] + box[3]
    x11, y11, x12, y12 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x21, y21, x22, y22 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    return point_in_box(x11, y11, box2) or point_in_box(x11, y12, box2) or point_in_box(x12, y11, box2) or point_in_box(x12, y12, box2) or \
        point_in_box(x21, y21, box1) or point_in_box(x21, y22, box1) or point_in_box(x22, y21, box1) or point_in_box(x22, y22, box1)


def region_seperate(input_heatmap, min_area=100, filter_ratio=50.):
    # grey_map = (input_heatmap / input_heatmap.max() * 255).astype("uint8")[0]
    grey_map = (input_heatmap * 255).astype("uint8")[0]
    # kernel = np.ones((5, 5), np.uint8)
    # grep_map = cv2.dilate(grey_map, kernel, iterations=1)

    contours, _ = cv2.findContours(grey_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    filtered_boxes = [cv2.boundingRect(contour) for contour in filtered_contours]
    
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        # Calculate the centroid (center of mass)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
            
    flag = True
    boxes = filtered_boxes
    while flag:
        merged_boxes, flag = [], False
        for box1 in filtered_boxes:
            # Get the bounding rectangle for the current contour
            x1, y1, w1, h1 = box1
            merged = False
            for box2 in merged_boxes:
                x2, y2, w2, h2 = box2
                if box_intersect(box1, box2):
                    # Bounding boxes intersect; merge them
                    # if w1 * h1 > w2 * h2:        
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = max(x1 + w1, x2 + w2) - x
                    h = max(y1 + h1, y2 + h2) - y
                    box2[0], box2[1], box2[2], box2[3] = x, y, w, h
                    merged, flag = True, True
            if not merged:
                merged_boxes.append([x1, y1, w1, h1])
        filtered_boxes = merged_boxes

    # [cv2.rectangle(grey_map, (x, y), (x+w, y+h), 255, 2) for (x, y, w, h) in merged_boxes]
    # cv2.imwrite("bboxes.png", grey_map)
    merged_boxes = list(set([tuple(box) for box in merged_boxes]))
    
    merged_centers_idx, merged_contour = [], []
    for mbox in merged_boxes:
        max_area = 0
        merged_centers_idx.append(-1)
        for idx, box in enumerate(boxes):
            if box_intersect(mbox, box) and box[2] * box[3] > max_area:
                max_area = box[2] * box[3]
                merged_centers_idx[-1] = idx
    
    if len(merged_boxes) == 0:
        return [], [], []
    merged_centers = np.array(centers)[merged_centers_idx, :]
    merged_contours = [filtered_contours[i] for i in merged_centers_idx]
    merged_boxes = np.array(merged_boxes)
    
    if filter_ratio > 0:
        boxes_output, contours_output, centers_output = [], [], []
        box_max = max([w * h for x, y, w, h in merged_boxes])
        for box, contour, center in zip(merged_boxes, merged_contours, merged_centers):
            if box[2] * box[3] * filter_ratio > box_max:
                boxes_output.append(box)
                contours_output.append(contour)
                centers_output.append(center)
        return boxes_output, contours_output, centers_output
    return merged_boxes, merged_contours, merged_centers


def mean_coord(dense_map):
    H, W = dense_map.shape[-2:]
    iv, jv = np.meshgrid(range(H), range(W), indexing="ij")
    i_index = int(np.sum(dense_map * iv) / np.sum(dense_map))
    j_index = int(np.sum(dense_map * jv) / np.sum(dense_map))
    return i_index, j_index


def contour_seperate(input_heatmap, min_area=100, filter_ratio=10.):

    grey_map = (input_heatmap * 255).astype("uint8")[0]
    contours, _ = cv2.findContours(grey_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        return [], [], []
    max_contour_area = max([cv2.contourArea(contour) for contour in contours])
    
    valid_contours, output_boxes, centers = [], [], []
    for contour in contours:
        if cv2.contourArea(contour) * filter_ratio < max_contour_area:
            continue
        valid_contours.append(contour)
        output_boxes.append(cv2.boundingRect(contour))
        
        contour_mask = np.zeros_like(input_heatmap[0], dtype="uint8")
        cv2.drawContours(contour_mask, [contour], -1, (1), thickness=cv2.FILLED)
        i_index, j_index = mean_coord(input_heatmap * contour_mask)
        if contour_mask[i_index, j_index] > 0:
            cX, cY = j_index, i_index
        else:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        centers.append([cX, cY])
    return output_boxes, valid_contours, centers


def sam_mask_regression(predictor, anomaly_map, rtype="softmax", use_boxes=False, kernel_sigma=-1, max_masks=-1, 
                        region_mask_threshold=-1., area_threshold_ratio=-1., iou_threshold=-1., anomaly_threshold=0.0):

    assert rtype in ["softmax", "max", "intersect"]
    anomaly_map = anomaly_map / (1e-6 + anomaly_map.max())
    anomaly_map[anomaly_map < anomaly_threshold] = 0.0
    boxes, contours, centers = contour_seperate(anomaly_map)
    # boxes, contours, centers = region_seperate(anomaly_map_valid)
    if len(boxes) == 0:
        output_mask = np.zeros_like(anomaly_map[0])
        return output_mask[np.newaxis, :, :], centers, boxes
    output_masks = []
    
    _, h, w = anomaly_map.shape
    area_threshold = area_threshold_ratio * h * w if area_threshold_ratio > 0 else h * w
    if use_boxes:
        input_boxes = torch.tensor([
            [x, y, x+w, y+h] for x, y, w, h in boxes
        ], device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, anomaly_map.shape[-2:])
        
        masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
        )

        masks, scores = masks.cpu().numpy(), scores.cpu().numpy()
        for mask, score, center, box in zip(masks, scores, centers, boxes):
            if np.max(score) < 0.8:
                continue
            if rtype == "softmax":
                _mask = np.sum(mask * np.exp(score[:, np.newaxis, np.newaxis]), axis=0) / np.sum(np.exp(score))
            elif rtype == "intersect":
                _mask = (np.sum(mask, axis=0) == mask.shape[0]).astype("float32")
            
            bx, by, bw, bh = box
            if bw * bh * 4 >= mask.sum():
                if kernel_sigma > 0:
                    _kernel = gaussian_kernel(center[0], center[1], h, w, kernel_sigma) 
                    output_masks.append(_mask * _kernel)
                else:
                    output_masks.append(_mask)
    else:
        points = torch.Tensor(centers).to(predictor.device).unsqueeze(1)
        labels = torch.Tensor([int(1) for _, l in centers]).to(predictor.device).unsqueeze(1)
        transformed_points = predictor.transform.apply_coords_torch(points, [h, w])
        # Predict
        batched_masks, batched_scores, batched_logits = predictor.predict_torch(
                point_coords=transformed_points,
                point_labels=labels,
                boxes=None,
                multimask_output=True,
            )
        batched_masks, batched_scores, batched_logits = \
            batched_masks.cpu().numpy(), batched_scores.cpu().numpy(), batched_logits.cpu().numpy()
        for center, contour, box, _masks, _scores, _logits in \
            zip(centers, contours, boxes, batched_masks, batched_scores, batched_logits):
            
            # _masks, _scores, _logits = predictor.predict(
            #     point_coords=np.array([center]),
            #     point_labels=np.array([1]),
            #     multimask_output=True,
            # )
            
            _logits = np.array([
                cv2.resize(1 / (np.exp(-_logit) + 1), (w, h), interpolation=cv2.INTER_LINEAR)
                    for _logit in np.clip(_logits.astype("float32"), -10., 10.)
            ])
            
            # filter masks by area maximum
            mask_areas = np.sum(np.sum(_masks, axis=1), axis=1)
            valid_masks = mask_areas < area_threshold
            masks, scores, logits = _masks[valid_masks], _scores[valid_masks], _logits[valid_masks]
            if masks.size == 0:
                continue
            
            # compute iou with contours
            contour_mask = np.zeros_like(masks[0], dtype="uint8")
            cv2.drawContours(contour_mask, [contour], -1, (1), thickness=cv2.FILLED)
            contour_mask_ious = np.sum(np.sum(masks * contour_mask[np.newaxis, :, :], axis=1), axis=1) / \
                (1e-6 + np.sum(np.sum(masks | contour_mask[np.newaxis, :, :], axis=1), axis=1))
                
            if np.max(contour_mask_ious) < iou_threshold:
                continue
            iou_scores = contour_mask_ious * scores
            mask = np.sum(iou_scores[:, np.newaxis, np.newaxis] * logits, axis=0) / (1e-6 + iou_scores.sum())
            output_masks.append(mask)
    output_masks = np.array(output_masks)
    
    if region_mask_threshold > 0.:
        contour_areas = np.array([cv2.contourArea(contour) for contour in contours])
        valid_masks = contour_areas * region_mask_threshold > np.array([mask.sum() for mask in output_masks])
        boxes = np.array(boxes)[valid_masks]
        centers = np.array(centers)[valid_masks]
        output_masks = np.array(output_masks)[valid_masks]
    if max_masks <= 0:
        output_mask = np.sum(output_masks, axis=0)
    else:
        max_values = np.array([np.max(mask * anomaly_map[0]) for mask in output_masks])
        top_index = np.argsort(-max_values)[:max_masks]
        output_mask = np.array(output_masks)[top_index]
        centers = np.array(centers)[top_index]
        boxes = np.array(boxes)[top_index]
    return output_masks, centers, boxes
    
    
def mask_merge(masks):
    weights = np.zeros_like(masks[0])
    for mask in masks:
        weights += (mask > 0.001).astype("float32")
    mask_output = np.sum(masks, axis=0)
    mask_output[weights > 0] /= weights[weights > 0]
    return mask_output
    
def mask_refine(mask, refinement, beta=2.0):
    ratio = 1.0 / (1.001 + beta * refinement)
    return mask ** ratio    
    
def patch_affinity(image_size, features):    
    H, W = image_size[-2:]
    B, N, C = features.shape
    features = torch.permute(features, [0, 2, 1])
    features_flattern = features.view(B * C, N)

    features_self_similarity = features_flattern.T @ features_flattern
    features_self_similarity = 0.5 * (1 - features_self_similarity)
    features_self_similarity = features_self_similarity.sort(dim=1)[0]

    features_self_similarity = torch.mean(features_self_similarity[:, :100], dim=1)
    pa_map = features_self_similarity.view(H, W).cpu().numpy()

    return pa_map
