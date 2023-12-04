import base64
from io import BytesIO
from typing import Callable, List

import numpy as np
import torch
import cv2
from .masks import face_mask_static, get_mask_sticker, get_region
from matplotlib import pyplot as plt
from insightface.utils import face_align


def crop_face(image_full: np.ndarray, app: Callable, crop_size: int) -> np.ndarray:
    """
    Crop face from image and resize
    """
    kps = app.get(image_full, crop_size)
    M, _ = face_align.estimate_norm(kps[0], crop_size, mode ='None') 
    align_img = cv2.warpAffine(image_full, M, (crop_size, crop_size), borderValue=0.0)         
    return [align_img]


def normalize_and_torch(image: np.ndarray) -> torch.tensor:
    """
    Normalize image and transform to torch
    """
    image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
    if image.max() > 1.:
        image = image/255.
    
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image


def normalize_and_torch_batch(frames: np.ndarray) -> torch.tensor:
    """
    Normalize batch images and transform to torch
    """
    batch_frames = torch.from_numpy(frames.copy()).cuda()
    if batch_frames.max() > 1.:
        batch_frames = batch_frames/255.
    
    batch_frames = batch_frames.permute(0, 3, 1, 2)
    batch_frames = (batch_frames - 0.5)/0.5

    return batch_frames


def add_sticker(crop_frames: List[np.ndarray],
                source: np.ndarray,
                handler):
    """
    Adding stickers to crop frames
    """
    output = [[] for i in range(len(crop_frames[0]))]
    for i in range(len(crop_frames[0])):
        landmarks = handler.get_without_detection_without_transform(crop_frames[0][i])
        # Get special region according to 106 keypoints
        # Get masks for stickers
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        _, source_mask = cv2.threshold(source_gray, 200, 255, cv2.THRESH_BINARY)
        source_mask = cv2.bitwise_not(source_mask)

        # Get stickers
        source_img = np.zeros((source.shape[0], source.shape[1]), 4, dtype=np.uint8)
        source_img[:, :, 0:3][source_mask > 0] = source[source_mask > 0]
        source_img[:, :, 3][source_mask > 0] = 255

        # Get rectangle containing stickers
        nonzero_pixels = cv2.findNonZero(source_mask)
        source_x, source_y, source_w, source_h = cv2.boundingRect(nonzero_pixels)
        source_mask_center_x = x + w // 2
        source_mask_center_y = y + h // 2

        # Get rectangle in frame
        frame_x, frame_y, frame_w, frame_h = get_region(landmarks, mode)
        frame_center_x, frame_center_y = int(frame_x + frame_w / 2), int(frame_y + frame_h / 2)

        # Transform sticker to frame
        source_mask_region = source_mask[source_y: source_y+source_h, source_x: source_x + source_w]
        source_img_region = source_img[source_y: source_y+source_h, source_x: source_x + source_w]
        if frame_w % 2 != 0:
            frame_w += 1
        if frame_h % 2 != 0:
            frame_h += 1
        resized_source_mask = cv2.resize(source_mask_region, (frame_w, frame_h))
        resized_source_region = cv2.resize(source_img_region, (frame_w, frame_h))

        # Adding transformed sticker to frame
        resized_source_mask_center_x, resized_source_mask_center_y = frame_w // 2, frame_h // 2
        x_offset, y_offset = source_mask_center_x - resized_source_mask_center_x, source_mask_center_y - resized_source_mask_center_y
        resized_source_img = np.ones_like(source_img) * 255
        resized_source_img[source_y-resized_source_mask_center_y: source_y+resized_source_mask_center_y, source_x-resized_source_mask_center_x: source_x+resized_source_mask_center_x][resized_source_mask > 0] \
            = resized_source_region[resized_source_mask > 0]
        
        resized_source_img_gray = cv2.cvtColor(resized_source_img, cv2.COLOR_BGR2GRAY)
        _, resized_source_mask = cv2.threshold(resized_source_img_gray, 200, 255, cv2.THRESH_BINARY)
        resized_source_mask = cv2.bitwise_not(resized_source_mask)

        resized_nonzero_pixels = cv2.findNonZero(resized_source_mask)
        resized_source_mask_x, resized_source_mask_y, resized_source_mask_w, resized_source_mask_h = cv2.boundingRect(resized_nonzero_pixels)
        resized_source_mask_center_x, resized_source_mask_center_y = resized_source_mask_x + resized_source_mask_x // 2, resized_source_mask_y + resized_source_mask_h // 2
        resized_x_offset, resized_y_offset = frame_center_x - resized_source_mask_center_x, frame_center_y - resized_source_mask_y
        x_start, y_start = max(resized_x_offset, 0), max(resized_y_offset, 0)
        x_end, y_end = min(resized_x_offset + resized_source_img.shape[1], crop_frames[0][i].shape[1]), min(resized_y_offset + resized_source_img.shape[0], crop_frames[0][i].shape[0])

        # Get final
        final_img = np.zeros((crop_frames[0][i].shape[0], crop_frames[0][i].shape[1], 4), dtype=np.uint8)
        final_img[:, :, 0:3] = crop_frames[0][i]
        final_img[:, :, 3] = 255
        resized_source_mask_offset = resized_source_mask[y_start-resized_y_offset: y_end-resized_y_offset, x_start-resized_x_offset: x_end-resized_x_offset]
        final_img[y_start: y_end, x_start: x_end][resized_source_mask_offset > 0] = resized_source_img[y_start-resized_y_offset: y_end-resized_y_offset, x_start-resized_x_offset: x_end-resized_x_offset][resized_source_mask_offset > 0]
        output[0].append(final_img)
    return output


def get_final_image_sticker(final_frames: List[np.ndarray],
                    crop_frames: List[np.ndarray],
                    full_frame: np.ndarray,
                    tfm_arrays: List[np.ndarray],
                    mode: str,
                    handler) -> None:
    """
    Create final video from frames and add stickers
    """
    final = full_frame.copy()
    params = [None for i in range(len(final_frames))]
    
    for i in range(len(final_frames)):
        frame = cv2.resize(final_frames[i][0], (224, 224))

        landmarks = handler.get_without_detection_without_transform(crop_frames[i][0])
        mask = get_mask_sticker(crop_frames[i][0], landmarks, mode)
        mat_rev = cv2.invertAffineTransform(tfm_arrays[i][0])

        sticker_t = cv2.warpAffine(frame, mat_rev, (full_frame.shape[1], full_frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        mask_t = cv2.warpAffine(mask, mat_rev, (full_frame.shape[1], full_frame.shape[0]))
        mask_t = np.expand_dims(mask_t, 2)

        final = mask_t * sticker_t + (1-mask_t)*final
    final = np.array(final, dtype='uint8')
    return final


def get_final_image(final_frames: List[np.ndarray],
                    crop_frames: List[np.ndarray],
                    full_frame: np.ndarray,
                    tfm_arrays: List[np.ndarray],
                    handler) -> None:
    """
    Create final video from frames
    """
    final = full_frame.copy()
    params = [None for i in range(len(final_frames))]
    
    for i in range(len(final_frames)):
        frame = cv2.resize(final_frames[i][0], (224, 224))
        
        landmarks = handler.get_without_detection_without_transform(frame)     
        landmarks_tgt = handler.get_without_detection_without_transform(crop_frames[i][0])

        mask, _ = face_mask_static(crop_frames[i][0], landmarks, landmarks_tgt, params[i])
        mat_rev = cv2.invertAffineTransform(tfm_arrays[i][0])

        swap_t = cv2.warpAffine(frame, mat_rev, (full_frame.shape[1], full_frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        mask_t = cv2.warpAffine(mask, mat_rev, (full_frame.shape[1], full_frame.shape[0]))
        mask_t = np.expand_dims(mask_t, 2)

        final = mask_t*swap_t + (1-mask_t)*final
    final = np.array(final, dtype='uint8')
    return final


def show_images(images: List[np.ndarray], 
                titles=None, 
                figsize=(20, 5), 
                fontsize=15):
    if titles:
        assert len(titles) == len(images), "Amount of images should be the same as the amount of titles"
    
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for idx, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image[:, :, ::-1])
        if titles:
            ax.set_title(titles[idx], fontsize=fontsize)
        ax.axis("off")
