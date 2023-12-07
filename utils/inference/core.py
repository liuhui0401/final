from typing import List, Tuple, Callable, Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pdb
import cv2
import kornia

from .faceshifter_run import faceshifter_batch
from .image_processing import crop_face, normalize_and_torch, normalize_and_torch_batch, add_sticker
from .video_processing import read_video, crop_frames_and_get_transforms, resize_frames
from .masks import face_mask_static, get_mask_sticker


def transform_target_to_torch(resized_frs: np.ndarray, half=True) -> torch.tensor:
    """
    Transform target, so it could be used by model
    """
    target_batch_rs = torch.from_numpy(resized_frs.copy()).cuda()
    target_batch_rs = target_batch_rs[:, :, :, [2,1,0]]/255.
        
    if half:
        target_batch_rs = target_batch_rs.half()
        
    target_batch_rs = (target_batch_rs - 0.5)/0.5 # normalize
    target_batch_rs = target_batch_rs.permute(0, 3, 1, 2)
    
    return target_batch_rs



def model_inference_all_new(full_frames: List[np.ndarray],
                    source: List,
                    target: List, 
                    sticker: List,
                    sticker_target: List,
                    netArc: Callable,
                    G: Callable,
                    app: Callable,
                    set_target: bool,
                    similarity_th=0.15,
                    crop_size=224,
                    BS=60,
                    half=True,
                    mode='glasses',
                    handler=None,
                    OUT_VIDEO_NAME='result.mp4',
                    fps=20.0):
    """
    Adding stickers to original images
    """
    result_frames = full_frames.copy()
    out = cv2.VideoWriter(f"{OUT_VIDEO_NAME}", cv2.VideoWriter_fourcc(*'mp4v'), fps, (full_frames[0].shape[1], full_frames[0].shape[0]))

    if len(target) != 0:
      # Get Arcface embeddings of target image
      target_norm = normalize_and_torch_batch(np.array(target))
      target_embeds = netArc(F.interpolate(target_norm, scale_factor=0.5, mode='bilinear', align_corners=True))

      # Get the cropped faces from original frames and transformations to get those crops
      crop_frames_list, tfm_array_list = crop_frames_and_get_transforms(result_frames, target_embeds, app, netArc, crop_size, set_target, similarity_th=similarity_th)

      # Normalize source images and transform to torch and get Arcface embeddings
      source_embeds = []
      for source_curr in source:
          source_curr = normalize_and_torch(source_curr)
          source_embeds.append(netArc(F.interpolate(source_curr, scale_factor=0.5, mode='bilinear', align_corners=True)))
      
      final_frames_list = []
      for idx, (crop_frames, tfm_array, source_embed) in enumerate(zip(crop_frames_list, tfm_array_list, source_embeds)):
          # Resize croped frames and get vector which shows on which frames there were faces
          resized_frs, present = resize_frames(crop_frames)
          resized_frs = np.array(resized_frs)

          # transform embeds of Xs and target frames to use by model
          target_batch_rs = transform_target_to_torch(resized_frs, half=half)

          if half:
              source_embed = source_embed.half()

          # run model
          size = target_batch_rs.shape[0]
          model_output = []

          for i in tqdm(range(0, size, BS)):
              Y_st = faceshifter_batch(source_embed, target_batch_rs[i:i+BS], G)
              model_output.append(Y_st)
          torch.cuda.empty_cache()
          model_output = np.concatenate(model_output)

          # create list of final frames with transformed faces
          final_frames = []
          idx_fs = 0

          for pres in tqdm(present):
              if pres == 1:
                  # output_frames = add_sticker(model_output[idx_fs], sticker[0], mode, handler)
                  # final_frames.append(output_frames)
                  final_frames.append(model_output[idx_fs])
                  idx_fs += 1
              else:
                  final_frames.append([])
          # _, output_frames = add_sticker(final_frames, sticker[0], mode, handler)
          # final_frames_list.append(output_frames)
          final_frames_list.append(final_frames)
    

      params = [None for i in range(len(crop_frames_list))]
      size = (result_frames[0].shape[0], result_frames[0].shape[1])
      for i in tqdm(range(len(result_frames))):
        if i == len(result_frames):
            break
        for j in range(len(crop_frames_list)): # length of target embedding
          try:
            swap = cv2.resize(final_frames_list[j][i][:, :, :3], (224, 224))
            cv2.imwrite('examples/images/vis2.png', swap)

            if len(crop_frames_list[j][i]) == 0:
                params[j] = None
                continue
                
            landmarks_face = handler.get_without_detection_without_transform(swap)
            if params[j] == None:     
                landmarks_tgt = handler.get_without_detection_without_transform(crop_frames_list[j][i])
                mask_face, params[j] = face_mask_static(swap, landmarks_face, landmarks_tgt, params[j])
            else:
                mask_face = face_mask_static(swap, landmarks_face, landmarks_tgt, params[j])    
                    
            mask_face = torch.from_numpy(mask_face).cuda().unsqueeze(0).unsqueeze(0).type(torch.float32)
            swap = torch.from_numpy(swap).cuda().permute(2,0,1).unsqueeze(0).type(torch.float32)
            full_frame = torch.from_numpy(result_frames[i]).cuda().permute(2,0,1).unsqueeze(0)
            mat = torch.from_numpy(tfm_array_list[j][i]).cuda().unsqueeze(0).type(torch.float32)
            
            mat_rev = kornia.invert_affine_transform(mat)
            swap_t = kornia.warp_affine(swap, mat_rev, size)
            mask_face_t = kornia.warp_affine(mask_face, mat_rev, size)
            final = (mask_face_t*swap_t + (1-mask_face_t)*full_frame).type(torch.uint8).squeeze().permute(1,2,0).cpu().detach().numpy()
            cv2.imwrite('examples/images/vis6.png', final)

            result_frames[i] = final
            torch.cuda.empty_cache()

          except Exception as e:
            print('face')
            pass
        if len(sticker) == 0:
          out.write(result_frames[i])


    if len(sticker) != 0:
      # Get Arcface embeddings of target image
      target_norm = normalize_and_torch_batch(np.array(sticker_target))
      target_embeds = netArc(F.interpolate(target_norm, scale_factor=0.5, mode='bilinear', align_corners=True))

      # Get the cropped faces from original frames and transformations to get those crops
      sticker_crop_frames_list, sticker_tfm_array_list = crop_frames_and_get_transforms(result_frames, target_embeds, app, netArc, crop_size, set_target, similarity_th=similarity_th)

      sticker_final_frames_list = []
      for idx, (crop_frames, tfm_array) in enumerate(zip(sticker_crop_frames_list, sticker_tfm_array_list)):
          # Resize croped frames and get vector which shows on which frames there were faces
          resized_frs, present = resize_frames(crop_frames)
          resized_frs = np.array(resized_frs)

          # Add stickers to frames
          output_frames, _ = add_sticker(resized_frs, sticker[idx], mode[idx], handler)

          # create list of final frames with transformed faces
          final_frames = []
          idx_fs = 0

          for pres in tqdm(present):
              if pres == 1:
                  final_frames.append(output_frames[idx_fs][0])
                  idx_fs += 1
              else:
                  final_frames.append([])
          sticker_final_frames_list.append(final_frames)
      

      size = (result_frames[0].shape[0], result_frames[0].shape[1])
      for i in tqdm(range(len(result_frames))):
        if i == len(result_frames):
            break
        for j in range(len(sticker_crop_frames_list)):
          try:
            sticker_swap = cv2.resize(sticker_final_frames_list[j][i][:, :, :3], (224, 224))
            cv2.imwrite('examples/images/vis21.png', sticker_swap)

            landmarks_sticker = handler.get_without_detection_without_transform(sticker_crop_frames_list[j][i])
            mask_sticker = get_mask_sticker(sticker_crop_frames_list[j][i], landmarks_sticker, mode[j])
            mask_sticker = torch.from_numpy(mask_sticker).cuda().unsqueeze(0).unsqueeze(0).type(torch.float32)

            sticker_swap = torch.from_numpy(sticker_swap).cuda().permute(2,0,1).unsqueeze(0).type(torch.float32)
            full_frame = torch.from_numpy(result_frames[i]).cuda().permute(2,0,1).unsqueeze(0)
            sticker_mat = torch.from_numpy(sticker_tfm_array_list[j][i]).cuda().unsqueeze(0).type(torch.float32)

            sticker_mat_rev = kornia.invert_affine_transform(sticker_mat)
            sticker_swap_t = kornia.warp_affine(sticker_swap, sticker_mat_rev, size)

            cv2.imwrite('examples/images/vis31.png', sticker_swap_t.squeeze().permute(1,2,0).cpu().detach().numpy())
            mask_sticker_t = kornia.warp_affine(mask_sticker, sticker_mat_rev, size)
            cv2.imwrite('examples/images/vis41.png', (mask_sticker_t*sticker_swap_t).squeeze().permute(1,2,0).cpu().detach().numpy())
            final = (mask_sticker_t*sticker_swap_t + (1-mask_sticker_t)*full_frame).type(torch.uint8).squeeze().permute(1,2,0).cpu().detach().numpy()

            cv2.imwrite('examples/images/vis11.png', final)

            result_frames[i] = final
            torch.cuda.empty_cache()

          except Exception as e:
              print('sticker')
              pass
                
        out.write(result_frames[i])

    out.release()

    # return final_frames_list, sticker_final_frames_list, crop_frames_list, tfm_array_list, sticker_crop_frames_list, sticker_tfm_array_list, full_frames


def model_inference_all(full_frames: List[np.ndarray],
                    source: List,
                    sticker: List,
                    target: List, 
                    netArc: Callable,
                    G: Callable,
                    app: Callable,
                    set_target: bool,
                    similarity_th=0.15,
                    crop_size=224,
                    BS=60,
                    half=True,
                    mode='glasses',
                    handler=None):
    """
    Adding stickers to original images
    """
    # Get Arcface embeddings of target image
    target_norm = normalize_and_torch_batch(np.array(target))
    target_embeds = netArc(F.interpolate(target_norm, scale_factor=0.5, mode='bilinear', align_corners=True))

    # Get the cropped faces from original frames and transformations to get those crops
    crop_frames_list, tfm_array_list = crop_frames_and_get_transforms(full_frames, target_embeds, app, netArc, crop_size, set_target, similarity_th=similarity_th)

    # Normalize source images and transform to torch and get Arcface embeddings
    source_embeds = []
    for source_curr in source:
        source_curr = normalize_and_torch(source_curr)
        source_embeds.append(netArc(F.interpolate(source_curr, scale_factor=0.5, mode='bilinear', align_corners=True)))
    
    final_frames_list = []
    for idx, (crop_frames, tfm_array, source_embed) in enumerate(zip(crop_frames_list, tfm_array_list, source_embeds)):
        # Resize croped frames and get vector which shows on which frames there were faces
        resized_frs, present = resize_frames(crop_frames)
        resized_frs = np.array(resized_frs)

        # transform embeds of Xs and target frames to use by model
        target_batch_rs = transform_target_to_torch(resized_frs, half=half)

        if half:
            source_embed = source_embed.half()

        # run model
        size = target_batch_rs.shape[0]
        model_output = []

        for i in tqdm(range(0, size, BS)):
            Y_st = faceshifter_batch(source_embed, target_batch_rs[i:i+BS], G)
            model_output.append(Y_st)
        torch.cuda.empty_cache()
        model_output = np.concatenate(model_output)

        # create list of final frames with transformed faces
        final_frames = []
        idx_fs = 0

        for pres in tqdm(present):
            if pres == 1:
                # output_frames = add_sticker(model_output[idx_fs], sticker[0], mode, handler)
                # final_frames.append(output_frames)
                final_frames.append(model_output[idx_fs])
                idx_fs += 1
            else:
                final_frames.append([])
        _, output_frames = add_sticker(final_frames, sticker[0], mode, handler)
        final_frames_list.append(output_frames)

    return final_frames_list, crop_frames_list, full_frames, tfm_array_list


def model_inference_sticker(full_frames: List[np.ndarray],
                    source: List,
                    target: List, 
                    netArc: Callable,
                    G: Callable,
                    app: Callable,
                    set_target: bool,
                    similarity_th=0.15,
                    crop_size=224,
                    BS=60,
                    half=True,
                    mode='glasses',
                    handler=None):
    """
    Adding stickers to original images
    """
    # Get Arcface embeddings of target image
    target_norm = normalize_and_torch_batch(np.array(target))
    target_embeds = netArc(F.interpolate(target_norm, scale_factor=0.5, mode='bilinear', align_corners=True))

    # Get the cropped faces from original frames and transformations to get those crops
    crop_frames_list, tfm_array_list = crop_frames_and_get_transforms(full_frames, target_embeds, app, netArc, crop_size, set_target, similarity_th=similarity_th)

    final_frames_list = []
    for idx, (crop_frames, tfm_array) in enumerate(zip(crop_frames_list, tfm_array_list)):
        # Resize croped frames and get vector which shows on which frames there were faces
        resized_frs, present = resize_frames(crop_frames)
        resized_frs = np.array(resized_frs)

        # Add stickers to frames
        output_frames, _ = add_sticker(resized_frs, source[0], mode, handler)

        # create list of final frames with transformed faces
        final_frames = []
        idx_fs = 0

        for pres in tqdm(present):
            if pres == 1:
                final_frames.append(output_frames[idx_fs][0])
                idx_fs += 1
            else:
                final_frames.append([])
        final_frames_list.append(final_frames)
    
    return final_frames_list, crop_frames_list, full_frames, tfm_array_list   


def model_inference(full_frames: List[np.ndarray],
                    source: List,
                    target: List, 
                    netArc: Callable,
                    G: Callable,
                    app: Callable,
                    set_target: bool,
                    similarity_th=0.15,
                    crop_size=224,
                    BS=60,
                    half=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Using original frames get faceswaped frames and transofrmations
    """
    # Get Arcface embeddings of target image
    target_norm = normalize_and_torch_batch(np.array(target))
    target_embeds = netArc(F.interpolate(target_norm, scale_factor=0.5, mode='bilinear', align_corners=True))
    
    # Get the cropped faces from original frames and transformations to get those crops
    crop_frames_list, tfm_array_list = crop_frames_and_get_transforms(full_frames, target_embeds, app, netArc, crop_size, set_target, similarity_th=similarity_th)
    
    # Normalize source images and transform to torch and get Arcface embeddings
    source_embeds = []
    for source_curr in source:
        source_curr = normalize_and_torch(source_curr)
        source_embeds.append(netArc(F.interpolate(source_curr, scale_factor=0.5, mode='bilinear', align_corners=True)))
    
    final_frames_list = []
    for idx, (crop_frames, tfm_array, source_embed) in enumerate(zip(crop_frames_list, tfm_array_list, source_embeds)):
        # Resize croped frames and get vector which shows on which frames there were faces
        resized_frs, present = resize_frames(crop_frames)
        resized_frs = np.array(resized_frs)

        # transform embeds of Xs and target frames to use by model
        target_batch_rs = transform_target_to_torch(resized_frs, half=half)

        if half:
            source_embed = source_embed.half()

        # run model
        size = target_batch_rs.shape[0]
        model_output = []

        for i in tqdm(range(0, size, BS)):
            Y_st = faceshifter_batch(source_embed, target_batch_rs[i:i+BS], G)
            model_output.append(Y_st)
        torch.cuda.empty_cache()
        model_output = np.concatenate(model_output)

        # create list of final frames with transformed faces
        final_frames = []
        idx_fs = 0

        for pres in tqdm(present):
            if pres == 1:
                final_frames.append(model_output[idx_fs])
                idx_fs += 1
            else:
                final_frames.append([])
        final_frames_list.append(final_frames)
    
    return final_frames_list, crop_frames_list, full_frames, tfm_array_list   