import numpy as np
import cv2


def expand_eyebrows(lmrks, eyebrows_expand_mod=1.0):

    lmrks = np.array( lmrks.copy(), dtype=np.int32 )

    # Top of the eye arrays
    bot_l = lmrks[[35, 41, 40, 42, 39]]
    bot_r = lmrks[[89, 95, 94, 96, 93]]

    # Eyebrow arrays
    top_l = lmrks[[43, 48, 49, 51, 50]]
    top_r = lmrks[[102, 103, 104, 105, 101]]

    # Adjust eyebrow arrays
    lmrks[[43, 48, 49, 51, 50]] = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
    lmrks[[102, 103, 104, 105, 101]] = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)
    return lmrks


def get_region(landmarks: np.ndarray, mode: str):
    if mode == 'eyes':
      # Get region according to 106 keypoints
      # left_points = landmarks[33:43]
      # right_points = landmarks[89:97]
      left_points = np.concatenate(([landmarks[1]],[landmarks[9]],[landmarks[10]], [landmarks[43]], [landmarks[48]]),axis=0)
      right_points = np.concatenate(([landmarks[17]],[landmarks[25]],[landmarks[26]], [landmarks[101]], [landmarks[105]]),axis=0)
    elif mode == 'cheek':
      left_points = landmarks[10: 16]
      right_points = landmarks[26: 32]
    elif mode == 'forehead':
      left_points = landmarks[48: 52]
      right_points = landmarks[102: 106]
    elif mode == 'face':
      left_points = np.concatenate((landmarks[10:17], landmarks[2:7]), axis=0)
      right_points = np.concatenate((landmarks[26:33], landmarks[18:23]), axis=0)
    elif mode == 'hair':
      left_points = landmarks[48: 52]
      right_points = landmarks[102: 106]
    elif mode == 'nose':
      left_points = landmarks[76:81]
      right_points = landmarks[82:87]
    elif mode == 'whole':
      left_points = landmarks[0: 78]
      right_points = landmarks[78: 106]
    elif mode == 'mouth' or mode == 'mouth_open':
      left_points = landmarks[52: 60]
      right_points = landmarks[60: 72]
    elif mode == 'left_eye':
      left_points = landmarks[33: 43]
      right_points = landmarks[33: 35]
    elif mode == 'right_eye':
      left_points = landmarks[89: 91]
      right_points = landmarks[89: 97]
    
    x1_left = int(min(left_points, key=lambda x: x[0])[0])
    x2_left = int(max(left_points, key=lambda x: x[0])[0])
    y1_left = int(min(left_points, key=lambda x: x[1])[1])
    y2_left = int(max(left_points, key=lambda x: x[1])[1])
    x1_right = int(min(right_points, key=lambda x: x[0])[0])
    x2_right = int(max(right_points, key=lambda x: x[0])[0])
    y1_right = int(min(right_points, key=lambda x: x[1])[1])
    y2_right = int(max(right_points, key=lambda x: x[1])[1])

    # Get region rectangle
    region = np.concatenate((left_points, right_points), axis=0)
    convexhull = cv2.convexHull(region)
    x, y, w, h = cv2.boundingRect(convexhull)

    if mode == 'eyes':
      # scale_factor_x = 1.5
      # scale_factor_y = 5
      # Get region rectangle
      center_x = x + w / 2
      center_y = y + h / 2
      scale_factor_x = 1
      scale_factor_y = 1.2
      new_x = int(center_x - (w * scale_factor_x) / 2)
      new_y = int(center_y - (h * scale_factor_y) / 2)
      new_w = int(w * scale_factor_x)
      new_h = int(h * scale_factor_y)
    elif mode == 'cheek':
      center_x = x + w / 2
      center_y = y + h / 2
      new_x, new_y, new_w, new_h = x, y, w, h
    elif mode == 'forehead':
      center_x = x + w / 2
      center_y = y + h / 2 - 30
      scale_factor_x = 1.5
      scale_factor_y = 4
      new_x = int(center_x - (w * scale_factor_x) / 2)
      new_y = int(center_y - (h * scale_factor_y) / 2)
      new_w = int(w * scale_factor_x)
      new_h = int(h * scale_factor_y)
    elif mode == 'face':
      center_x = x + w / 2
      center_y = y + h / 2
      new_x, new_y, new_w, new_h = x, y, w, h
    elif mode == 'hair':
      center_x = x + w / 2
      center_y = y + h / 2 - 50
      scale_factor_x = 1.5
      scale_factor_y = 4
      new_x = int(center_x - (w * scale_factor_x) / 2)
      new_y = int(center_y - (h * scale_factor_y) / 2)
      new_w = int(w * scale_factor_x)
      new_h = int(h * scale_factor_y)
    elif mode == 'nose':
      center_x = x + w / 2
      center_y = y + h / 2
      new_x, new_y, new_w, new_h = x, y, w, h
    elif mode == 'whole':
      center_x = x + w / 2
      center_y = y + h / 2 - 30
      scale_factor_x = 1.2
      scale_factor_y = 1.5
      new_x = int(center_x - (w * scale_factor_x) / 2)
      new_y = int(center_y - (h * scale_factor_y) / 2)
      new_w = int(w * scale_factor_x)
      new_h = int(h * scale_factor_y)
    elif mode == 'mouth' or mode == 'mouth_open':
      center_x = x + w / 2
      center_y = y + h / 2
      scale_factor_x = 1
      scale_factor_y = 1.2
      new_x = int(center_x - (w * scale_factor_x) / 2)
      new_y = int(center_y - (h * scale_factor_y) / 2)
      new_w = int(w * scale_factor_x)
      new_h = int(h * scale_factor_y)
    elif mode == 'left_eye' or mode == 'right_eye':
      center_x = x + w / 2
      center_y = y + h / 2
      scale_factor_x = 1.2
      scale_factor_y = 2
      new_x = int(center_x - (w * scale_factor_x) / 2)
      new_y = int(center_y - (h * scale_factor_y) / 2)
      new_w = int(w * scale_factor_x)
      new_h = int(h * scale_factor_y)
    
    return new_x, new_y, new_w, new_h


def get_mask_sticker(image: np.ndarray, landmarks: np.ndarray, mode: str) -> np.ndarray:
    """
    Get face mask for stickers
    """

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    new_x, new_y, new_w, new_h = get_region(landmarks, mode)
    cv2.rectangle(mask, (new_x, new_y), (new_x + new_w, new_y + new_h), 255, -1)
    
    return mask / 255


def get_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Get face mask of image size using given landmarks of person
    """

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)

    points = np.array(landmarks, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)
    
    return mask


def face_mask_static(image: np.ndarray, landmarks: np.ndarray, landmarks_tgt: np.ndarray, params = None) -> np.ndarray:
    """
    Get the final mask, using landmarks and applying blur
    """
    if params is None:
    
        left = np.sum((landmarks[1][0]-landmarks_tgt[1][0], landmarks[2][0]-landmarks_tgt[2][0], landmarks[13][0]-landmarks_tgt[13][0]))
        right = np.sum((landmarks_tgt[17][0]-landmarks[17][0], landmarks_tgt[18][0]-landmarks[18][0], landmarks_tgt[29][0]-landmarks[29][0]))
        
        offset = max(left, right)
        
        if offset > 6:
            erode = 15
            sigmaX = 15
            sigmaY = 10
        elif offset > 3:
            erode = 10
            sigmaX = 10
            sigmaY = 8
        elif offset < -3:
            erode = -5
            sigmaX = 5
            sigmaY = 10
        else:
            erode = 5
            sigmaX = 5
            sigmaY = 5
        
    else:
        erode = params[0]
        sigmaX = params[1]
        sigmaY = params[2]
    
    if erode == 15:
        eyebrows_expand_mod=2.7
    elif erode == -5:
        eyebrows_expand_mod=0.5
    else:
        eyebrows_expand_mod=2.0
    landmarks = expand_eyebrows(landmarks, eyebrows_expand_mod=eyebrows_expand_mod)
    
    mask = get_mask(image, landmarks)
    mask = erode_and_blur(mask, erode, sigmaX, sigmaY, True)
    
    if params is None:
        return mask/255, [erode, sigmaX, sigmaY]
        
    return mask/255


def erode_and_blur(mask_input, erode, sigmaX, sigmaY, fade_to_border = True):
    mask = np.copy(mask_input)
    
    if erode > 0:
        kernel = np.ones((erode, erode), 'uint8')
        mask = cv2.erode(mask, kernel, iterations=1)
    
    else:
        kernel = np.ones((-erode, -erode), 'uint8')
        mask = cv2.dilate(mask, kernel, iterations=1)
        
    if fade_to_border:
        clip_size = sigmaY * 2
        mask[:clip_size,:] = 0
        mask[-clip_size:,:] = 0
        mask[:,:clip_size] = 0
        mask[:,-clip_size:] = 0
    
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX = sigmaX, sigmaY = sigmaY)
        
    return mask
