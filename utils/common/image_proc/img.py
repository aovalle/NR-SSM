import numpy as np
import cv2

def rgb2gray(img):
    if img.shape[0] == 3:   # Check for pytorch ordering
        img = np.transpose(img, (1, 2, 0))
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def preprocessing(img, frame_size=(84,84), grayscale=True):
    if grayscale:
        img = rgb2gray(img)
        img = cv2.resize(img, frame_size, interpolation=cv2.INTER_AREA)
    else:
        img = np.transpose(img, (1, 2, 0)) # for pytorch
        img = cv2.resize(img, frame_size, interpolation=cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1))
    return img


def stack_and_process(stacked_frames, obs, is_new_episode, frame_size=(84,84), grayscale=True):
    frame = preprocessing(obs, frame_size, grayscale)

    if is_new_episode:
        if grayscale:
            stacked_frames = np.stack([frame for _ in range(4)], axis=0)
        else:
            stacked_frames = np.concatenate([frame for _ in range(4)], axis=0)
    else:
        if grayscale:
            stacked_frames = stacked_frames[1:, ...]
            stacked_frames = np.concatenate([stacked_frames, np.expand_dims(frame, axis=0)], axis=0)
        else:
            stacked_frames = stacked_frames[3:, ...]
            stacked_frames = np.concatenate([stacked_frames, frame], axis=0)
    #print(stacked_frames.shape)
    #raise
    return stacked_frames