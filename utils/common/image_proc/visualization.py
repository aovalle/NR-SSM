import torch
import os
import cv2
import numpy as np

from torchvision.utils import make_grid

class VideoRecorder(object):

    def __init__(self, dir, fps=30, with_recon=False):
        self.frames = []
        self.fps = fps
        self.videodir = os.path.join(dir, "videos")
        self.with_recon = with_recon

    def append(self, obs, decoded_obs=None, rescale='recon'):    # rescale can be 'real', 'recon' or 'both'
        if not self.with_recon:
            self.frames.append(obs)
        else:
            if rescale == 'recon':
                decoded_obs = (decoded_obs.squeeze(0).cpu().numpy() + 0.5) * 255.
            catt = np.concatenate((obs, decoded_obs), axis=-1)
            self.frames.append(catt)

    def flush(self):
        self.frames = []

    def record(self, episode, rescale=False):
        file = os.path.join(self.videodir, "vid_{}.mp4".format(episode))

        frames = np.stack(self.frames, axis=0).transpose((0, 2, 3, 1))  # (n frames, w, h, c)
        if rescale:
            frames = np.multiply(frames, 255).clip(0, 255).astype(np.uint8)
        else:
            frames = frames.clip(0, 255).astype(np.uint8)
        frames = frames[:, :, :, ::-1]  # Reverse channels
        _, h, w, _ = frames.shape  # numpy is (row, col) but cv is (height, width)
        # Codecs mp4v, mp42
        writer = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*'mp4v'), float(self.fps), (w, h))  # , color)
        for frame in frames:
            writer.write(frame)
        writer.release()
        print("Saved _video_ at path `{}`".format(file))

''' for model-based agents as it includes the reconstruction '''
# def save_video(dir, episode, frames, rescale=False, fps=30.0, color=True):
#     videodir = os.path.join(dir, "videos")
#     path = os.path.join(videodir, "vid_{}.mp4".format(episode))
#     frames = np.stack(frames, axis=0).transpose(0, 2, 3, 1)     # (n frames, w, h, c)
#     if rescale:
#         frames = np.multiply(frames, 255).clip(0, 255).astype(np.uint8)
#     else:
#         frames = frames.clip(0, 255).astype(np.uint8)
#     frames = frames[:, :, :, ::-1]  # Reverse channels
#     _, h, w, _ = frames.shape   # numpy is row,col but cv is height,width
#     writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), color)
#     for frame in frames:
#         writer.write(frame)
#     writer.release()
#     print("Saved _video_ at path `{}`".format(path))