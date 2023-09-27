
import gym
import numpy as np

from griddly import GymWrapperFactory, gd
from griddly.RenderTools import VideoRecorder
from griddly.RenderTools import RenderToFile

renderer = RenderToFile()

levels = [
    # lv 0
    'wwwwwww\n'
    'wA....w\n'
    'w.w.w.w\n'
    'w.....w\n'
    'w....bw\n'
    'w....cw\n'
    'wwwwwww\n',
    # lv 1
    'wwwwwww\n' \
    'wA....w\n' \
    'w.w.w.w\n' \
    'w.....w\n' \
    'w....bw\n' \
    'wh...cw\n' \
    'wwwwwww\n',
    # lv 2
    'wwwwwww\n'
    'wA....w\n'
    'w.w.w.w\n'
    'w..s..w\n'
    'w....bw\n'
    'wh...cw\n'
    'wwwwwww\n'
]

# TODO:
# Ask for game's yaml file and extract levels automatically
# Do this once when initializing the env with the wrappers and all thar
# we don't want to do it for every env.reset()
def generate_level(n_level, object=None):
    if n_level == 1:
        objects = ['b', 'c', 'h', 'A']
    elif n_level == 2:
        objects = ['b', 'c', 'h', 's', 'A']

    l = levels[n_level]
    for object in objects:
        #l = np.asarray(list(levels[n_level]))       # To numpy
        l = np.asarray(list(l))       # To numpy
        object_loc = np.where(l==object)[0]          # Find object's original position
        l[object_loc] = '.'                          # Set it to empty
        empty_loc = np.where(l=='.')[0]             # Find candidate locations where to put the object
        sampled_loc = np.random.choice(empty_loc)   # Sample one of those locations
        l[sampled_loc] = object                     # Place object in sampled location (here is the agent)
        #gen_level = ''.join(l)                      # Get new generated level string
        l = ''.join(l)                      # Get new generated level string

    return l
    #return gen_level

