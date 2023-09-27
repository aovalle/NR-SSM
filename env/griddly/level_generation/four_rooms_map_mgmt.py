import numpy as np
#from four_rooms_maps import maps

maps = [
    # 0
    'W  W  W  W  W  W  W  W  W  W  W  W  W\n' \
    'W  .  .  .  .  .  W  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  W  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  W  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  .  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  W  .  .  .  .  .  W\n' \
    'W  W  W  .  W  W  W  W  .  W  W  W  W\n' \
    'W  .  .  .  .  .  W  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  W  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  W  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  .  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  W  .  .  .  .  .  W\n' \
    'W  W  W  W  W  W  W  W  W  W  W  W  W\n',
    # 1
    'W  W  W  W  W  W  W  W  W  W  W  W  W\n' \
    'W  .  .  .  .  .  f  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  f  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  f  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  .  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  f  .  .  .  .  .  W\n' \
    'W  f  f  .  f  f  W  f  .  f  f  f  W\n' \
    'W  .  .  .  .  .  f  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  f  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  f  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  .  .  .  .  .  .  W\n' \
    'W  .  .  .  .  .  f  .  .  .  .  .  W\n' \
    'W  W  W  W  W  W  W  W  W  W  W  W  W\n',
]

class FourRoomsLevelMgmt():

    def __init__(self, nlevel, seed=None):
        self.objects = {0: ['A', 'g'], 1: ['A', 'g'], 2: ['A', 'g', 'r'], 3: ['A', 'g', 'r'], 4: ['A', 'g', 's'],
                   5: ['A', 'g', 's'], 6: ['A', 'g', 'r', 's'], 7: ['A', 'g', 'r', 's']}

        self.nlevel = nlevel
        self.level = maps[nlevel % 2]
        # Original positions
        self.pos = {'A':421, 'g':141, 'r':44, 's':451}
        self.original_level = self.get_original_level()
        # l = np.asarray(list(self.level))
        # print(np.where(l=='A')[0] )
        # print(np.where(l=='g')[0] )
        # print(np.where(l=='r')[0] )
        # print(np.where(l=='s')[0] )
        #print(self.objects[7])
        if seed:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random.default_rng(seed=np.random.randint(0,99999999))

    ''' Takes default map structure and places the elements '''
    def generate_rand_level(self):
        flat_level = np.asarray(list(self.level))
        empty_loc = np.where(flat_level == '.')[0]  # Find candidate locations where to put the objects

        for obj in self.objects[self.nlevel]:
            #sampled_idx = np.random.randint(0, empty_loc.size) # Sample one (index) of the candidate locations
            sampled_idx = self.rng.integers(0, empty_loc.size) # Sample one (index) of the candidate locations
            flat_level[empty_loc[sampled_idx]] = obj  # Place object in sampled location
            empty_loc = np.delete(empty_loc, sampled_idx)  # Update candidate locations

        return ''.join(flat_level)  # Return properly formatted generated level string

    def move_agent(self, level_string=None):
        if level_string:
            new_level = self.move_object(level_string, 'A')
        else:               # If no string passed we just take the original corresponding level
            new_level = self.move_object(self.original_level, 'A')
        return new_level


    def move_goal(self, level_string=None):
        if level_string:
            new_level = self.move_object(level_string, 'g')
        else:               # If no string passed we just take the original corresponding level
            new_level = self.move_object(self.original_level, 'g')
        return new_level

    def move_object(self, level_string, obj):
        flat_level = np.asarray(list(level_string))
        object_loc = np.where(flat_level == obj)[0]  # Find the object's original position
        flat_level[object_loc] = '.'  # Set it to empty
        empty_loc = np.where(flat_level == '.')[0]  # Find candidate locations where to put the object
        #sampled_idx = np.random.randint(0, empty_loc.size)  # Sample one (index) of the candidate locations
        sampled_idx = self.rng.integers(0, empty_loc.size)  # Sample one (index) of the candidate locations
        flat_level[empty_loc[sampled_idx]] = obj  # Place object in sampled location

        return ''.join(flat_level)  # Return properly formatted generated level string

    def get_original_level(self):
        flat_level = np.asarray(list(self.level))
        for obj in self.objects[self.nlevel]:
            flat_level[self.pos[obj]] = obj
        return ''.join(flat_level)  # Return properly formatted generated level string



# a = FourRoomsLevelMgmt(0)
# a.generate_rand_level()
#
# print(a.move_agent(level_string=maps[0]))
# print(a.move_agent())
