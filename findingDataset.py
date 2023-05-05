import h5py
import numpy as np

# Open the h5 file in read mode
# with h5py.File('SpaceInvaders-v4Image244_Simple Segmentation.h5', 'r') as f:
with h5py.File('SpaceInvaders-v4Image279_Simple Segmentation.h5', 'r') as f:
    for dataset_name in f: #The dataset is 'exported_data', but just in case that doesn't work we use this
        dset = f[dataset_name]
        nparr = np.array(dset).flatten()
        print(np.unique(nparr))


def whichClass(game, num):
    if game == "SpaceInvaders":
        match(num):
            case 1:
                return "Agent"
            case 2:
                return "Enemy"
            case 3:
                return "Background"
            case 4:
                return "Bullet"
            case 5:
                return "Reward"
            case 6:
                return "Wall"
            case 7:
                return "Death"
            case 8:
                return "Boost"
    elif game == "BattleZone":
        match(num):
            case 1:
                return "Agent"
            case 2:
                return "Enemy"
            case 3:
                return "Bullet"
            case 4:
                return "Death"
            case 5:
                return "Background"
            case 6:
                return "Boost"