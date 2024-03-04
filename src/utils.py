import os
import glob
import json

def delete_old_files(directory, num_to_keep=10):
    files = sorted(glob.iglob(directory), key=os.path.getctime, reverse=True)
    for i in range(num_to_keep, len(files)):
        os.remove(files[i])
    print("Old files deleted")

def load_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    pass