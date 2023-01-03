import os

dir = "data/katakana/"

def del_symbol(dir):
    path = os.listdir(dir)
    for file in path:
        if not '.jpeg' in file:
            new_dir = dir + '/' + file
            del_symbol(new_dir)
        if '[' in file:
            new_name = file.replace('[', '').replace(']', '')
            os.rename(os.path.join(dir, file), os.path.join(dir, new_name))
            