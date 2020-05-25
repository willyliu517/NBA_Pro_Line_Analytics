'''
    Module where all helper functions are written
'''
import os
import yaml 
import pandas as pd

def load_yaml_file(path):
    "loads in the yaml file specified in the path"
    with open(path, 'r') as stream:
        try:
            config = yaml.load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
           
        
def ensure_dir_exists(dir_path):
    "checks if the directory exists and creates it if not"
    try:
        os.makedirs(dir_path)
        return dir_path
    except FileExistsError:
        return dir_path