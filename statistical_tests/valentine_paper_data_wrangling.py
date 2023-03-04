# Wrangle test results from Valentine paper to be in the same format as my test results
import os
import pickle
import json
from tqdm import tqdm

file_names = []
for root, dirs, files in os.walk("Valentine-output/", topdown=False):
    for name in files:
        if name.endswith(".json"):
            file_names.append(os.path.join(root, name))

# Find all folders with previous test results
dir_names = []
for root, dirs, files in os.walk("Valentine-datasets/", topdown=False):
    for fname in files:
        if fname.endswith("properties.json"):
            dir_names.append(root)

dir_name_location = {}
for dir_name in dir_names:
    experiment = dir_name.split("\\")[-1]
    dir_name_location[experiment] = dir_name

# first, delete all that are not JL_Colnames_only so we get JLcolnames only from my own tests + the rest from Valentine
# paper experiments
for file_name in tqdm(file_names):
    method_name = file_name.split("\\")[-1].replace(".json", "")
    experiment = method_name.split("__")[0]
    if experiment in dir_name_location:
        experiment_folder = dir_name_location[experiment]
        with open(experiment_folder + '/test_results.pkl', 'rb') as fp:
            test_results = pickle.load(fp)
            test_results = {k: v for k, v in test_results.items() if k.startswith('JL_ColumnNamesOnly')}
        with open(experiment_folder + '/test_results.pkl', 'wb') as fp:
            pickle.dump(test_results, fp)
    else:
        raise RuntimeError("experiment dir not found", experiment)

# Then, add all Valentine paper results
for file_name in tqdm(file_names):
    folder_name = file_name.split("\\")[0]
    method_name = file_name.split("\\")[-1].replace(".json", "")
    method = method_name.split("__")[-1]
    experiment = method_name.split("__")[0]
    with open(file_name, 'r') as fp:
        metrics_results = json.load(fp)
    if experiment in dir_name_location:
        experiment_folder = dir_name_location[experiment]
        with open(experiment_folder + '/test_results.pkl', 'rb') as fp:
            test_results = pickle.load(fp)
            test_results[method] = {'matches': metrics_results["matches"],
                                    'metrics': metrics_results["metrics"],
                                    'elapsed_proces_time': metrics_results["run_times"]["total_time"]}
        with open(experiment_folder + '/test_results.pkl', 'wb') as fp:
            pickle.dump(test_results, fp)
    else:
        raise RuntimeError("experiment dir not found", experiment)
