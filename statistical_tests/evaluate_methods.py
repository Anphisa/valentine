import json
import os
from tqdm import tqdm
import pickle
import pandas as pd

# Find all folders that contain a comparison
dir_names = []
for root, dirs, files in os.walk(".", topdown=False):
    for name in dirs:
        for fname in os.listdir(os.path.join(root, name)):
            if fname == "test_results.pkl":
                dir_names.append(os.path.join(root, name))

results = {}
for comp_dir in tqdm(dir_names):
    comp_dir = comp_dir + "/"
    experiment_name = comp_dir.split('\\')[-1][:-1]
    for fname in os.listdir(comp_dir):
        if fname.endswith("test_results.pkl"):
            with open(comp_dir + 'test_results.pkl', 'rb') as fp:
                test_results = pickle.load(fp)
                with open(comp_dir + 'properties.json', 'r') as fp_properties:
                    properties = json.load(fp_properties)
                    join_type = properties["relationship_type"]
                    schema_noisy_verbatim = properties.get("schema_noisy_verbatim", "verbatim")
                    instances_noisy_verbatim = properties.get("instances_noisy_verbatim", "verbatim")
                for method in test_results:
                    if experiment_name not in results:
                        results[experiment_name] = {}
                    results[experiment_name][method] = {
                        'elapsed_process_time': test_results[method]['elapsed_proces_time'],
                        'metrics': test_results[method]['metrics'],
                        'join_type': join_type,
                        'schema_noisy_verbatim': schema_noisy_verbatim,
                        'instances_noisy_verbatim': instances_noisy_verbatim}

df = pd.DataFrame.from_dict(results, orient='index')
df = pd.melt(df.reset_index(), id_vars=['index'])
df.rename(columns={'index': 'experiment_name', 'variable': 'method', 'value': 'metrics'}, inplace=True)
method_results = pd.concat([df.drop(['metrics'], axis=1), df['metrics'].apply(pd.Series)], axis=1)
method_results = pd.concat([method_results.drop(['metrics'], axis=1), method_results['metrics'].apply(pd.Series)], axis=1)
method_results.to_csv("results/method_results.csv", index=False)
