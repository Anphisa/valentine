# What is the best method? (according to some decision, e.g. highest recall@ground truth)
# And what are the other dataset features?
import json
import os
import statistics
import numpy as np
import pandas as pd
from tqdm import tqdm

# Find all folders that contain a comparison
dir_names = []
for root, dirs, files in os.walk(".", topdown=False):
    for name in dirs:
        for fname in os.listdir(os.path.join(root, name)):
            if fname == "test_metrics_results.json":
                dir_names.append(os.path.join(root, name))


def find_best_method(ex):
    method_results = pd.read_csv("results/method_results.csv")
    experiment_results = method_results[method_results["experiment_name"] == ex]
    max_idx = experiment_results.loc[experiment_results['recall_at_sizeof_ground_truth'] == experiment_results['recall_at_sizeof_ground_truth'].max()].index
    if len(max_idx) > 1:
        # Multiple best performers, break ties by lowest execution time
        best_performers = experiment_results.loc[max_idx]
        min_times_idx = best_performers.loc[best_performers['elapsed_process_time'] == best_performers['elapsed_process_time'].min()].index
        best_performing_methods = best_performers.loc[min_times_idx]["method"]
        recall_at_sizeof_ground_truth = experiment_results.loc[min_times_idx]["recall_at_sizeof_ground_truth"]
        min_times = experiment_results.loc[min_times_idx]["elapsed_process_time"]
        return {'best_performing_methods': best_performing_methods.tolist(),
                'min_times': min_times.tolist(),
                'recall_at_sizeof_ground_truth': recall_at_sizeof_ground_truth.tolist()}
    else:
        best_performing_methods = experiment_results.loc[max_idx]["method"]
        min_times = experiment_results.loc[max_idx]["elapsed_process_time"]
        recall_at_sizeof_ground_truth = experiment_results.loc[max_idx]["recall_at_sizeof_ground_truth"]
        return {'best_performing_methods': best_performing_methods.tolist(),
                'min_times': min_times.tolist(),
                'recall_at_sizeof_ground_truth': recall_at_sizeof_ground_truth.tolist()}


results = {}
for comp_dir in tqdm(dir_names):
    comp_dir = comp_dir + "/"
    # print("Directory:", comp_dir)
    experiment_name = comp_dir.split('\\')[-1][:-1]
    print("Experiment name:", experiment_name)
    best_methods = find_best_method(experiment_name)
    for fname in os.listdir(comp_dir):
        if fname.endswith("properties.json"):
            with open(comp_dir + 'properties.json', 'r') as fp:
                properties = json.load(fp)
                if experiment_name not in results:
                    results[experiment_name] = {}
                properties_reformat = {'cleaned': properties['cleaned'],
                                       'percentage_cols_involved_in_rel': properties['cols_involved_in_rel']['relative'].values(),
                                       'column_names_string_distance': properties['column_names_string_distance'],
                                       # 'dataset_class_size': properties['datasets_class']['size'],
                                       'dataset_class_type': properties.get('datasets_class', {"type": "no dataset class type"})['type'],
                                       'datasets_nrows': [i[0] for i in properties['df_shapes'].values()],
                                       'datasets_ncols': [i[1] for i in properties['df_shapes'].values()],
                                       'rel_numeric': properties['joins_on_numeric'],
                                       'rel_string': properties['joins_on_string'],
                                       'avg_obfuscators_by_colname': [np.mean(properties['obfuscator_columns_by_colname'][i], axis=0) for i in properties['obfuscator_columns_by_colname']],
                                       'avg_obfuscators_by_dtype': [np.mean(properties['obfuscator_columns_by_datatype'][i], axis=0) for i in properties['obfuscator_columns_by_datatype']],
                                       'provenance': properties['provenance'],
                                       'split': properties.get("split", ""),
                                       'schema_noisy_verbatim': properties.get("schema_noisy_verbatim", "verbatim"),
                                       'instances_noisy_verbatim': properties.get("instances_noisy_verbatim", "verbatim"),
                                       'relationship_type': properties['relationship_type'],
                                       'avg_intersection_perc_5_samples': properties.get('avg_intersection_perc_5', 0),
                                       'avg_intersection_perc_10_samples': properties.get('avg_intersection_perc_10', 0),
                                       'avg_intersection_perc_15_samples': properties.get('avg_intersection_perc_15', 0),
                                       'avg_intersection_perc_50_samples': properties.get('avg_intersection_perc_50', 0),
                                       'avg_intersection_perc_100_samples': properties.get('avg_intersection_perc_100', 0),
                                       'avg_exact_col_match': properties.get('avg_exact_col_match', 0),
                                       'avg_fuzzy_col_match': properties.get('avg_fuzzy_col_match', 0),
                                       'avg_null_attributes': properties['avg_null_attributes']}
                str_dists = []
                for rel in properties['matches']['matches']:
                    source_col = rel['source_column']
                    target_col = rel['target_column']
                    # Sometimes the relationship is defined "the other way round"
                    if source_col in properties['avg_col_lengths'][0]:
                        str_dist = properties['avg_col_lengths'][0][source_col] - \
                                   properties['avg_col_lengths'][1][target_col]
                    else:
                        str_dist = properties['avg_col_lengths'][1][source_col] - \
                                   properties['avg_col_lengths'][0][target_col]
                    str_dists.append(str_dist)
                properties_reformat['avg_str_dists'] = sum(str_dists) / len(str_dists)
                properties_reformat['median_str_dists'] = statistics.median(str_dists)
                properties_reformat['raw_str_dists'] = str_dists
                unique_values_diffs = []
                for rel in properties['matches']['matches']:
                    source_col = rel['source_column']
                    target_col = rel['target_column']
                    if source_col in properties['unique_values_in_relationship_columns'][0]:
                        unique_values_diff = properties['unique_values_in_relationship_columns'][0][source_col] - \
                                             properties['unique_values_in_relationship_columns'][1][target_col]
                    else:
                        unique_values_diff = properties['unique_values_in_relationship_columns'][1][source_col] - \
                                             properties['unique_values_in_relationship_columns'][0][target_col]
                    unique_values_diffs.append(unique_values_diff)
                properties_reformat['avg_unique_values_dists'] = sum(unique_values_diffs) / len(unique_values_diffs)
                properties_reformat['median_unique_values_dists'] = statistics.median(unique_values_diffs)
                properties_reformat['raw_unique_values_dists'] = unique_values_diffs
                results[experiment_name] = {'best_methods': best_methods,
                                            'properties': properties_reformat}

df = pd.DataFrame.from_dict(results, orient='index')
print(df.head())
best_method_properties = pd.concat([df.drop(['properties'], axis=1), df['properties'].apply(pd.Series)], axis=1)
print(best_method_properties.head())
best_method_properties = pd.concat([best_method_properties.drop(['split'], axis=1),
                                    best_method_properties['split'].apply(pd.Series)], axis=1)
best_method_properties = pd.concat([best_method_properties.drop(['split_amount'], axis=1),
                                    best_method_properties['split_amount'].apply(pd.Series)], axis=1)
print(best_method_properties.head())
best_method_properties.to_csv("results/best_method_properties.csv")
best_method_properties.to_excel("results/best_method_properties.xlsx")
