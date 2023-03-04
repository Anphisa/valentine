import os
import json
import pickle
import time
import pandas as pd
from tqdm import tqdm
from valentine.algorithms import Coma, JaccardLevenMatcher, DistributionBased, SimilarityFlooding, Cupid, \
    JaccardLevenMatcherColNamesOnly
from valentine import valentine_match, valentine_metrics

coma_strategies = [('COMA_OPT', Coma(strategy="COMA_OPT")),
                   ('COMA_OPT_INST', Coma(strategy="COMA_OPT_INST"))]
cupid_strategies = [('Cupid_0.0_0.0_0.3', Cupid(0.0, 0.0, 0.3)),
                    ('Cupid_0.0_0.0_0.4', Cupid(0.0, 0.0, 0.4)),
                    ('Cupid_0.0_0.0_0.5', Cupid(0.0, 0.0, 0.5)),
                    ('Cupid_0.0_0.0_0.6', Cupid(0.0, 0.0, 0.6)),
                    ('Cupid_0.0_0.0_0.7', Cupid(0.0, 0.0, 0.7)),
                    ('Cupid_0.0_0.0_0.8', Cupid(0.0, 0.0, 0.8)),
                    ('Cupid_0.0_0.0_0.9', Cupid(0.0, 0.0, 0.9)),
                    ('Cupid_0.0_0.2_0.3', Cupid(0.0, 0.2, 0.3)),
                    ('Cupid_0.0_0.2_0.4', Cupid(0.0, 0.2, 0.4)),
                    ('Cupid_0.0_0.2_0.5', Cupid(0.0, 0.2, 0.5)),
                    ('Cupid_0.0_0.2_0.6', Cupid(0.0, 0.2, 0.6)),
                    ('Cupid_0.0_0.2_0.7', Cupid(0.0, 0.2, 0.7)),
                    ('Cupid_0.0_0.2_0.8', Cupid(0.0, 0.2, 0.8)),
                    ('Cupid_0.0_0.2_0.9', Cupid(0.0, 0.2, 0.9)),
                    ('Cupid_0.0_0.4_0.3', Cupid(0.0, 0.4, 0.3)),
                    ('Cupid_0.0_0.4_0.4', Cupid(0.0, 0.4, 0.4)),
                    ('Cupid_0.0_0.4_0.5', Cupid(0.0, 0.4, 0.5)),
                    ('Cupid_0.0_0.4_0.6', Cupid(0.0, 0.4, 0.6)),
                    ('Cupid_0.0_0.4_0.7', Cupid(0.0, 0.4, 0.7)),
                    ('Cupid_0.0_0.4_0.8', Cupid(0.0, 0.4, 0.8)),
                    ('Cupid_0.0_0.4_0.9', Cupid(0.0, 0.4, 0.9)),
                    ('Cupid_0.0_0.6_0.3', Cupid(0.0, 0.6, 0.3)),
                    ('Cupid_0.0_0.6_0.4', Cupid(0.0, 0.6, 0.4)),
                    ('Cupid_0.0_0.6_0.5', Cupid(0.0, 0.6, 0.5)),
                    ('Cupid_0.0_0.6_0.6', Cupid(0.0, 0.6, 0.6)),
                    ('Cupid_0.0_0.6_0.7', Cupid(0.0, 0.6, 0.7)),
                    ('Cupid_0.0_0.6_0.8', Cupid(0.0, 0.6, 0.8)),
                    ('Cupid_0.0_0.6_0.9', Cupid(0.0, 0.6, 0.9)),
                    ('Cupid_0.2_0.0_0.3', Cupid(0.2, 0.0, 0.3)),
                    ('Cupid_0.2_0.0_0.4', Cupid(0.2, 0.0, 0.4)),
                    ('Cupid_0.2_0.0_0.5', Cupid(0.2, 0.0, 0.5)),
                    ('Cupid_0.2_0.0_0.6', Cupid(0.2, 0.0, 0.6)),
                    ('Cupid_0.2_0.0_0.7', Cupid(0.2, 0.0, 0.7)),
                    ('Cupid_0.2_0.0_0.8', Cupid(0.2, 0.0, 0.8)),
                    ('Cupid_0.2_0.0_0.9', Cupid(0.2, 0.0, 0.9)),
                    ('Cupid_0.2_0.2_0.3', Cupid(0.2, 0.2, 0.3)),
                    ('Cupid_0.2_0.2_0.4', Cupid(0.2, 0.2, 0.4)),
                    ('Cupid_0.2_0.2_0.5', Cupid(0.2, 0.2, 0.5)),
                    ('Cupid_0.2_0.2_0.6', Cupid(0.2, 0.2, 0.6)),
                    ('Cupid_0.2_0.2_0.7', Cupid(0.2, 0.2, 0.7)),
                    ('Cupid_0.2_0.2_0.8', Cupid(0.2, 0.2, 0.8)),
                    ('Cupid_0.2_0.2_0.9', Cupid(0.2, 0.2, 0.9)),
                    ('Cupid_0.2_0.4_0.3', Cupid(0.2, 0.4, 0.3)),
                    ('Cupid_0.2_0.4_0.4', Cupid(0.2, 0.4, 0.4)),
                    ('Cupid_0.2_0.4_0.5', Cupid(0.2, 0.4, 0.5)),
                    ('Cupid_0.2_0.4_0.6', Cupid(0.2, 0.4, 0.6)),
                    ('Cupid_0.2_0.4_0.7', Cupid(0.2, 0.4, 0.7)),
                    ('Cupid_0.2_0.4_0.8', Cupid(0.2, 0.4, 0.8)),
                    ('Cupid_0.2_0.4_0.9', Cupid(0.2, 0.4, 0.9)),
                    ('Cupid_0.2_0.6_0.3', Cupid(0.2, 0.6, 0.3)),
                    ('Cupid_0.2_0.6_0.4', Cupid(0.2, 0.6, 0.4)),
                    ('Cupid_0.2_0.6_0.5', Cupid(0.2, 0.6, 0.5)),
                    ('Cupid_0.2_0.6_0.6', Cupid(0.2, 0.6, 0.6)),
                    ('Cupid_0.2_0.6_0.7', Cupid(0.2, 0.6, 0.7)),
                    ('Cupid_0.2_0.6_0.8', Cupid(0.2, 0.6, 0.8)),
                    ('Cupid_0.2_0.6_0.9', Cupid(0.2, 0.6, 0.9)),
                    ('Cupid_0.4_0.0_0.3', Cupid(0.4, 0.0, 0.3)),
                    ('Cupid_0.4_0.0_0.4', Cupid(0.4, 0.0, 0.4)),
                    ('Cupid_0.4_0.0_0.5', Cupid(0.4, 0.0, 0.5)),
                    ('Cupid_0.4_0.0_0.6', Cupid(0.4, 0.0, 0.6)),
                    ('Cupid_0.4_0.0_0.7', Cupid(0.4, 0.0, 0.7)),
                    ('Cupid_0.4_0.0_0.8', Cupid(0.4, 0.0, 0.8)),
                    ('Cupid_0.4_0.0_0.9', Cupid(0.4, 0.0, 0.9)),
                    ('Cupid_0.4_0.2_0.3', Cupid(0.4, 0.2, 0.3)),
                    ('Cupid_0.4_0.2_0.4', Cupid(0.4, 0.2, 0.4)),
                    ('Cupid_0.4_0.2_0.5', Cupid(0.4, 0.2, 0.5)),
                    ('Cupid_0.4_0.2_0.6', Cupid(0.4, 0.2, 0.6)),
                    ('Cupid_0.4_0.2_0.7', Cupid(0.4, 0.2, 0.7)),
                    ('Cupid_0.4_0.2_0.8', Cupid(0.4, 0.2, 0.8)),
                    ('Cupid_0.4_0.2_0.9', Cupid(0.4, 0.2, 0.9)),
                    ('Cupid_0.4_0.4_0.3', Cupid(0.4, 0.4, 0.3)),
                    ('Cupid_0.4_0.4_0.4', Cupid(0.4, 0.4, 0.4)),
                    ('Cupid_0.4_0.4_0.5', Cupid(0.4, 0.4, 0.5)),
                    ('Cupid_0.4_0.4_0.6', Cupid(0.4, 0.4, 0.6)),
                    ('Cupid_0.4_0.4_0.7', Cupid(0.4, 0.4, 0.7)),
                    ('Cupid_0.4_0.4_0.8', Cupid(0.4, 0.4, 0.8)),
                    ('Cupid_0.4_0.4_0.9', Cupid(0.4, 0.4, 0.9)),
                    ('Cupid_0.4_0.6_0.3', Cupid(0.4, 0.6, 0.3)),
                    ('Cupid_0.4_0.6_0.4', Cupid(0.4, 0.6, 0.4)),
                    ('Cupid_0.4_0.6_0.5', Cupid(0.4, 0.6, 0.5)),
                    ('Cupid_0.4_0.6_0.6', Cupid(0.4, 0.6, 0.6)),
                    ('Cupid_0.4_0.6_0.7', Cupid(0.4, 0.6, 0.7)),
                    ('Cupid_0.4_0.6_0.8', Cupid(0.4, 0.6, 0.8)),
                    ('Cupid_0.4_0.6_0.9', Cupid(0.4, 0.6, 0.9)),
                    ('Cupid_0.6_0.0_0.3', Cupid(0.6, 0.0, 0.3)),
                    ('Cupid_0.6_0.0_0.4', Cupid(0.6, 0.0, 0.4)),
                    ('Cupid_0.6_0.0_0.5', Cupid(0.6, 0.0, 0.5)),
                    ('Cupid_0.6_0.0_0.6', Cupid(0.6, 0.0, 0.6)),
                    ('Cupid_0.6_0.0_0.7', Cupid(0.6, 0.0, 0.7)),
                    ('Cupid_0.6_0.0_0.8', Cupid(0.6, 0.0, 0.8)),
                    ('Cupid_0.6_0.0_0.9', Cupid(0.6, 0.0, 0.9)),
                    ('Cupid_0.6_0.2_0.3', Cupid(0.6, 0.2, 0.3)),
                    ('Cupid_0.6_0.2_0.4', Cupid(0.6, 0.2, 0.4)),
                    ('Cupid_0.6_0.2_0.5', Cupid(0.6, 0.2, 0.5)),
                    ('Cupid_0.6_0.2_0.6', Cupid(0.6, 0.2, 0.6)),
                    ('Cupid_0.6_0.2_0.7', Cupid(0.6, 0.2, 0.7)),
                    ('Cupid_0.6_0.2_0.8', Cupid(0.6, 0.2, 0.8)),
                    ('Cupid_0.6_0.2_0.9', Cupid(0.6, 0.2, 0.9)),
                    ('Cupid_0.6_0.4_0.3', Cupid(0.6, 0.4, 0.3)),
                    ('Cupid_0.6_0.4_0.4', Cupid(0.6, 0.4, 0.4)),
                    ('Cupid_0.6_0.4_0.5', Cupid(0.6, 0.4, 0.5)),
                    ('Cupid_0.6_0.4_0.6', Cupid(0.6, 0.4, 0.6)),
                    ('Cupid_0.6_0.4_0.7', Cupid(0.6, 0.4, 0.7)),
                    ('Cupid_0.6_0.4_0.8', Cupid(0.6, 0.4, 0.8)),
                    ('Cupid_0.6_0.4_0.9', Cupid(0.6, 0.4, 0.9)),
                    ('Cupid_0.6_0.6_0.3', Cupid(0.6, 0.6, 0.3)),
                    ('Cupid_0.6_0.6_0.4', Cupid(0.6, 0.6, 0.4)),
                    ('Cupid_0.6_0.6_0.5', Cupid(0.6, 0.6, 0.5)),
                    ('Cupid_0.6_0.6_0.6', Cupid(0.6, 0.6, 0.6)),
                    ('Cupid_0.6_0.6_0.7', Cupid(0.6, 0.6, 0.7)),
                    ('Cupid_0.6_0.6_0.8', Cupid(0.6, 0.6, 0.8))]
distribution_strategies = [('DistributionBased_0.3_0.3', DistributionBased(0.3, 0.3)),
                           ('DistributionBased_0.3_0.4', DistributionBased(0.3, 0.4)),
                           ('DistributionBased_0.3_0.5', DistributionBased(0.3, 0.5)),
                           ('DistributionBased_0.4_0.3', DistributionBased(0.4, 0.3)),
                           ('DistributionBased_0.4_0.4', DistributionBased(0.4, 0.4)),
                           ('DistributionBased_0.4_0.5', DistributionBased(0.4, 0.5)),
                           ('DistributionBased_0.5_0.3', DistributionBased(0.5, 0.3)),
                           ('DistributionBased_0.5_0.4', DistributionBased(0.5, 0.4)),
                           ('DistributionBased_0.5_0.5', DistributionBased(0.5, 0.5))]
similarity_strategies = [('SimilarityFlooding', SimilarityFlooding())]
naive_strategies = [('JL_ColumnNamesOnly_0.4', JaccardLevenMatcherColNamesOnly(0.4)),
                    ('JL_ColumnNamesOnly_0.5', JaccardLevenMatcherColNamesOnly(0.5)),
                    ('JL_ColumnNamesOnly_0.6', JaccardLevenMatcherColNamesOnly(0.6)),
                    ('JL_ColumnNamesOnly_0.7', JaccardLevenMatcherColNamesOnly(0.7)),
                    ('JL_ColumnNamesOnly_0.8', JaccardLevenMatcherColNamesOnly(0.8))]
naive_slow_strategies = [('JaccardLevenMatcher_0.4', JaccardLevenMatcher(0.4)),
                         ('JaccardLevenMatcher_0.5', JaccardLevenMatcher(0.5)),
                         ('JaccardLevenMatcher_0.6', JaccardLevenMatcher(0.6)),
                         ('JaccardLevenMatcher_0.7', JaccardLevenMatcher(0.7)),
                         ('JaccardLevenMatcher_0.8', JaccardLevenMatcher(0.8))]
all_strategies = naive_strategies

# Find all folders that contain csv files to match
dir_names = []
for root, dirs, files in os.walk("Valentine-datasets/", topdown=False):
    for fname in files:
        if fname.endswith("properties.json"):
            dir_names.append(root)

print(len(dir_names))

for comp_dir in tqdm(dir_names):
    pkl_exists = False
    for fname in os.listdir(comp_dir):
        if fname.endswith("results.pkl"):
            pkl_exists = True
            with open(os.path.join(comp_dir, fname), 'rb') as fp:
                pkl = pickle.load(fp)
    if pkl_exists:
        redo_strategies = []
        for strategy in all_strategies:
            if strategy[0] not in pkl or str(pkl[strategy[0]]['matches']).startswith('Matcher failed!'):
                redo_strategies.append(strategy)
        print("redoing {} strategies".format(len(redo_strategies)))

    comp_dir = comp_dir + "/"
    print("DIRECTORY:", comp_dir)
    csv_files = []
    csv_names = []
    test_results = {}
    metrics_results = {}
    for file in os.listdir(comp_dir):
        if file.endswith(".csv"):
            csv_files.append(os.path.join(comp_dir, file))
            csv_names.append(file)
        if file.endswith("mapping.json"):
            ground_truth_file = os.path.join(comp_dir, file)
            with open(ground_truth_file, encoding="utf-8") as json_file:
                ground_truth_matches = json.load(json_file)
        if file == "properties.json":
            properties_file = os.path.join(comp_dir, file)
            with open(properties_file, encoding="utf-8") as json_file:
                properties = json.load(json_file)
    if len(csv_files) == 2:
        df1 = pd.read_csv(csv_files[0])
        df1_name = csv_names[0].split('.')[0]
        df2 = pd.read_csv(csv_files[1])
        df2_name = csv_names[1].split('.')[0]
    else:
        raise RuntimeError("No two csv files found!")

    # Validate matches and build ground truth list for Valentine's metrics comparison
    ground_truth = []
    for m in ground_truth_matches["matches"]:
        if m["source_table"] == df1_name and m["target_table"] == df2_name:
            if m["source_column"] in df1.columns and m["target_column"] in df2.columns:
                ground_truth.append((m["source_column"], m["target_column"]))
            else:
                raise RuntimeError("Column name error!", m)
        elif m["source_table"] == df2_name and m["target_table"] == df1_name:
            if m["source_column"] in df2.columns and m["target_column"] in df1.columns:
                ground_truth.append((m["target_column"], m["source_column"]))
            else:
                raise RuntimeError("Column name error!", m)
        else:
            raise RuntimeError("Table name error!", m)
    # print(ground_truth)

    if pkl_exists:
        if redo_strategies:
            for strategy in tqdm(redo_strategies):
                start = time.process_time()
                strategy_name = strategy[0]
                print("STRATEGY", strategy_name)
                matcher = strategy[1]
                try:
                    matches = valentine_match(df1, df2, matcher)
                    metrics = valentine_metrics.all_metrics(matches, ground_truth)
                except Exception as e:
                    print("matcher failed", e)
                    matches = 'Matcher failed! ' + str(e)
                    metrics = 'Matcher failed! ' + str(e)
                    pass

                elapsed_time = time.process_time() - start

                test_results[strategy_name] = {'matches': matches,
                                               'metrics': metrics,
                                               'elapsed_proces_time': elapsed_time}
                metrics_results[strategy_name] = metrics
                # print(metrics_results)

            with open(comp_dir + 'test_results.pkl', 'rb') as fp:
                previous_test_results = pickle.load(fp)
            with open(comp_dir + 'test_metrics_results.json', 'r') as fp:
                previous_metrics_results = json.load(fp)

            updated_test_results = previous_test_results | test_results
            updated_metrics_results = previous_metrics_results | metrics_results

            with open(comp_dir + 'test_results.pkl', 'wb') as fp:
                pickle.dump(updated_test_results, fp)
            with open(comp_dir + 'test_metrics_results.json', 'w') as fp:
                json.dump(updated_metrics_results, fp)
    if not pkl_exists:
        # Try out all strategy/parameter combinations we're interested in
        for strategy in tqdm(all_strategies):
            start = time.process_time()
            strategy_name = strategy[0]
            print("STRATEGY", strategy_name)
            matcher = strategy[1]
            try:
                matches = valentine_match(df1, df2, matcher)
                metrics = valentine_metrics.all_metrics(matches, ground_truth)
            except Exception as e:
                print("matcher failed", e)
                matches = 'Matcher failed! ' + str(e)
                metrics = 'Matcher failed! ' + str(e)
                pass

            elapsed_time = time.process_time() - start
            test_results[strategy_name] = {'matches': matches,
                                           'metrics': metrics,
                                           'elapsed_proces_time': elapsed_time}
            metrics_results[strategy_name] = metrics
            # print(metrics_results)

        with open(comp_dir + 'test_results.pkl', 'wb') as fp:
            pickle.dump(test_results, fp)
        with open(comp_dir + 'test_metrics_results.json', 'w') as fp:
            json.dump(metrics_results, fp)
