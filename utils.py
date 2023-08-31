import pandas as pd
import dill as pickle
from collections import defaultdict
from scipy.spatial import distance
import json
from pathlib import Path
import pandas as pd
from py_stringsimjoin.index.inverted_index import InvertedIndex
from py_stringsimjoin.utils.generic_helper import (
    COMP_OP_MAP,
)
from py_stringmatching.tokenizer.whitespace_tokenizer import WhitespaceTokenizer
from six import iteritems
from overlap_blocker_helper import get_attrs_to_project, find_candidates
import copy
import math


def single_fairness(
    candidate_set,
    ltable_sensitive_attr,
    rtable_sensitive_attr,
    ltable_id,
    rtable_id,
    distributions,
    threshold=10,
):
    """Measures the single fairness in the candidate set
    Args:
        candidate_set (DataFrame): the output of the blocking method
        ltable_sensitive_attr (DataFrame): the DataFrame of the sensitive attribute for TableA
        rtable_sensitive_attr (DataFrame): the DataFrame of the sensitive attribute for TableB
        ltable_id (String): the key attribute in the DataFrame of the sensitive attribute for TableA
        rtable_id (String): the key attribute in the DataFrame of the sensitive attribute for TableB
        distributions (Dictionary): the distribution of groups in the cartesian product of the two tables
        threshold (Integer): removes the groups with frequency less than specified threshold
    Return:
        disparity (float): division-based disparity compuation between the group with maximum statistical parity and the group with minimum statistical parity
        max_sp (float): maximum statistical parity value
        max_sp_group (String): group with maximum statistical parity value
        min_sp (float): minimum statistical parity value
        min_sp_group (String): group with minimum statistical parity value
    """
    sensitive_attrs = candidate_set_cardinality(
        candidate_set,
        ltable_sensitive_attr,
        rtable_sensitive_attr,
        ltable_id,
        rtable_id,
        threshold=threshold,
        type="single",
    )
    sp = {}
    for key, value in sensitive_attrs.items():
        sp[key] = value / distributions[key]

    max_sp = sp[max(sp, key=sp.get)]
    max_sp_group = max(sp, key=sp.get)
    min_sp = sp[min(sp, key=sp.get)]
    min_sp_group = min(sp, key=sp.get)
    disparity = (max_sp / min_sp) - 1

    return disparity, max_sp, max_sp_group, min_sp, min_sp_group


def pairwise_fairness(
    candidate_set,
    ltable_sensitive_attr,
    rtable_sensitive_attr,
    ltable_id,
    rtable_id,
    distributions,
    threshold=10,
):
    """Measures the pairwise fairness in the candidate set
    Args:
        candidate_set (DataFrame): the output of the blocking method
        ltable_sensitive_attr (DataFrame): the DataFrame of the sensitive attribute for TableA
        rtable_sensitive_attr (DataFrame): the DataFrame of the sensitive attribute for TableB
        ltable_id (String): the key attribute in the DataFrame of the sensitive attribute for TableA
        rtable_id (String): the key attribute in the DataFrame of the sensitive attribute for TableB
        distributions (Dictionary): the distribution of groups in the cartesian product of the two tables
        threshold (Integer): removes the groups with frequency less than specified threshold
    Return:
        disparity (float): division-based disparity compuation between the group with maximum statistical parity and the group with minimum statistical parity
        max_sp (float): maximum statistical parity value
        max_sp_group (String): group with maximum statistical parity value
        min_sp (float): minimum statistical parity value
        min_sp_group (String): group with minimum statistical parity value
    """
    sensitive_attr_pairs = candidate_set_cardinality(
        candidate_set,
        ltable_sensitive_attr,
        rtable_sensitive_attr,
        ltable_id,
        rtable_id,
        threshold=threshold,
        type="pairwise",
    )
    sp = {}
    for key, value in sensitive_attr_pairs.items():
        sp[key] = value / distributions[key]

    max_sp = sp[max(sp, key=sp.get)]
    max_sp_group = max(sp, key=sp.get)
    min_sp = sp[min(sp, key=sp.get)]
    min_sp_group = min(sp, key=sp.get)
    disparity = (max_sp / min_sp) - 1

    return disparity, max_sp, max_sp_group, min_sp, min_sp_group


def recall(candidate_set, golden_set, ltable_id="ltable_id", rtable_id="rtable_id"):
    """Calculates the recall for the candidate set
    Args:
        candidate_set (DataFrame): the output of the blocking method
        golden_set (DataFrame): table with the ground-truth values
        ltable_id (String): ID of the left entity set in both of candidate_set and golden_set tables (should be identical in both tables)
        rtable_id (String): ID of the right entity set in both of candidate_set and golden_set tables (should be identical in both tables)
    Return:
        recall (Float): recall value of the candidate set
    """
    positive_candidate_set = pd.merge(
        candidate_set, golden_set, on=[ltable_id, rtable_id]
    )
    return len(positive_candidate_set) / len(golden_set)


def single_table_cardinality(sensitive_attr, threshold=10):
    """Calculate the cardinality of each group in a single table
    Args:
        table (DataFrame): input table
        sensitive_attr (String): the column name of the sensitive attribute in the table
        threshold (Integer): removes the groups with frequency less than specified threshold
    Return:
        cardinality (Dictionary): a dictionary in which the group name is the key and the cardinality is the value for each group
    """
    cardinality = {}

    for i in range(len(sensitive_attr[sensitive_attr.columns[1]].value_counts())):
        if not pd.isna(
            sensitive_attr[sensitive_attr.columns[1]].value_counts().index[i]
        ):
            cardinality[
                sensitive_attr[sensitive_attr.columns[1]
                               ].value_counts().index[i]
            ] = sensitive_attr[sensitive_attr.columns[1]].value_counts()[i]
        else:
            continue
    cardinality = remove_low_cardinality_groups(cardinality, threshold)
    return cardinality


def candidate_set_cardinality(
    candidate_set,
    ltable_sensitive_attr,
    rtable_sensitive_attr,
    ltable_id,
    rtable_id,
    threshold=10,
    type="pairwise",
    map=None,
):
    """Calcute the cardinality of each group in the candidate set
    Args:
        candidate_set (DataFrame): the output of the blocking method
        ltable_sensitive_attr (String): the column name of the sensitive attribute for the left entities
        rtable_sensitive_attr (String): the column name of the sensitive attribute for the right entities
        ltable_id (String): the key attribute in the DataFrame of the sensitive attribute for TableA
        rtable_id (String): the key attribute in the DataFrame of the sensitive attribute for TableB
        threshold (Integer): removes the groups with frequency less than specified threshold
        type (String): determines the single or pairwise distributions of the groups in the candidate set
    Return:
        cardinality (Dictionary): a dictionary in which the group name is the key and the cardinality is the value for each group
    """
    cardinality = {}
    if type == "pairwise":
        for idx, row in candidate_set.iterrows():
            # print(idx, candidate_set.shape[0])
            key_left = ltable_sensitive_attr.loc[
                ltable_sensitive_attr[ltable_sensitive_attr.columns[0]]
                == row[ltable_id],
                ltable_sensitive_attr.columns[1],
            ].values[0]
            key_right = rtable_sensitive_attr.loc[
                rtable_sensitive_attr[rtable_sensitive_attr.columns[0]]
                == row[rtable_id],
                rtable_sensitive_attr.columns[1],
            ].values[0]

            if map is not None:
                key_left = map[key_left]
                key_right = map[key_right]

            if not pd.isna(key_left) and not pd.isna(key_right):
                key = str((key_left, key_right))
                if key in cardinality.keys():
                    cardinality[key] += 1
                else:
                    cardinality[key] = 1
            else:
                continue
    elif type == "single":
        for idx, row in candidate_set.iterrows():
            # print(idx, candidate_set.shape[0])
            key_left = ltable_sensitive_attr.loc[
                ltable_sensitive_attr[ltable_sensitive_attr.columns[0]]
                == row[ltable_id],
                ltable_sensitive_attr.columns[1],
            ].values[0]
            key_right = rtable_sensitive_attr.loc[
                rtable_sensitive_attr[rtable_sensitive_attr.columns[0]]
                == row[rtable_id],
                rtable_sensitive_attr.columns[1],
            ].values[0]

            if map is not None:
                if key_left in map.keys():
                    key_left = map[key_left]
                if key_right in map.keys():
                    key_right = map[key_right]

            if not pd.isna(key_left) and not pd.isna(key_right):
                if key_left in cardinality.keys():
                    cardinality[key_left] += 1
                else:
                    cardinality[key_left] = 1
                if key_right in cardinality.keys():
                    cardinality[key_right] += 1
                else:
                    cardinality[key_right] = 1
            else:
                continue
    cardinality = remove_low_cardinality_groups(cardinality, threshold)
    return cardinality


def positive_set_cardinality(
    candidate_set,
    golden_set,
    ltable_sensitive_attr,
    rtable_sensitive_attr,
    ltable_id="ltable_id",
    rtable_id="rtable_id",
):
    """Calcute the cardinality of each group among the true positive set in the candidate set
    Args:
        candidate_set (DataFrame): the output of the blocking method
        golden_set (DataFrame): table with the ground-truth values
        ltable_sensitive_attr (String): the column name of the sensitive attribute for the left entities
        rtable_sensitive_attr (String): the column name of the sensitive attribute for the right entities
        ltable_id (String): ID of the left entity set in both of candidate_set and golden_set tables (should be identical in both tables)
        rtable_id (String): ID of the right entity set in both of candidate_set and golden_set tables (should be identical in both tables)
        ordered (Boolean): sort the groups in the output in an descending order of the cardinalities
    Return:
        cardinality (Dictionary): a dictionary in which the group name is the key and the cardinality is the value for each group
    """
    positive_candidate_set = pd.merge(
        candidate_set, golden_set, on=[ltable_id, rtable_id]
    )
    return candidate_set_cardinality(
        positive_candidate_set,
        ltable_sensitive_attr,
        rtable_sensitive_attr,
        ltable_id,
        rtable_id,
    )


def negative_set_cardinality(
    candidate_set,
    golden_set,
    ltable_sensitive_attr,
    rtable_sensitive_attr,
    ltable_id="ltable_id",
    rtable_id="rtable_id",
):
    """Calcute the cardinality of each group among the false positive set in the candidate set
    Args:
        candidate_set (DataFrame): the output of the blocking method
        golden_set (DataFrame): table with the ground-truth values
        ltable_sensitive_attr (String): the column name of the sensitive attribute for the left entities
        rtable_sensitive_attr (String): the column name of the sensitive attribute for the right entities
        ltable_id (String): ID of the left entity set in both of candidate_set and golden_set tables (should be identical in both tables)
        rtable_id (String): ID of the right entity set in both of candidate_set and golden_set tables (should be identical in both tables)
        ordered (Boolean): sort the groups in the output in an descending order of the cardinalities
    Return:
        cardinality (Dictionary): a dictionary in which the group name is the key and the cardinality is the value for each group
    """
    positive_candidate_set = pd.merge(
        candidate_set, golden_set, on=[ltable_id, rtable_id]
    )
    negative_candidate_set = pd.concat(
        [
            candidate_set[[ltable_id, rtable_id]],
            positive_candidate_set[[ltable_id, rtable_id]],
        ]
    ).drop_duplicates(keep=False)

    return candidate_set_cardinality(
        negative_candidate_set,
        ltable_sensitive_attr,
        rtable_sensitive_attr,
        ltable_id,
        rtable_id,
    )


def create_golden_set(folder_root):
    """Merge train/test/valid sets that include the ground-truth values to use it for recall evaluation and create a csv file from the positive instances.
    Args:
        folder_root (String): path to the dataset folder where {train,test,valid}.csv are located.
    Return:
        None
    """
    df1 = pd.read_csv(folder_root + "/train.csv")
    df2 = pd.read_csv(folder_root + "/valid.csv")
    df3 = pd.read_csv(folder_root + "/test.csv")
    df1 = df1[df1["label"] == 1]
    df2 = df2[df2["label"] == 1]
    df3 = df3[df3["label"] == 1]
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df[["ltable_id", "rtable_id"]].to_csv(
        folder_root + "/matches.csv", header=True, index=False
    )


def load_csv(path):
    """load a .csv file into a DataFrame
    Args:
        path (String): path to the .csv file
    Return:
        table (DataFrame):
    """
    table = pd.read_csv(path)
    return table


def save_csv(path, table):
    """load a DataFrame into a .csv fike
    Args:
        path (String): path to the location to save the .csv file
        table (DataFrame): the table to be saved on the .csv file
    Return:
        None
    """
    table.to_csv(path, header=True, index=False)


def load_pickle(path):
    """load a .pkl file into an Array
    Args:
        path (String): path to the .pkl file
    Return:
        embeddings (Array): unpickled representation of the object
    """
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def save_pickle(path, embeddings):
    """save an object into a .pkl file
    Args:
        path (String): path to the location to save the .pkl file
        embeddings (Array): the array to be pickled
    Return:
        None
    """
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)


def get_sensitive_attr(table, id, sensitive_attr):
    """collects the sensitive attribute column from a DataFrame
    Args:
        table (DataFrame): input table
        sensitive_attr (String): the column name of the sensitive attribute in the table
    Return:
        sensitive_attr_vals (DataFrame): DataFrame of sensitive attribute values of a table
    """
    sensitive_attr_vals = table[[id, sensitive_attr]]
    return sensitive_attr_vals


def save_sensitive_attr(dataset, id, sensitive_attr):
    tableA = load_csv("datasets/" + dataset + "/tableA.csv")
    tableB = load_csv("datasets/" + dataset + "/tableB.csv")
    tableA_sensitive_attr = get_sensitive_attr(tableA, id, sensitive_attr)
    tableB_sensitive_attr = get_sensitive_attr(tableB, id, sensitive_attr)
    save_csv(
        "datasets/" + dataset + "/tableA_sensitive_attr.csv", tableA_sensitive_attr
    )
    save_csv(
        "datasets/" + dataset + "/tableB_sensitive_attr.csv", tableB_sensitive_attr
    )


def remove_low_cardinality_groups(distributions, threshold):
    """removing low cardinality groups from the dictionary
    Args:
        distributions (Dictionary): a dictionary in which the group name is the key and the cardinality is the value for each group
        threhold (Integer): removes the groups with frequency less than specified threshold
    Return:
        distributions (Dictionary): dictionary trimmed with values larger than a specific threshold
    """
    distributions = {k: v for k, v in distributions.items() if v >= threshold}
    return distributions


def get_group_distributions(dataset, sensitive_attr, type, map=None):
    """returns the distribution of groups in the cartesian product of the two tables
    Return:
        distributions (Dictionary): dictionary with groups as keys and cardinalities as the values
    """
    if Path("datasets/" + dataset + "/distribution_" + type + ".json").is_file():
        with open("datasets/" + dataset + "/distribution_" + type + ".json") as file:
            distributions = json.load(file)
    else:
        tableA = load_csv("datasets/" + dataset + "/tableA.csv")
        tableB = load_csv("datasets/" + dataset + "/tableB.csv")
        merged_table = pd.merge(tableA, tableB, how="cross")[
            ["id_x", sensitive_attr + "_x", "id_y", sensitive_attr + "_y"]
        ]
        merged_table = merged_table.rename(
            columns={
                "id_x": "ltable_id",
                sensitive_attr + "_x": "ltable_" + sensitive_attr,
                "id_y": "rtable_id",
                sensitive_attr + "_y": "rtable_" + sensitive_attr,
            }
        )
        ltable_sensitive_attr = load_csv(
            "datasets/" + dataset + "/tableA_sensitive_attr.csv"
        )
        rtable_sensitive_attr = load_csv(
            "datasets/" + dataset + "/tableB_sensitive_attr.csv"
        )
        ltable_id = "ltable_id"
        rtable_id = "rtable_id"
        distributions = candidate_set_cardinality(
            merged_table,
            ltable_sensitive_attr,
            rtable_sensitive_attr,
            ltable_id,
            rtable_id,
            threshold=0,
            type=type,
            map=map,
        )

        with open(
            "datasets/" + dataset + "/distribution_" + type + ".json", "w"
        ) as file:
            file.write(json.dumps(distributions))
    return distributions


def get_pair_id(dataset, idx):
    tableA = load_csv("datasets/" + dataset + "/tableA.csv")
    tableB = load_csv("datasets/" + dataset + "/tableB.csv")
    n_A = tableA.shape[0]
    n_B = tableB.shape[0]
    ltable_id = idx // n_A
    rtable_id = idx % n_B
    return ltable_id, rtable_id


def get_sensitive_attr_from_id(
    tableA_sensitive_attr, tableB_sensitive_attr, ltable_id, rtable_id
):
    return (
        tableA_sensitive_attr[
            tableA_sensitive_attr[tableA_sensitive_attr.columns[0]] == ltable_id
        ][tableA_sensitive_attr.columns[1]].values[0],
        tableB_sensitive_attr[
            tableB_sensitive_attr[tableB_sensitive_attr.columns[0]] == rtable_id
        ][tableB_sensitive_attr.columns[1]].values[0],
    )


def get_sorted_tuples_Autoencoder(dataset, ltable_embedding, rtable_embedding):
    if Path("datasets/" + dataset + "/sorted_tuples_Autoencoder.pkl").is_file():
        output_rows = load_pickle(
            "datasets/" + dataset + "/sorted_tuples_Autoencoder.pkl"
        )
    else:
        output_rows = []
        for idx1 in range(len(ltable_embedding)):
            for idx2 in range(len(rtable_embedding)):
                cosine_sim = 1 - distance.cosine(
                    ltable_embedding[idx1], rtable_embedding[idx2]
                )
                output_rows.append([idx1, idx2, cosine_sim])
        output_rows.sort(key=lambda x: x[2], reverse=True)
        save_pickle(
            "datasets/" + dataset + "/sorted_tuples_Autoencoder.pkl", output_rows
        )
    return output_rows


def get_sorted_tuples_Sudowoodo(dataset, ltable_embedding, rtable_embedding):
    if Path("datasets/" + dataset + "/sorted_tuples_Sudowoodo.pkl").is_file():
        output_rows = load_pickle(
            "datasets/" + dataset + "/sorted_tuples_Sudowoodo.pkl"
        )
    else:
        output_rows = []
        for idx1 in range(len(ltable_embedding)):
            for idx2 in range(len(rtable_embedding)):
                cosine_sim = 1 - distance.cosine(
                    ltable_embedding[idx1], rtable_embedding[idx2]
                )
                output_rows.append([idx1, idx2, cosine_sim])
        output_rows.sort(key=lambda x: x[2], reverse=True)
        save_pickle("datasets/" + dataset +
                    "/sorted_tuples_Sudowoodo.pkl", output_rows)
    return output_rows


def get_sorted_tuples_OverlapBlocker(
    dataset,
    key_attr,
    filter_attr,
    overlap_size,
    out_attrs=None,
    comp_op=">=",
):
    if Path("datasets/" + dataset + "/sorted_tuples_OverlapBlocker.pkl").is_file():
        output_rows = load_pickle(
            "datasets/" + dataset + "/sorted_tuples_OverlapBlocker.pkl"
        )
    else:
        print("Sorting started...")
        ltable = load_pickle("datasets/" + dataset +
                             "/tableA_OverlapBlocker.pkl")
        rtable = load_pickle("datasets/" + dataset +
                             "/tableB_OverlapBlocker.pkl")
        l_columns = get_attrs_to_project(out_attrs, key_attr, filter_attr)
        r_columns = get_attrs_to_project(out_attrs, key_attr, filter_attr)
        l_key_attr_index = l_columns.index(key_attr)
        l_filter_attr_index = l_columns.index(filter_attr)
        tokenizer = WhitespaceTokenizer(return_set=True)
        r_key_attr_index = r_columns.index(key_attr)
        r_filter_attr_index = r_columns.index(filter_attr)
        inverted_index = InvertedIndex(ltable, l_filter_attr_index, tokenizer)
        inverted_index.build(False)
        comp_fn = COMP_OP_MAP[comp_op]
        output_rows = []
        for overlap_size_ in range(overlap_size, 0, -1):
            for r_row in rtable:
                r_string = r_row[r_filter_attr_index]
                r_filter_attr_tokens = tokenizer.tokenize(r_string)
                candidate_overlap = find_candidates(
                    r_filter_attr_tokens, inverted_index
                )
                for cand, overlap in iteritems(candidate_overlap):
                    output_row = (
                        ltable[cand][l_key_attr_index],
                        r_row[r_key_attr_index],
                    )
                    if comp_fn(overlap, overlap_size_):
                        output_rows.append(output_row)

        print("Adding less similar tuples...")
        for r_row in rtable:
            for l_row in ltable:
                output_row = (
                    l_row[l_key_attr_index],
                    r_row[r_key_attr_index],
                )
                output_rows.append(output_row)

        print("Removing duplicates...")
        output_rows = list(dict.fromkeys(output_rows))
        output_rows = [list(elem) for elem in output_rows]
        save_pickle(
            "datasets/" + dataset + "/sorted_tuples_OverlapBlocker.pkl", output_rows
        )
    return output_rows


def get_sorted_tuples_OverlapBlocker_Advanced(
    dataset,
    key_attr,
    filter_attr,
    max_overlap_size,
    out_attrs=None,
    comp_op=">=",
):
    if Path(
        "datasets/" + dataset + "/sorted_tuples_OverlapBlocker_Advanced.pkl"
    ).is_file():
        output_rows = load_pickle(
            "datasets/" + dataset + "/sorted_tuples_OverlapBlocker_Advanced.pkl"
        )
    else:
        print("Sorting started...")
        ltable = load_pickle("datasets/" + dataset +
                             "/tableA_OverlapBlocker.pkl")
        rtable = load_pickle("datasets/" + dataset +
                             "/tableB_OverlapBlocker.pkl")
        l_columns = get_attrs_to_project(out_attrs, key_attr, filter_attr)
        r_columns = get_attrs_to_project(out_attrs, key_attr, filter_attr)
        l_key_attr_index = l_columns.index(key_attr)
        l_filter_attr_index = l_columns.index(filter_attr)
        tokenizer = WhitespaceTokenizer(return_set=True)
        r_key_attr_index = r_columns.index(key_attr)
        r_filter_attr_index = r_columns.index(filter_attr)
        inverted_index = InvertedIndex(ltable, l_filter_attr_index, tokenizer)
        inverted_index.build(False)
        comp_fn = COMP_OP_MAP[comp_op]
        output_rows = {}
        for overlap_size in range(max_overlap_size, 0, -1):
            for r_row in rtable:
                r_string = r_row[r_filter_attr_index]
                r_filter_attr_tokens = tokenizer.tokenize(r_string)
                candidate_overlap = find_candidates(
                    r_filter_attr_tokens, inverted_index
                )
                for cand, overlap in iteritems(candidate_overlap):
                    output_row = (
                        ltable[cand][l_key_attr_index],
                        r_row[r_key_attr_index],
                    )
                    if comp_fn(overlap, overlap_size):
                        if output_row not in output_rows.keys():
                            output_rows[output_row] = overlap_size

        print("Adding less similar tuples...")
        for r_row in rtable:
            for l_row in ltable:
                output_row = (
                    l_row[l_key_attr_index],
                    r_row[r_key_attr_index],
                )
                if output_row not in output_rows.keys():
                    output_rows[output_row] = 0

        print("Finalizing...")
        output_rows = [
            [elem[0], elem[1], output_rows[elem]]
            for elem in sorted(output_rows, key=output_rows.get, reverse=True)
        ]
        save_pickle(
            "datasets/" + dataset + "/sorted_tuples_OverlapBlocker_Advanced.pkl",
            output_rows,
        )
    return output_rows


def measure_disparity(sp, filter=None):
    sp = copy.deepcopy(sp)
    if filter is not None:
        for key in list(sp.keys()):
            if key not in filter:
                sp.pop(key, None)

    for val in sorted(sp, key=sp.get):
        if sp[val] != 0:
            key_min = val
            break
    key_max = max(sp, key=sp.get)
    return sp[key_max] / sp[key_min] - 1


def measure_disparity_advanced(sp, filter):
    for val in sorted(sp, key=sp.get):
        if filter is not None:
            if val in filter:
                key_min = val
                break
        else:
            key_min = val
            break
        
    for val in sorted(sp, key=sp.get, reverse=True):
        if filter is not None:
            if val in filter:
                key_max = val
                break
        else:
            key_max = val
            break

    if sp[key_min] == 0:
        return math.inf
    else:
        return sp[key_max] / sp[key_min] - 1


def write_results_to_file(dataset, k, threshold, recall, disparity, duration, blocker, algorithm):
    print(
        "Recall and Cardinality",
        dataset,
        ":",
        recall,
        k,
    )
    print("Disparity:", disparity)
    print("Duration:", duration)
    print()
    f = open("datasets/" + dataset + "/" + dataset + "_results.txt", "a+")
    f.write(blocker + " , " + algorithm + ", " +
            dataset + " , " + str(k) + " , " + str(threshold))
    f.write("\n")
    f.write(str(round(disparity, 3)) + "  " +
            str(round(duration, 3)) + "  " + str(round(recall, 3)))
    f.write("\n\n")
    f.close()


def measure_disparity_(result, distributions, filter=None):
    sp = defaultdict(int)
    for element in result:
        if filter is not None:
            if element[1] in filter:
                sp[element[1]] += 1 / distributions[element[1]]
        else:
            print(element)
            sp[element[1]] += 1 / distributions[element[1]]
        if filter is not None:
            if element[3] in filter:
                sp[element[3]] += 1 / distributions[element[3]]
        else:
            sp[element[3]] += 1 / distributions[element[3]]
    key_min = min(sp, key=sp.get)
    key_max = max(sp, key=sp.get)
    if sp[key_min] != 0:
        return (sp[key_max] / sp[key_min]) - 1
    else:
        return math.inf


def is_pruned_count(current, distributions, k, tau, filter=None):
    sp = defaultdict(int)
    for element in current:
        if filter is not None:
            if element[1] in filter:
                sp[element[1]] += 1 / distributions[element[1]]
        else:
            sp[element[1]] += 1 / distributions[element[1]]
        if filter is not None:
            if element[3] in filter:
                sp[element[3]] += 1 / distributions[element[3]]
        else:
            sp[element[3]] += 1 / distributions[element[3]]
    if sp:
        min_val = min(sp.values())
        max_val = max(sp.values())
        key_min_list = [k for k, v in sp.items() if v == min_val]
        key_max_list = [k for k, v in sp.items() if v == max_val]
        for key_max in key_max_list:
            for key_min in key_min_list:
                max_val = max(
                    (sp[key_max]),
                    sp[key_min] + ((k - len(current)) * 2) /
                    distributions[key_min],
                )
                min_val = min(
                    (sp[key_max]),
                    sp[key_min] + ((k - len(current)) * 2) /
                    distributions[key_min],
                )
                if min_val != 0 and max_val == sp[key_max]:
                    disparity = (max_val / min_val) - 1
                    if disparity > tau:
                        return True
    return False


def is_pruned_sum(current, top_k_results, sorted_tuples, i, k):
    if top_k_results is not None:
        if sum([item[2] for item in sorted_tuples[i + 1: i + k - len(current)]]) < sum(
            [item[4] for item in top_k_results[len(current) + 1:]]
        ):
            return True
    return False
