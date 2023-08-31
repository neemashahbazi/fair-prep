from utils import (
    get_sensitive_attr_from_id,
    get_group_distributions,
    load_csv,
    measure_disparity,
    measure_disparity_,
    measure_disparity_advanced,
    save_pickle,
    load_pickle,
    is_pruned_count,
    is_pruned_sum,
)
import timeit
import pandas as pd
from collections import defaultdict
from pathlib import Path
import copy
import numpy as np
import math
import heapq
from numpy import nan


def baseline_no_fairness(
    sorted_tuples, dataset, k, sensitive_attr, map=None, filter=None
):
    distributions = get_group_distributions(
        dataset, sensitive_attr, type="single", map=map
    )
    tableA_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableA_sensitive_attr.csv"
    )
    tableB_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableB_sensitive_attr.csv"
    )
    top_k_results = []
    sp = defaultdict(int)
    start = timeit.default_timer()
    idx = 0
    while len(top_k_results) < k:
        ltable_id = sorted_tuples[idx][0]
        rtable_id = sorted_tuples[idx][1]
        idx += 1
        (
            ltable_sensitive_attr_val,
            rtable_sensitive_attr_val,
        ) = get_sensitive_attr_from_id(
            tableA_sensitive_attr, tableB_sensitive_attr, ltable_id, rtable_id
        )

        if map is not None:
            if ltable_sensitive_attr_val in map.keys():
                ltable_sensitive_attr_val = map[ltable_sensitive_attr_val]
            if rtable_sensitive_attr_val in map.keys():
                rtable_sensitive_attr_val = map[rtable_sensitive_attr_val]

        top_k_results.append([ltable_id, rtable_id])

        if not pd.isna(ltable_sensitive_attr_val):
            sp[ltable_sensitive_attr_val] += (
                1 / distributions[ltable_sensitive_attr_val]
            )
        if not pd.isna(rtable_sensitive_attr_val):
            sp[rtable_sensitive_attr_val] += (
                1 / distributions[rtable_sensitive_attr_val]
            )

    disparity = measure_disparity(sp, filter)
    candidate_sets = pd.DataFrame(
        data={
            "ltable_id": [x[0] for x in top_k_results],
            "rtable_id": [x[1] for x in top_k_results],
        }
    )
    stop = timeit.default_timer()
    return candidate_sets, disparity, stop - start


def baseline_greedy(
    sorted_tuples, dataset, k, sensitive_attr, threshold, map=None, filter=None
):
    distributions = get_group_distributions(
        dataset, sensitive_attr, type="single", map=map
    )
    tableA_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableA_sensitive_attr.csv"
    )
    tableB_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableB_sensitive_attr.csv"
    )
    top_k_results = []
    sp = defaultdict(int)
    start = timeit.default_timer()
    idx = 0
    print("Filling the top-K list...")
    while len(top_k_results) < k:
        ltable_id = sorted_tuples[idx][0]
        rtable_id = sorted_tuples[idx][1]
        idx += 1
        (
            ltable_sensitive_attr_val,
            rtable_sensitive_attr_val,
        ) = get_sensitive_attr_from_id(
            tableA_sensitive_attr, tableB_sensitive_attr, ltable_id, rtable_id
        )
        if map is not None:
            if ltable_sensitive_attr_val in map.keys():
                ltable_sensitive_attr_val = map[ltable_sensitive_attr_val]
            if rtable_sensitive_attr_val in map.keys():
                rtable_sensitive_attr_val = map[rtable_sensitive_attr_val]
        top_k_results.append(
            [ltable_id, ltable_sensitive_attr_val,
                rtable_id, rtable_sensitive_attr_val]
        )
        if not pd.isna(ltable_sensitive_attr_val):
            sp[ltable_sensitive_attr_val] += (
                1 / distributions[ltable_sensitive_attr_val]
            )
        if not pd.isna(rtable_sensitive_attr_val):
            sp[rtable_sensitive_attr_val] += (
                1 / distributions[rtable_sensitive_attr_val]
            )

    disparity = measure_disparity(sp, filter)
    if disparity < threshold:
        candidate_sets = pd.DataFrame(
            data={
                "ltable_id": [x[0] for x in top_k_results],
                "rtable_id": [x[2] for x in top_k_results],
            }
        )
        stop = timeit.default_timer()
        return candidate_sets, disparity, stop - start

    print("Greedy Fixing fairness issues...")
    for idx1 in range(k, len(sorted_tuples)):
        for val in sorted(sp, key=sp.get, reverse=True):
            if filter is not None:
                if val in filter:
                    key_max = val
                    break
            else:
                key_max = val
                break
        for idx2 in range(len(top_k_results) - 1, -1, -1):
            if top_k_results[idx2][1] == key_max or top_k_results[idx2][3] == key_max:
                new_ltable_id = sorted_tuples[idx1][0]
                new_rtable_id = sorted_tuples[idx1][1]
                (
                    new_ltable_sensitive_attr_val,
                    new_rtable_sensitive_attr_val,
                ) = get_sensitive_attr_from_id(
                    tableA_sensitive_attr,
                    tableB_sensitive_attr,
                    new_ltable_id,
                    new_rtable_id,
                )
                if map is not None:
                    if new_ltable_sensitive_attr_val in map.keys():
                        new_ltable_sensitive_attr_val = map[
                            new_ltable_sensitive_attr_val
                        ]
                    if new_rtable_sensitive_attr_val in map.keys():
                        new_rtable_sensitive_attr_val = map[
                            new_rtable_sensitive_attr_val
                        ]

                old_ltable_id = top_k_results[idx2][0]
                old_rtable_id = top_k_results[idx2][2]
                (
                    old_ltable_sensitive_attr_val,
                    old_rtable_sensitive_attr_val,
                ) = get_sensitive_attr_from_id(
                    tableA_sensitive_attr,
                    tableB_sensitive_attr,
                    old_ltable_id,
                    old_rtable_id,
                )
                if map is not None:
                    if old_ltable_sensitive_attr_val in map.keys():
                        old_ltable_sensitive_attr_val = map[
                            old_ltable_sensitive_attr_val
                        ]
                    if old_rtable_sensitive_attr_val in map.keys():
                        old_rtable_sensitive_attr_val = map[
                            old_rtable_sensitive_attr_val
                        ]

                if not pd.isna(old_ltable_sensitive_attr_val):
                    sp[old_ltable_sensitive_attr_val] -= (
                        1 / distributions[old_ltable_sensitive_attr_val]
                    )
                if not pd.isna(old_rtable_sensitive_attr_val):
                    sp[old_rtable_sensitive_attr_val] -= (
                        1 / distributions[old_rtable_sensitive_attr_val]
                    )
                if not pd.isna(new_ltable_sensitive_attr_val):
                    sp[new_ltable_sensitive_attr_val] += (
                        1 / distributions[new_ltable_sensitive_attr_val]
                    )
                if not pd.isna(new_rtable_sensitive_attr_val):
                    sp[new_rtable_sensitive_attr_val] += (
                        1 / distributions[new_rtable_sensitive_attr_val]
                    )
                disparity_new = measure_disparity(sp, filter)
                if disparity_new < disparity:
                    top_k_results[idx2] = [
                        new_ltable_id,
                        new_ltable_sensitive_attr_val,
                        new_rtable_id,
                        new_rtable_sensitive_attr_val,
                    ]
                    disparity = disparity_new
                    print(disparity)
                    if disparity < threshold:
                        candidate_sets = pd.DataFrame(
                            data={
                                "ltable_id": [x[0] for x in top_k_results],
                                "rtable_id": [x[2] for x in top_k_results],
                            }
                        )
                        stop = timeit.default_timer()
                        return candidate_sets, disparity, stop - start
                    else:
                        break
                else:
                    if not pd.isna(old_ltable_sensitive_attr_val):
                        sp[old_ltable_sensitive_attr_val] += (
                            1 / distributions[old_ltable_sensitive_attr_val]
                        )
                    if not pd.isna(old_rtable_sensitive_attr_val):
                        sp[old_rtable_sensitive_attr_val] += (
                            1 / distributions[old_rtable_sensitive_attr_val]
                        )
                    if not pd.isna(new_ltable_sensitive_attr_val):
                        sp[new_ltable_sensitive_attr_val] -= (
                            1 / distributions[new_ltable_sensitive_attr_val]
                        )
                    if not pd.isna(new_rtable_sensitive_attr_val):
                        sp[new_rtable_sensitive_attr_val] -= (
                            1 / distributions[new_rtable_sensitive_attr_val]
                        )
    stop = timeit.default_timer()
    return None, disparity, stop - start


def baseline_stratified(
    sorted_tuples, dataset, k, sensitive_attr, threshold, map=None, filter=None
):
    distributions = get_group_distributions(
        dataset, sensitive_attr, type="single", map=map
    )
    tableA_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableA_sensitive_attr.csv"
    )
    tableB_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableB_sensitive_attr.csv"
    )
    top_k_results = []
    dict_of_queues = defaultdict(list)
    sp = defaultdict(int)
    start = timeit.default_timer()
    print("Starting bucketization...")
    for idx in range(len(sorted_tuples[:1000000])):
        ltable_id = sorted_tuples[idx][0]
        rtable_id = sorted_tuples[idx][1]
        (
            ltable_sensitive_attr_val,
            rtable_sensitive_attr_val,
        ) = get_sensitive_attr_from_id(
            tableA_sensitive_attr, tableB_sensitive_attr, ltable_id, rtable_id
        )
        if map is not None:
            if ltable_sensitive_attr_val in map.keys():
                ltable_sensitive_attr_val = map[ltable_sensitive_attr_val]
            if rtable_sensitive_attr_val in map.keys():
                rtable_sensitive_attr_val = map[rtable_sensitive_attr_val]

        dict_of_queues[ltable_sensitive_attr_val].append(
            [
                ltable_id,
                ltable_sensitive_attr_val,
                rtable_id,
                rtable_sensitive_attr_val,
            ]
        )

        dict_of_queues[rtable_sensitive_attr_val].append(
            [
                ltable_id,
                ltable_sensitive_attr_val,
                rtable_id,
                rtable_sensitive_attr_val,
            ]
        )
        if not pd.isna(ltable_sensitive_attr_val):
            sp[ltable_sensitive_attr_val] = 0
        if not pd.isna(ltable_sensitive_attr_val):
            sp[rtable_sensitive_attr_val] = 0

    print("Stratified Fixing fairness issues...")
    while True:
        if len(top_k_results) < k:
            if dataset in ["Walmart-Amazon", "DBLP-Scholar"]:
                for val in sorted(sp, key=sp.get):
                    if len(dict_of_queues[val]) != 0:
                        if filter is not None:
                            if val in filter:
                                key_min = val
                                break
                        else:
                            key_min = val
                            break
            else:
                for val in sorted(sp, key=sp.get):
                    if len(dict_of_queues[val]) != 0:
                        key_min = val
                        break
            tuple = dict_of_queues[key_min].pop(0)
            ltable_sensitive_attr_val = tuple[1]
            rtable_sensitive_attr_val = tuple[3]
            if tuple not in top_k_results:
                top_k_results.append(tuple)
                if not pd.isna(ltable_sensitive_attr_val):
                    sp[ltable_sensitive_attr_val] += (
                        1 / distributions[ltable_sensitive_attr_val]
                    )
                if not pd.isna(rtable_sensitive_attr_val):
                    sp[rtable_sensitive_attr_val] += (
                        1 / distributions[rtable_sensitive_attr_val]
                    )
        else:
            disparity = measure_disparity(sp, filter)
            if disparity < threshold:
                candidate_sets = pd.DataFrame(
                    data={
                        "ltable_id": [x[0] for x in top_k_results],
                        "rtable_id": [x[2] for x in top_k_results],
                    }
                )
                stop = timeit.default_timer()

                return (
                    candidate_sets,
                    disparity,
                    stop - start,
                )
            else:
                for val in sorted(sp, key=sp.get, reverse=True):
                    if filter is not None:
                        if val in filter:
                            key_max = val
                            break
                    else:
                        key_max = val
                        break

                if all(
                    len(val) == 0
                    for key, val in dict_of_queues.items()
                    if key != key_max
                ):
                    candidate_sets = pd.DataFrame(
                        data={
                            "ltable_id": [x[0] for x in top_k_results],
                            "rtable_id": [x[2] for x in top_k_results],
                        }
                    )
                    stop = timeit.default_timer()
                    return None, disparity, stop - start
                else:
                    for idx in range(len(top_k_results) - 1, -1, -1):
                        for val in sorted(sp, key=sp.get):
                            if len(dict_of_queues[val]) != 0:
                                if filter is not None:
                                    if val in filter:
                                        key_min = val
                                        break
                                else:
                                    key_min = val
                                    break
                        if (
                            top_k_results[idx][1] == key_max
                            or top_k_results[idx][3] == key_max
                        ):
                            new_tuple = dict_of_queues[key_min].pop(0)
                            if new_tuple not in top_k_results:
                                new_ltable_sensitive_attr_val = new_tuple[1]
                                new_rtable_sensitive_attr_val = new_tuple[3]
                                old_ltable_sensitive_attr_val = top_k_results[idx][1]
                                old_rtable_sensitive_attr_val = top_k_results[idx][3]

                                if not pd.isna(old_ltable_sensitive_attr_val):
                                    sp[old_ltable_sensitive_attr_val] -= (
                                        1 /
                                        distributions[old_ltable_sensitive_attr_val]
                                    )
                                if not pd.isna(old_rtable_sensitive_attr_val):
                                    sp[old_rtable_sensitive_attr_val] -= (
                                        1 /
                                        distributions[old_rtable_sensitive_attr_val]
                                    )
                                if not pd.isna(new_ltable_sensitive_attr_val):
                                    sp[new_ltable_sensitive_attr_val] += (
                                        1 /
                                        distributions[new_ltable_sensitive_attr_val]
                                    )
                                if not pd.isna(new_rtable_sensitive_attr_val):
                                    sp[new_rtable_sensitive_attr_val] += (
                                        1 /
                                        distributions[new_rtable_sensitive_attr_val]
                                    )
                                disparity_new = measure_disparity(sp, filter)
                                if disparity_new < disparity:
                                    top_k_results[idx] = new_tuple
                                    print(disparity_new)
                                    disparity = disparity_new
                                    if disparity < threshold:
                                        candidate_sets = pd.DataFrame(
                                            data={
                                                "ltable_id": [
                                                    x[0] for x in top_k_results
                                                ],
                                                "rtable_id": [
                                                    x[2] for x in top_k_results
                                                ],
                                            }
                                        )
                                        stop = timeit.default_timer()
                                        return (
                                            candidate_sets,
                                            disparity,
                                            stop - start,
                                        )
                                    else:
                                        break
                                else:
                                    if not pd.isna(old_ltable_sensitive_attr_val):
                                        sp[old_ltable_sensitive_attr_val] += (
                                            1
                                            / distributions[
                                                old_ltable_sensitive_attr_val
                                            ]
                                        )
                                    if not pd.isna(old_rtable_sensitive_attr_val):
                                        sp[old_rtable_sensitive_attr_val] += (
                                            1
                                            / distributions[
                                                old_rtable_sensitive_attr_val
                                            ]
                                        )
                                    if not pd.isna(new_ltable_sensitive_attr_val):
                                        sp[new_ltable_sensitive_attr_val] -= (
                                            1
                                            / distributions[
                                                new_ltable_sensitive_attr_val
                                            ]
                                        )
                                    if not pd.isna(new_rtable_sensitive_attr_val):
                                        sp[new_rtable_sensitive_attr_val] -= (
                                            1
                                            / distributions[
                                                new_rtable_sensitive_attr_val
                                            ]
                                        )


def baseline_greedy_OverlapBlocker(
    sorted_tuples, dataset, k, sensitive_attr, threshold, map=None, filter=None
):
    distributions = get_group_distributions(
        dataset, sensitive_attr, type="single", map=map
    )
    tableA_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableA_sensitive_attr.csv"
    )
    tableB_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableB_sensitive_attr.csv"
    )
    top_k_results = []
    sp = defaultdict(int)
    start = timeit.default_timer()
    idx = 0
    print("Filling the top-K list...")
    while len(top_k_results) < k:
        ltable_id = sorted_tuples[idx][0]
        rtable_id = sorted_tuples[idx][1]
        idx += 1
        (
            ltable_sensitive_attr_val,
            rtable_sensitive_attr_val,
        ) = get_sensitive_attr_from_id(
            tableA_sensitive_attr, tableB_sensitive_attr, ltable_id, rtable_id
        )
        if map is not None:
            if ltable_sensitive_attr_val in map.keys():
                ltable_sensitive_attr_val = map[ltable_sensitive_attr_val]
            if rtable_sensitive_attr_val in map.keys():
                rtable_sensitive_attr_val = map[rtable_sensitive_attr_val]
        top_k_results.append(
            [ltable_id, ltable_sensitive_attr_val,
                rtable_id, rtable_sensitive_attr_val]
        )
        if not pd.isna(ltable_sensitive_attr_val):
            sp[ltable_sensitive_attr_val] += (
                1 / distributions[ltable_sensitive_attr_val]
            )
        if not pd.isna(rtable_sensitive_attr_val):
            sp[rtable_sensitive_attr_val] += (
                1 / distributions[rtable_sensitive_attr_val]
            )

    disparity = measure_disparity(sp, filter)
    if disparity < threshold:
        candidate_sets = pd.DataFrame(
            data={
                "ltable_id": [x[0] for x in top_k_results],
                "rtable_id": [x[2] for x in top_k_results],
            }
        )
        stop = timeit.default_timer()
        return candidate_sets, disparity, stop - start

    if Path("datasets/" + dataset + "/dict_of_overlaps.pkl").is_file():
        dict_of_overlaps = load_pickle(
            "datasets/" + dataset + "/dict_of_overlaps.pkl")
    else:
        dict_of_overlaps = defaultdict(lambda: defaultdict(list))
        for idx in range(k, len(sorted_tuples[:2000000])):
            ltable_id = sorted_tuples[idx][0]
            rtable_id = sorted_tuples[idx][1]
            overlap_size = int(sorted_tuples[idx][2])
            (
                ltable_sensitive_attr_val,
                rtable_sensitive_attr_val,
            ) = get_sensitive_attr_from_id(
                tableA_sensitive_attr, tableB_sensitive_attr, ltable_id, rtable_id
            )
            if map is not None:
                if ltable_sensitive_attr_val in map.keys():
                    ltable_sensitive_attr_val = map[ltable_sensitive_attr_val]
                if rtable_sensitive_attr_val in map.keys():
                    rtable_sensitive_attr_val = map[rtable_sensitive_attr_val]

            dict_of_overlaps[overlap_size][ltable_sensitive_attr_val].append(
                [
                    ltable_id,
                    ltable_sensitive_attr_val,
                    rtable_id,
                    rtable_sensitive_attr_val,
                ]
            )
            dict_of_overlaps[overlap_size][rtable_sensitive_attr_val].append(
                [
                    ltable_id,
                    ltable_sensitive_attr_val,
                    rtable_id,
                    rtable_sensitive_attr_val,
                ]
            )
        save_pickle("datasets/" + dataset +
                    "/dict_of_overlaps.pkl", dict_of_overlaps)
    flag = True
    for overlap_key in dict(sorted(dict_of_overlaps.items(), reverse=True)):
        print(overlap_key)
        while True:
            for val in sorted(sp, key=sp.get, reverse=True):
                if filter is not None:
                    if val in filter:
                        key_max = val
                        break
                else:
                    key_max = val
                    break
            if not flag:
                flag = True
                break
            else:
                for idx in range(len(top_k_results) - 1, -1, -1):
                    if (
                        top_k_results[idx][1] == key_max
                        or top_k_results[idx][3] == key_max
                    ):
                        for val in sorted(sp, key=sp.get):
                            if len(dict_of_overlaps[overlap_key][val]) != 0:
                                if filter is not None:
                                    if val in filter:
                                        key_min = val
                                        break
                                else:
                                    key_min = val
                                    break

                        if key_min == key_max:
                            flag = False
                            break

                        new_tuple = dict_of_overlaps[overlap_key][key_min].pop(
                            0)
                        if new_tuple not in top_k_results:
                            new_ltable_sensitive_attr_val = new_tuple[1]
                            new_rtable_sensitive_attr_val = new_tuple[3]
                            old_ltable_sensitive_attr_val = top_k_results[idx][1]
                            old_rtable_sensitive_attr_val = top_k_results[idx][3]

                            if not pd.isna(old_ltable_sensitive_attr_val):
                                sp[old_ltable_sensitive_attr_val] -= (
                                    1 /
                                    distributions[old_ltable_sensitive_attr_val]
                                )
                            if not pd.isna(old_rtable_sensitive_attr_val):
                                sp[old_rtable_sensitive_attr_val] -= (
                                    1 /
                                    distributions[old_rtable_sensitive_attr_val]
                                )
                            if not pd.isna(new_ltable_sensitive_attr_val):
                                sp[new_ltable_sensitive_attr_val] += (
                                    1 /
                                    distributions[new_ltable_sensitive_attr_val]
                                )
                            if not pd.isna(new_rtable_sensitive_attr_val):
                                sp[new_rtable_sensitive_attr_val] += (
                                    1 /
                                    distributions[new_rtable_sensitive_attr_val]
                                )
                            disparity_new = measure_disparity(sp, filter)
                            if disparity_new < disparity:
                                top_k_results[idx] = new_tuple
                                print(disparity_new)
                                disparity = disparity_new
                                if disparity < threshold:
                                    candidate_sets = pd.DataFrame(
                                        data={
                                            "ltable_id": [x[0] for x in top_k_results],
                                            "rtable_id": [x[2] for x in top_k_results],
                                        }
                                    )
                                    stop = timeit.default_timer()
                                    return (
                                        candidate_sets,
                                        disparity,
                                        stop - start,
                                    )
                                else:
                                    break
                            else:
                                if not pd.isna(old_ltable_sensitive_attr_val):
                                    sp[old_ltable_sensitive_attr_val] += (
                                        1 /
                                        distributions[old_ltable_sensitive_attr_val]
                                    )
                                if not pd.isna(old_rtable_sensitive_attr_val):
                                    sp[old_rtable_sensitive_attr_val] += (
                                        1 /
                                        distributions[old_rtable_sensitive_attr_val]
                                    )
                                if not pd.isna(new_ltable_sensitive_attr_val):
                                    sp[new_ltable_sensitive_attr_val] -= (
                                        1 /
                                        distributions[new_ltable_sensitive_attr_val]
                                    )
                                if not pd.isna(new_rtable_sensitive_attr_val):
                                    sp[new_rtable_sensitive_attr_val] -= (
                                        1 /
                                        distributions[new_rtable_sensitive_attr_val]
                                    )


def exact_bt(
    sorted_tuples, dataset, k, sensitive_attr, threshold, map=None, filter=None
):
    def backtrack(current, start, current_score):
        nonlocal top_k_score
        nonlocal top_k_results
        if len(current) == k:
            disparity = measure_disparity_(current, distributions, filter)
            if disparity <= threshold:
                if current_score > top_k_score:
                    top_k_score = current_score
                    top_k_results = copy.deepcopy(current)

        else:
            for i in range(start, len(sorted_tuples)):
                ltable_id = sorted_tuples[i][0]
                rtable_id = sorted_tuples[i][1]
                (
                    ltable_sensitive_attr_val,
                    rtable_sensitive_attr_val,
                ) = get_sensitive_attr_from_id(
                    tableA_sensitive_attr,
                    tableB_sensitive_attr,
                    ltable_id,
                    rtable_id,
                )
                score = sorted_tuples[i][2]
                if map is not None:
                    if ltable_sensitive_attr_val in map.keys():
                        ltable_sensitive_attr_val = map[ltable_sensitive_attr_val]
                    if rtable_sensitive_attr_val in map.keys():
                        rtable_sensitive_attr_val = map[rtable_sensitive_attr_val]
                if not is_pruned_count(
                    current, distributions, k, threshold, filter=None
                ) and not is_pruned_sum(current, top_k_results, sorted_tuples, i, k):
                    current.append(
                        [
                            ltable_id,
                            ltable_sensitive_attr_val,
                            rtable_id,
                            rtable_sensitive_attr_val,
                            score,
                        ]
                    )
                    backtrack(current, i + 1, current_score + score)
                    current.pop()

    top_k_score = 0
    top_k_results = None
    distributions = get_group_distributions(
        dataset, sensitive_attr, type="single", map=map
    )
    tableA_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableA_sensitive_attr.csv"
    )
    tableB_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableB_sensitive_attr.csv"
    )
    start = timeit.default_timer()
    backtrack([], 0, 0)
    candidate_sets = pd.DataFrame(
        data={
            "ltable_id": [x[0] for x in top_k_results],
            "rtable_id": [x[2] for x in top_k_results],
        }
    )
    disparity = measure_disparity_(top_k_results, distributions, filter)
    stop = timeit.default_timer()
    return candidate_sets, disparity, stop - start


def advanced_single_pass(
    sorted_tuples, dataset, k, sensitive_attr, threshold, m, map=None, filter=None
):
    top_k_results = []
    sp = defaultdict(int)
    dict_of_queues_pair = defaultdict(list)
    dict_of_queues = defaultdict(list)
    distributions = get_group_distributions(
        dataset, sensitive_attr, type="single", map=map
    )
    tableA_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableA_sensitive_attr.csv"
    )
    tableB_sensitive_attr = load_csv(
        "datasets/" + dataset + "/tableB_sensitive_attr.csv"
    )
    start = timeit.default_timer()
    print("Start bucketization...")
    for idx in range(len(sorted_tuples[:1000000])):
        ltable_id = sorted_tuples[idx][0]
        rtable_id = sorted_tuples[idx][1]
        (
            ltable_sensitive_attr_val,
            rtable_sensitive_attr_val,
        ) = get_sensitive_attr_from_id(
            tableA_sensitive_attr, tableB_sensitive_attr, ltable_id, rtable_id
        )
        if map is not None:
            if ltable_sensitive_attr_val in map.keys():
                ltable_sensitive_attr_val = map[ltable_sensitive_attr_val]
            if rtable_sensitive_attr_val in map.keys():
                rtable_sensitive_attr_val = map[rtable_sensitive_attr_val]

        sim_score = sorted_tuples[idx][2]
        dict_of_queues_pair[str((ltable_sensitive_attr_val, rtable_sensitive_attr_val))].append(
            [
                ltable_id,
                ltable_sensitive_attr_val,
                rtable_id,
                rtable_sensitive_attr_val,
                sim_score,
            ]
        )
        if ltable_sensitive_attr_val not in dict_of_queues:
            dict_of_queues[ltable_sensitive_attr_val] = [
                ltable_id,
                ltable_sensitive_attr_val,
                rtable_id,
                rtable_sensitive_attr_val,
            ]

        if ltable_sensitive_attr_val not in dict_of_queues:
            dict_of_queues[rtable_sensitive_attr_val] = [
                ltable_id,
                ltable_sensitive_attr_val,
                rtable_id,
                rtable_sensitive_attr_val,
            ]

        if not pd.isna(ltable_sensitive_attr_val):
            sp[ltable_sensitive_attr_val] = 0
        if not pd.isna(rtable_sensitive_attr_val):
            sp[rtable_sensitive_attr_val] = 0

    for key in dict_of_queues.keys():
        if filter is not None:
            if key in filter:
                pair = dict_of_queues[key]
                top_k_results.append(pair)
                if not pd.isna(pair[1]):
                    sp[pair[1]] += (
                        1 / distributions[pair[1]]
                    )
                if not pd.isna(pair[3]):
                    sp[pair[3]] += (
                        1 / distributions[pair[3]]
                    )
        else:
            pair = dict_of_queues[key]
            top_k_results.append(pair)
            if not pd.isna(pair[1]):
                sp[pair[1]] += (
                    1 / distributions[pair[1]]
                )
            if not pd.isna(pair[3]):
                sp[pair[3]] += (
                    1 / distributions[pair[3]]
                )

    print("Adding K elements to the results...")
    first_iter = True
    C = []
    while len(top_k_results) < k:
        if first_iter:
            for key in dict_of_queues_pair.keys():
                for _ in range(m):
                    if len(dict_of_queues_pair[key]) != 0:
                        pair = dict_of_queues_pair[key].pop(0)
                        score = compute_score(pair, sp, distributions, filter)
                        heapq.heappush(C, (-1*score, pair))
            first_iter = False
        for _ in range(m):
            pair_max = heapq.heappop(C)[1]
            if pair_max not in top_k_results:
                top_k_results.append(pair_max)
            if not pd.isna(pair_max[1]):
                sp[pair_max[1]] += (
                    1 / distributions[pair_max[1]]
                )
            if not pd.isna(pair_max[3]):
                sp[pair_max[3]] += (
                    1 / distributions[pair_max[3]]
                )
            if len(dict_of_queues_pair[str((pair_max[1], pair_max[3]))]) != 0:
                pair = dict_of_queues_pair[str(
                    (pair_max[1], pair_max[3]))].pop(0)
                score = compute_score(pair, sp, distributions, filter)
                heapq.heappush(C, (-1*score, pair))
            if len(top_k_results) >= k:
                break

    print("Fixing whatever issues left...")
    disparity = measure_disparity_advanced(sp, filter)
    if disparity < threshold:
        stop = timeit.default_timer()
        candidate_sets = pd.DataFrame(
            data={
                "ltable_id": [x[0] for x in top_k_results],
                "rtable_id": [x[2] for x in top_k_results],
            }
        )
        return candidate_sets, disparity, stop - start
    else:
        while True:
            for idx in range(len(top_k_results) - 1, -1, -1):
                for val in sorted(sp, key=sp.get, reverse=True):
                    if filter is not None:
                        if val in filter:
                            key_max = val
                            break
                    else:
                        key_max = val
                        break
                for val in sorted(sp, key=sp.get):
                    if filter is not None:
                        if val in filter:
                            key_min = val
                            break
                    else:
                        key_min = val
                        break
                if (
                    top_k_results[idx][1] == key_max
                    or top_k_results[idx][3] == key_max
                ):
                    new_tuple = None

                    if len(dict_of_queues_pair[str((key_min, key_min))]) != 0:
                        new_tuple = dict_of_queues_pair[str((
                            key_min, key_min))].pop(0)
                    else:
                        for key in dict_of_queues_pair.keys():
                            key = eval(key)
                            if (key[0] == key_min or key[1] == key_min) and (key[0] != key_max or key[1] != key_max):
                                if len(dict_of_queues_pair[str(
                                        key)]) != 0:
                                    new_tuple = dict_of_queues_pair[str(
                                        key)].pop(0)

                    if new_tuple is None:
                        for key in dict_of_queues_pair.keys():
                            key = eval(key)
                            if (key[0] != key_max or key[1] != key_max):
                                if len(dict_of_queues_pair[str(
                                        key)]) != 0:
                                    new_tuple = dict_of_queues_pair[str(
                                        key)].pop(0)

                    new_ltable_sensitive_attr_val = new_tuple[1]
                    new_rtable_sensitive_attr_val = new_tuple[3]
                    old_ltable_sensitive_attr_val = top_k_results[idx][1]
                    old_rtable_sensitive_attr_val = top_k_results[idx][3]

                    if not pd.isna(old_ltable_sensitive_attr_val):
                        sp[old_ltable_sensitive_attr_val] -= (
                            1 /
                            distributions[old_ltable_sensitive_attr_val]
                        )
                    if not pd.isna(old_rtable_sensitive_attr_val):
                        sp[old_rtable_sensitive_attr_val] -= (
                            1 /
                            distributions[old_rtable_sensitive_attr_val]
                        )
                    if not pd.isna(new_ltable_sensitive_attr_val):
                        sp[new_ltable_sensitive_attr_val] += (
                            1 /
                            distributions[new_ltable_sensitive_attr_val]
                        )
                    if not pd.isna(new_rtable_sensitive_attr_val):
                        sp[new_rtable_sensitive_attr_val] += (
                            1 /
                            distributions[new_rtable_sensitive_attr_val]
                        )

                    top_k_results[idx] = new_tuple
                    disparity = measure_disparity_advanced(sp, filter)
                    # print(disparity)
                    if disparity < threshold:
                        candidate_sets = pd.DataFrame(
                            data={
                                "ltable_id": [
                                    x[0] for x in top_k_results
                                ],
                                "rtable_id": [
                                    x[2] for x in top_k_results
                                ],
                            }
                        )
                        stop = timeit.default_timer()
                        return (
                            candidate_sets,
                            disparity,
                            stop - start,
                        )
                    else:
                        break


def compute_score(pair, sp, distributions, filter):
    sim_score = pair[4]
    left_sensitive_attr = pair[1]
    right_sensitive_attr = pair[3]

    if filter is not None and left_sensitive_attr not in filter and right_sensitive_attr not in filter:
        return sim_score
    else:
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
        if left_sensitive_attr != key_max or left_sensitive_attr != key_min or right_sensitive_attr != key_max or right_sensitive_attr != key_min:
            return sim_score
        else:
            disparity_before = measure_disparity_advanced(sp, filter)
            if not pd.isna(left_sensitive_attr):
                sp[left_sensitive_attr] += 1 / \
                    distributions[left_sensitive_attr]
            if not pd.isna(right_sensitive_attr):
                sp[right_sensitive_attr] += 1 / \
                    distributions[right_sensitive_attr]
            disparity_after = measure_disparity_advanced(sp, filter)
            if not pd.isna(left_sensitive_attr):
                sp[left_sensitive_attr] -= 1 / \
                    distributions[left_sensitive_attr]
            if not pd.isna(right_sensitive_attr):
                sp[right_sensitive_attr] -= 1 / \
                    distributions[right_sensitive_attr]
            if disparity_after == math.inf or disparity_before == math.inf:
                return sim_score
            else:
                contribution = disparity_before - disparity_after
                return sim_score + contribution
