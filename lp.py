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
import mosek
import sys
import pickle

def streamprinter(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


def lp_solve(
    sorted_tuples, dataset, k, sensitive_attr, threshold, map=None, filter=None
):   
    top_k_results = []
    sp = defaultdict(int)
    dict_of_queues_pair = defaultdict(list)
    group_to_id = dict()
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
    try:
        dict_of_queues_pair = pickle.load(open('./queues_pair_{}.pkl'.format(dataset), 'rb'))
    except:
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

            # print(ltable_sensitive_attr_val, rtable_sensitive_attr_val)
            # print(sim_score)
            dict_of_queues_pair[str((ltable_sensitive_attr_val, rtable_sensitive_attr_val))].append(
                [
                    ltable_id,
                    ltable_sensitive_attr_val,
                    rtable_id,
                    rtable_sensitive_attr_val,
                    sorted_tuples[idx][2],
                ]
            )
            if ltable_sensitive_attr_val not in group_to_id:
                group_to_id[ltable_sensitive_attr_val] = len(group_to_id)
            if rtable_sensitive_attr_val not in group_to_id:
                group_to_id[rtable_sensitive_attr_val] = len(group_to_id)
        pickle.dump(dict_of_queues_pair, open('./queues_pair_{}.pkl'.format(dataset), 'wb'))
        
    print(len(dict_of_queues_pair), k)
    
    
    pair_keys = list(dict_of_queues_pair.keys())       
    for p in pair_keys:
        for j in range(len(dict_of_queues_pair[p])):
            l_val = dict_of_queues_pair[p][j][1]
            r_val = dict_of_queues_pair[p][j][3]

            if l_val not in group_to_id and not pd.isna(l_val):
                if filter is None or l_val in filter:
                    group_to_id[l_val] = len(group_to_id)
            if r_val not in group_to_id and not pd.isna(r_val):
                if filter is None or r_val in filter:
                    group_to_id[r_val] = len(group_to_id)
     
    upper = int(k * (threshold+1) / len(group_to_id))
    
    var_id = 0
    w_list = [] # sim score of each pair
    var_pair_map = {} # map var_id to pair
    pair_var_list = {} # list of var_ids for each pair type
    group_var_list = {} # list of var_ids for each group
    constraint_list = []
    for p in pair_keys:                                                                                                               
        for j in range(min(len(dict_of_queues_pair[p]), upper)):
            l_val = dict_of_queues_pair[p][j][1]
            r_val = dict_of_queues_pair[p][j][3]
            w_list.append(dict_of_queues_pair[p][j][-1])
            if p not in pair_var_list:
                pair_var_list[p] = []
            if filter is not None:
                if l_val in filter and l_val not in group_var_list:
                    group_var_list[l_val] = []
                if r_val in filter and r_val not in group_var_list:
                    group_var_list[r_val] = []
                if l_val != r_val:
                    if l_val in filter:
                        group_var_list[l_val].append(var_id)
                    if r_val in filter:
                        group_var_list[r_val].append(var_id)
                else:
                    if l_val in filter:
                        group_var_list[l_val].append(var_id)
            else:
                if l_val not in group_var_list:
                    group_var_list[l_val] = []
                if r_val not in group_var_list:
                    group_var_list[r_val] = []
                if l_val != r_val:
                    group_var_list[l_val].append(var_id)
                    group_var_list[r_val].append(var_id)
                else:
                    group_var_list[l_val].append(var_id)
            pair_var_list[p].append(var_id)
            var_pair_map[var_id] = dict_of_queues_pair[p][j].copy()
            var_id += 1
            
    # constraints: each pair of groups should also satisfy the disparity score constraint
    groups = list(group_var_list.keys())
    print(len(groups))
    for gid1 in range(len(groups)):
        for gid2 in range(len(groups)):
            if gid1 == gid2:
                continue
            g1 = groups[gid1]
            g2 = groups[gid2]
            if pd.isna(g1) or pd.isna(g2):
                continue
            if filter is not None:
                if g1 not in filter or g2 not in filter:
                    continue
            constraint_list.append([])
            for v1 in group_var_list[g1]:
                if var_pair_map[v1][1] == var_pair_map[v1][3]:
                    constraint_list[-1].append((v1, 2.0/distributions[g1]))
                else:
                    constraint_list[-1].append((v1, 1.0/distributions[g1]))
            for v2 in group_var_list[g2]:
                if var_pair_map[v2][1] == var_pair_map[v2][3]:
                    constraint_list[-1].append((v2, (threshold+1) * -2.0/distributions[g2]))
                else:
                    constraint_list[-1].append((v2, (threshold+1) * -1.0/distributions[g2]))

    lp_outf = open('./lp_inputs/lp_{}_{}_{}.lp'.format(dataset, k, threshold), 'w')
    lp_outf.write('maximize\n')
    lp_outf.write('obj: ')
    obj_str = ' + '.join([(str(w_list[i]) + ' x' +str(i))   for i in range(len(w_list))])
    lp_outf.write(obj_str + '\nsubject to\n')
    for i, c in enumerate(constraint_list):
        lp_outf.write('c{}:'.format(i))
        lp_outf.write(' + '.join([(str(x[1]) + ' x' + str(x[0])) for x in c]))
        lp_outf.write(' <= 0\n')
    lp_outf.write('c{}:'.format(len(constraint_list)))
    # constraint: sum of x is k
    lp_outf.write(' + '.join([('x' + str(i)) for i in range(len(w_list))]))
    lp_outf.write(' = {}\n'.format(k))
    lp_outf.write('bounds\n')
    for i in range(len(w_list)):
        lp_outf.write(' 0 <= x{} <= 1\n'.format(i))
    lp_outf.write('end\n')
    lp_outf.close()
    print(distributions)
    # return './lp_{}_{}_{}.lp'.format(dataset, k, threshold), var_pair_map, start

    lp_input_path = './lp_inputs/lp_{}_{}_{}.lp'.format(dataset, k, threshold)
    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)

            # We assume that a problem file was given as the first command
            # line argument (received in `argv')

            # task.readdata('./test_lp.lp')
            task.readdata(lp_input_path)

            # Solve the problem
            task.optimize()

            # Print a summary of the solution
            task.solutionsummary(mosek.streamtype.log)

            # If an output file was specified, save problem to a file
            # If using OPF format, these parameters will specify what to include in output
            task.putintparam(mosek.iparam.opf_write_solutions, mosek.onoffkey.on)
            task.putintparam(mosek.iparam.opf_write_problem, mosek.onoffkey.on)
            task.putintparam(mosek.iparam.opf_write_hints, mosek.onoffkey.off)
            task.putintparam(mosek.iparam.opf_write_parameters, mosek.onoffkey.off)
            # task.writedata('./output_lp_DBLP-ACM_1000_0.1.lp')
            # Get status information about the solution
            solsta = task.getsolsta(mosek.soltype.bas)
            cnt_by_group = {}
            if (solsta == mosek.solsta.optimal):
                xx = task.getxx(mosek.soltype.bas)
                
                print("Optimal solution: ")
                # variables need to be rounded
                for i in range(len(xx)):
                    if xx[i] > 0 and xx[i] < 1:
                        print("x[" + str(i) + "]=" + str(xx[i]))
                        
                top_k_results = []
                sp = {}
                for i in range(len(xx)):
                    if xx[i] == 1:
                        # currently ignore rounding
                        top_k_results.append(var_pair_map[i])                        
                        l_val = var_pair_map[i][1]
                        r_val = var_pair_map[i][3]
                        if not pd.isna(l_val):
                            if l_val not in sp:
                                sp[l_val] = 0
                                cnt_by_group[l_val] = 0
                            sp[l_val] += 1 / distributions[l_val]
                            cnt_by_group[l_val] += 1
                        if not pd.isna(r_val):
                            if r_val not in sp:
                                sp[r_val] = 0
                                cnt_by_group[r_val] = 0
                            sp[r_val] += 1 / distributions[r_val]
                            cnt_by_group[r_val] += 1
                disparity = measure_disparity_advanced(sp, filter)
                print(disparity, filter)
                print(cnt_by_group)
                for g in cnt_by_group:
                    print(g, cnt_by_group[g] / distributions[g])
                # for gid1 in range(len(groups)):
                #     for gid2 in range(len(groups)):
                #         if gid1 == gid2:
                #             continue
                #         g1 = groups[gid1]
                #         g2 = groups[gid2]
                #         cur_sum = 0.0
                #         for v1 in group_var_list[g1]:
                #             if xx[v1] != 1:
                #                 continue
                #             if var_pair_map[v1][1] == var_pair_map[v1][3]:
                #                 cur_sum += xx[v1] * 2.0/distributions[g1]
                #             else:
                #                 cur_sum += xx[v1] * 1.0/distributions[g1]
                #         for v2 in group_var_list[g2]:
                #             if xx[v2] != 1:
                #                 continue
                #             if var_pair_map[v2][1] == var_pair_map[v2][3]:
                #                 cur_sum += xx[v2] * (threshold+1) * -2.0/distributions[g2]
                #             else:
                #                 cur_sum += xx[v2] * (threshold+1) * -1.0/distributions[g2]
                #         print(g1, g2, cur_sum)
                
                # TODO: add rounding and have an accurate thresholding
                # if disparity <= threshold:
                if True:
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
            elif (solsta == mosek.solsta.dual_infeas_cer or
                solsta == mosek.solsta.prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")