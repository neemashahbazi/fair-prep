from blocking import (
    baseline_greedy,
    baseline_stratified,
    baseline_no_fairness,
    baseline_greedy_OverlapBlocker,
    advanced_single_pass,
)
from utils import (
    recall,
    load_csv,
    save_csv,
    get_sorted_tuples_Autoencoder,
    get_sorted_tuples_OverlapBlocker_Advanced,
    get_sorted_tuples_OverlapBlocker,
    get_sorted_tuples_Sudowoodo,
    load_pickle,
    write_results_to_file,
)
from lp import lp_solve
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="dataset name")
parser.add_argument("--threshold", type=float, help="fairness threshold")
parser.add_argument("--k", type=int, help="k value")
parser.add_argument("--sensitive_attr", type=str, help="sensitive attribute")
parser.add_argument("--method", type=str, help="blocking method")

args = parser.parse_args()

blocking_attr = "title"
key_attr = "id"
map = None
filter = None

if args.dataset in ["Musicbrainz-persons", "Musicbrainz-groups"]:
    blocking_attr = "artist"
    filter = [
        "US",
        "GB",
        "DE",
        "FR",
        "JP",
        "IT",
        "AT",
        "CA",
        "NL",
        "RU",
        "FI",
        "ES",
    ]
    overlap_size = 1

if args.dataset == "DBLP-ACM":
    map = {
        "sigmod record": "sigmod rec.",
        "acm trans . database syst .": "tods",
        "sigmod conference": "sigmod",
        "vldb": "vldb",
        "vldb j.": "vldb j.",
        "acm sigmod record": "sigmod rec.",
        "acm transactions on database systems ( tods )": "tods",
        "very large data bases": "vldb",
        "the vldb journal -- the international journal on very large data bases": "vldb j.",
        "international conference on management of data": "sigmod",
    }
    overlap_size = 4
if args.dataset == "DBLP-Scholar":
    map = {
        "sigmod record": "sigmod rec.",
        "acm trans . database syst .": "tods",
        "sigmod conference": "sigmod",
        "vldb": "vldb",
        "vldb j.": "vldb j.",
        "acm sigmod record": "sigmod rec.",
        "acm transactions on database systems ( tods )": "tods",
        "acm transactions on database systems": "tods",
        "tods": "tods",
        "very large data bases": "vldb",
        "the vldb journal -- the international journal on very large data bases": "vldb j.",
        "international conference on management of data": "sigmod",
        "the vldb journal the international journal on very large & hellip ;": "vldb j.",
        "proceedings of the 31st international conference on very & hellip ;": "vldb",
        "proceedings of the 2005 acm sigmod international conference & hellip ;": "sigmod",
        "vldb journal": "vldb j.",
        "proceedings of the acm sigmod international conference on": "sigmod",
        "acm tods": "tods",
        "proceedings of the acm sigmod international conference on & hellip ;": "sigmod",
        "proc . acm sigmod": "sigmod",
        "proceedings of the 2004 acm sigmod international conference & hellip ;": "sigmod",
        "proceedings of the 1995 acm sigmod international conference & hellip ;": "sigmod",
        "acm sigmod": "sigmod",
        "proc . acm sigmod": "sigmod",
        "sigmod": "sigmod",
        "sigmod record ,": "sigmod rec.",
        "acm trans . database syst . ,": "tods",
        "sigmod conference ,": "sigmod",
        "vldb ,": "vldb",
        "vldb j. ,": "vldb j.",
        "acm sigmod record ,": "sigmod rec.",
        "acm transactions on database systems ( tods ) ,": "tods",
        "acm transactions on database systems ,": "tods",
        "tods ,": "tods",
        "very large data bases ,": "vldb",
        "the vldb journal -- the international journal on very large data bases ,": "vldb j.",
        "international conference on management of data ,": "sigmod",
        "the vldb journal the international journal on very large & hellip ; ,": "vldb j.",
        "proceedings of the 31st international conference on very & hellip ; ,": "vldb",
        "proceedings of the 2005 acm sigmod international conference & hellip ; ,": "sigmod",
        "vldb journal ,": "vldb j.",
        "proceedings of the acm sigmod international conference on ,": "sigmod",
        "acm tods ,": "tods",
        "proceedings of the acm sigmod international conference on & hellip ; ,": "sigmod",
        "proc . acm sigmod ,": "sigmod",
        "proceedings of the 2004 acm sigmod international conference & hellip ; ,": "sigmod",
        "proceedings of the 1995 acm sigmod international conference & hellip ; ,": "sigmod",
        "acm sigmod ,": "sigmod",
        "proc . acm sigmod ,": "sigmod",
        "sigmod ,": "sigmod",
    }
    filter = ["sigmod rec.", "tods", "sigmod", "vldb", "vldb j."]
    overlap_size = 4

if args.dataset == "Walmart-Amazon":
    filter = [
        "electronics - general",
        "stationery & office machinery",
        "mp3 accessories",
        "printers",
        "computers accessories",
        "laminating supplies",
        "headphones",
        "storage presentation materials",
        "inkjet printer ink",
        "bags cases",
        "mice",
        "cases sleeves",
        "audio video accessories",
        "point shoot digital cameras",
        "projection screens",
        "usb flash drives",
        "cases",
        "cases bags",
        "laser printer toner",
        "memory",
        "printer ink toner",
        "accessories supplies",
    ]
    overlap_size = 3
if args.method == "OverlapBlocker":
    print("=================", args.dataset, "=================")
    print("================= OverlapBlocker =================")
    sorted_tuples = get_sorted_tuples_OverlapBlocker(
        dataset=args.dataset,
        key_attr=key_attr,
        filter_attr=blocking_attr,
        overlap_size=overlap_size,
    )
    print("********** No Fairness **********")
    candidate_set, disparity, duration = baseline_no_fairness(
        sorted_tuples, args.dataset, args.k, args.sensitive_attr, map, filter
    )
    save_csv(
        "candidate_sets/OverlapBlocker/"
        + args.dataset
        + "/candidate_set_no_fairness.csv",
        candidate_set,
    )
    recall_ = recall(
        candidate_set,
        load_csv("datasets/" + args.dataset + "/matches.csv"),
        "ltable_id",
        "rtable_id",
    )
    write_results_to_file(
        args.dataset,
        args.k,
        args.threshold,
        recall_,
        disparity,
        duration,
        args.method,
        "No Fairness",
    )

    print("********** Stratified **********")
    candidate_set, disparity, duration = baseline_stratified(
        sorted_tuples,
        args.dataset,
        args.k,
        args.sensitive_attr,
        args.threshold,
        map,
        filter,
    )
    save_csv(
        "candidate_sets/OverlapBlocker/"
        + args.dataset
        + "/candidate_set_stratified.csv",
        candidate_set,
    )
    recall_ = recall(
        candidate_set,
        load_csv("datasets/" + args.dataset + "/matches.csv"),
        "ltable_id",
        "rtable_id",
    )
    write_results_to_file(
        args.dataset,
        args.k,
        args.threshold,
        recall_,
        disparity,
        duration,
        args.method,
        "Stratified",
    )

    print("********** Greedy Advanced **********")
    sorted_tuples = get_sorted_tuples_OverlapBlocker_Advanced(
        dataset=args.dataset,
        key_attr=key_attr,
        filter_attr=blocking_attr,
        max_overlap_size=overlap_size,
    )
    candidate_set, disparity, duration = baseline_greedy_OverlapBlocker(
        sorted_tuples,
        args.dataset,
        args.k,
        args.sensitive_attr,
        args.threshold,
        map,
        filter,
    )
    save_csv(
        "candidate_sets/OverlapBlocker/"
        + args.dataset
        + "/candidate_set_greedy_OverlapBlocker.csv",
        candidate_set,
    )
    recall_ = recall(
        candidate_set,
        load_csv("datasets/" + args.dataset + "/matches.csv"),
        "ltable_id",
        "rtable_id",
    )
    write_results_to_file(
        args.dataset,
        args.k,
        args.threshold,
        recall_,
        disparity,
        duration,
        args.method,
        "Greedy",
    )

if args.method == "Autoencoder":
    print("================= Autoencoder =================")
    ltable_embedding = load_pickle(
        "datasets/" + args.dataset + "/tableA_Autoencoder.pkl"
    )
    rtable_embedding = load_pickle(
        "datasets/" + args.dataset + "/tableB_Autoencoder.pkl"
    )
    sorted_tuples = get_sorted_tuples_Autoencoder(
        args.dataset, ltable_embedding, rtable_embedding
    )

    print("********** No Fairness **********")
    candidate_set, disparity, duration = baseline_no_fairness(
        sorted_tuples, args.dataset, args.k, args.sensitive_attr, map, filter
    )
    save_csv(
        "candidate_sets/Autoencoder/" + args.dataset + "/candidate_set_no_fairness.csv",
        candidate_set,
    )
    recall_ = recall(
        candidate_set,
        load_csv("datasets/" + args.dataset + "/matches.csv"),
        "ltable_id",
        "rtable_id",
    )
    write_results_to_file(
        args.dataset,
        args.k,
        args.threshold,
        recall_,
        disparity,
        duration,
        args.method,
        "No Fairness",
    )

    print("********** Greedy **********")
    candidate_set, disparity, duration = baseline_greedy(
        sorted_tuples,
        args.dataset,
        args.k,
        args.sensitive_attr,
        args.threshold,
        map,
        filter,
    )
    save_csv(
        "candidate_sets/Autoencoder/" + args.dataset + "/candidate_set_greedy.csv",
        candidate_set,
    )
    recall_ = recall(
        candidate_set,
        load_csv("datasets/" + args.dataset + "/matches.csv"),
        "ltable_id",
        "rtable_id",
    )
    write_results_to_file(
        args.dataset,
        args.k,
        args.threshold,
        recall_,
        disparity,
        duration,
        args.method,
        "Greedy",
    )

    print("********** Stratified **********")
    candidate_set, disparity, duration = baseline_stratified(
        sorted_tuples,
        args.dataset,
        args.k,
        args.sensitive_attr,
        args.threshold,
        map,
        filter,
    )
    save_csv(
        "candidate_sets/Autoencoder/" + args.dataset + "/candidate_set_stratified.csv",
        candidate_set,
    )
    recall_ = recall(
        candidate_set,
        load_csv("datasets/" + args.dataset + "/matches.csv"),
        "ltable_id",
        "rtable_id",
    )
    write_results_to_file(
        args.dataset,
        args.k,
        args.threshold,
        recall_,
        disparity,
        duration,
        args.method,
        "Stratified",
    )
    print("********** Advanced Single Pass **********")
    candidate_set, disparity, duration = advanced_single_pass(
        sorted_tuples,
        args.dataset,
        args.k,
        args.sensitive_attr,
        args.threshold,
        1,
        map,
        filter,
    )
    save_csv(
        "candidate_sets/Autoencoder/" + args.dataset + "/candidate_set_single_pass.csv",
        candidate_set,
    )
    recall_ = recall(
        candidate_set,
        load_csv("datasets/" + args.dataset + "/matches.csv"),
        "ltable_id",
        "rtable_id",
    )
    write_results_to_file(
        args.dataset,
        args.k,
        args.threshold,
        recall_,
        disparity,
        duration,
        args.method,
        "Advanced Single Pass",
    )


if args.method == "Sudowoodo":
    print("================= Sudowoodo =================")
    ltable_embedding = load_pickle("datasets/" + args.dataset + "/tableA_Sudowoodo.pkl")
    rtable_embedding = load_pickle("datasets/" + args.dataset + "/tableB_Sudowoodo.pkl")
    sorted_tuples = get_sorted_tuples_Sudowoodo(
        args.dataset, ltable_embedding, rtable_embedding
    )

    # print("********** No Fairness **********")
    # candidate_set, disparity, duration = baseline_no_fairness(
    #     sorted_tuples, args.dataset, args.k, args.sensitive_attr, map, filter
    # )
    # save_csv(
    #     "candidate_sets/Sudowoodo/" +
    #     args.dataset + "/candidate_set_no_fairness.csv",
    #     candidate_set,
    # )
    # recall_ = recall(
    #     candidate_set,
    #     load_csv("datasets/" + args.dataset + "/matches.csv"),
    #     "ltable_id",
    #     "rtable_id",
    # )
    # write_results_to_file(args.dataset, args.k,
    #                       args.threshold, recall_, disparity, duration, args.method, "No Fairness")

    # print("********** Greedy **********")
    # candidate_set, disparity, duration = baseline_greedy(
    #     sorted_tuples, args.dataset, args.k, args.sensitive_attr, args.threshold, map, filter
    # )
    # save_csv(
    #     "candidate_sets/Sudowoodo/" +
    #     args.dataset + "/candidate_set_greedy.csv",
    #     candidate_set,
    # )
    # recall_ = recall(
    #     candidate_set,
    #     load_csv("datasets/" + args.dataset + "/matches.csv"),
    #     "ltable_id",
    #     "rtable_id",
    # )
    # write_results_to_file(args.dataset, args.k,
    #                       args.threshold, recall_, disparity, duration, args.method, "Greedy")

    # print("********** Stratified **********")
    # candidate_set, disparity, duration = baseline_stratified(
    #     sorted_tuples, args.dataset, args.k, args.sensitive_attr, args.threshold, map, filter
    # )
    # save_csv(
    #     "candidate_sets/Sudowoodo/" +
    #     args.dataset + "/candidate_set_stratified.csv",
    #     candidate_set,
    # )
    # recall_ = recall(
    #     candidate_set,
    #     load_csv("datasets/" + args.dataset + "/matches.csv"),
    #     "ltable_id",
    #     "rtable_id",
    # )
    # write_results_to_file(args.dataset, args.k,
    #                       args.threshold, recall_, disparity, duration, args.method, "Stratified")

    # print("********** Advanced Single Pass **********")
    # candidate_set, disparity, duration = advanced_single_pass(
    #     sorted_tuples, args.dataset, args.k, args.sensitive_attr, args.threshold, 1, map, filter)
    # save_csv(
    #     "candidate_sets/Sudowoodo/" +
    #     args.dataset + "/candidate_set_single_pass.csv",
    #     candidate_set,
    # )
    # recall_ = recall(
    #     candidate_set,
    #     load_csv("datasets/" + args.dataset + "/matches.csv"),
    #     "ltable_id",
    #     "rtable_id",
    # )
    # write_results_to_file(args.dataset, args.k,
    #                       args.threshold, recall_, disparity, duration, args.method, "Advanced Single Pass")

    candidate_set, disparity, duration = lp_solve(
        sorted_tuples,
        args.dataset,
        args.k,
        args.sensitive_attr,
        args.threshold,
        map=map,
        filter=filter,
    )
    save_csv(
        "candidate_sets/Sudowoodo/" + args.dataset + "/candidate_set_lp.csv",
        candidate_set,
    )
    recall_ = recall(
        candidate_set,
        load_csv("datasets/" + args.dataset + "/matches.csv"),
        "ltable_id",
        "rtable_id",
    )
    write_results_to_file(
        args.dataset,
        args.k,
        args.threshold,
        recall_,
        disparity,
        duration,
        args.method,
        "LP",
    )
