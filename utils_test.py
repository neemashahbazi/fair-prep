from utils import (
    single_fairness,
    pairwise_fairness,
    single_table_cardinality,
    candidate_set_cardinality,
    positive_set_cardinality,
    load_csv,
    save_sensitive_attr,
    negative_set_cardinality,
    get_group_distributions,
)
import pandas as pd


methods = [
    "OverlapBlocker",
    "Autoencoder",
]
datasets = [
    "DBLP-ACM",
    # "Amazon-Google",
    # "DBLP-Scholar",
    # "Walmart-Amazon",
    "Musicbrainz-persons",
    "Musicbrainz-groups",
]
sensitive_attrs = ["venue", 
                #    "manufacturer", "venue", "category", 
                   "country", "country"]

# for idx in range(len(datasets)):
#     save_sensitive_attr(datasets[idx],"id",sensitive_attrs[idx])

# Fairness evaluation
for idx in range(len(datasets)):
    print("==================== Dataset: " + datasets[idx] + "====================")
    tableA = load_csv("datasets/" + datasets[idx] + "/tableA.csv")
    tableB = load_csv("datasets/" + datasets[idx] + "/tableB.csv")
    ltable_sensitive_attr = load_csv(
        "datasets/" + datasets[idx] + "/tableA_sensitive_attr.csv"
    )
    rtable_sensitive_attr = load_csv(
        "datasets/" + datasets[idx] + "/tableB_sensitive_attr.csv"
    )
    ltable_id = "ltable_id"
    rtable_id = "rtable_id"

    distributions_single = get_group_distributions(
        datasets[idx], sensitive_attrs[idx], type="single"
    )
    distributions_pairwise = get_group_distributions(
        datasets[idx], sensitive_attrs[idx], type="pairwise"
    )

    for method in methods:
        print(
            "==================== Blocking Method: " + method + "===================="
        )
        candidate_set = load_csv(
            "candidate_sets/" + method + "/" + datasets[idx] + "/candidate_set.csv"
        )
        print("single fairness:")
        print(
            single_fairness(
                candidate_set,
                ltable_sensitive_attr,
                rtable_sensitive_attr,
                ltable_id,
                rtable_id,
                distributions_single
            )
        )
        print("pairwise fairness:")
        print(
            pairwise_fairness(
                candidate_set,
                ltable_sensitive_attr,
                rtable_sensitive_attr,
                ltable_id,
                rtable_id,
                distributions_pairwise
            )
        )
        print(
            "=========================================================================================="
        )

# Cardinality and recall evaluation
# for idx in range(len(datasets)):
#     print("==================== Dataset: ", datasets[idx] + "====================")
#     tableA = load_csv("datasets/" + datasets[idx] + "/tableA.csv")
#     tableB = load_csv("datasets/" + datasets[idx] + "/tableB.csv")
#     ltable_sensitive_attr = load_csv(
#         "datasets/" + datasets[idx] + "/tableA_sensitive_attr.csv"
#     )
#     rtable_sensitive_attr = load_csv(
#         "datasets/" + datasets[idx] + "/tableB_sensitive_attr.csv"
#     )
#     ltable_id = "ltable_id"
#     rtable_id = "rtable_id"
#     print("====================  Input Results ==================== ")
#     print("====================  tableA ==================== ")
#     print(single_table_cardinality(ltable_sensitive_attr))
#     print("====================  tableB ==================== ")
#     print(single_table_cardinality(rtable_sensitive_attr))
#     print("====================  Cross Product ==================== ")
#     input_set = pd.merge(tableA, tableB, how="cross")
#     print(
#         candidate_set_cardinality(
#             input_set,
#             ltable_sensitive_attr,
#             rtable_sensitive_attr,
#             ltable_id="id_x",
#             rtable_id="id_y",
#         )
#     )
#     for method in methods:
#         print(
#             "==================== Blocking Method: " + method + "===================="
#         )
#         candidate_set = load_csv(
#             "candidate_sets/" + method + "/" + datasets[idx] + "/candidate_set.csv"
#         )
#         golden_set = load_csv("datasets/" + datasets[idx] + "/matches.csv")

#         print("====================  Candidate Set ==================== ")
#         print(
#             candidate_set_cardinality(
#                 candidate_set,
#                 ltable_sensitive_attr,
#                 rtable_sensitive_attr,
#                 ltable_id,
#                 rtable_id,
#             )
#         )
#         print("====================  Positive Set ==================== ")
#         print(
#             positive_set_cardinality(
#                 candidate_set,
#                 golden_set,
#                 ltable_sensitive_attr,
#                 rtable_sensitive_attr,
#                 ltable_id,
#                 rtable_id,
#             )
#         )
#         print("====================  Negative Set ==================== ")
#         print(
#             negative_set_cardinality(
#                 candidate_set,
#                 golden_set,
#                 ltable_sensitive_attr,
#                 rtable_sensitive_attr,
#                 ltable_id,
#                 rtable_id,
#             )
#         )
