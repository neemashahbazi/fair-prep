from overlap_blocker_helper import OverlapBlocker
from utils import save_pickle, load_csv
from libraries.DeepBlocker.tuple_embedding_models import AutoEncoderTupleEmbedding
from libraries.DeepBlocker.deep_blocker import DeepBlocker
from libraries.DeepBlocker.vector_pairing_models import ExactTopKVectorPairing
from libraries.DeepBlocker.vector_pairing_models import ExactTopKVectorPairing


def preprocess_OverlapBlocker(dataset, key, blocking_attr, output_attrs):
    """for each table this function dumps the intermediate results (bag of tokens) preprocessed by OverlapBlocker in the data set folder
    Args:
        dataset (String): name of dataset
        key (String):
        blocking_attr (String): the attribute blocking is being applied on
        output_attrs (List):
    Return:
        None
    """
    ob = OverlapBlocker()
    tableA_array, tableB_array = ob.save_intermediate_results(
        dataset, key, blocking_attr, output_attrs
    )

    save_pickle("datasets/" + dataset + "/tableA_OverlapBlocker.pkl", tableA_array)
    save_pickle("datasets/" + dataset + "/tableB_OverlapBlocker.pkl", tableB_array)


def preprocess_Autoencoder(
    dataset,
    blocking_attr,
):
    """for each table this function dumps the intermediate results (bag of tokens) preprocessed by Autoencoder in the data set folder
    Args:
        dataset (String): name of dataset
        blocking_attr (List): list of attributes blocking is being applied on
    Return:
        None

    """
    tableA = load_csv("datasets/" + dataset + "/tableA.csv")
    tableB = load_csv("datasets/" + dataset + "/tableB.csv")
    tuple_embedding_model = AutoEncoderTupleEmbedding()
    vector_pairing_model = ExactTopKVectorPairing(K=50)
    db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
    tableA_embedding, tableB_embedding = db.save_intermediate_results(
        tableA, tableB, blocking_attr
    )
    save_pickle("datasets/" + dataset + "/tableA_Autoencoder.pkl", tableA_embedding)
    save_pickle("datasets/" + dataset + "/tableB_Autoencoder.pkl", tableB_embedding)


