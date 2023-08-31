from preprocessor import preprocess_OverlapBlocker, preprocess_Autoencoder


datasets = [
    "DBLP-ACM",
    "DBLP-Scholar",
    "Amazon-Google",
    "Walmart-Amazon",
    "Musicbrainz-persons",
    "Musicbrainz-groups",
]
blocking_attr = "title"
key_attr = "id"
output_attrs = [
    ["title", "authors", "venue", "year"],
    ["title", "authors", "venue", "year"],
    ["title", "manufacturer", "price"],
    ["title", "category", "brand", "modelno", "price"],
    ["title", "length", "artist", "album", "year", "language", "gender", "country"],
    ["title", "length", "artist", "album", "year", "language", "country"],
]
datasets=["Compas"]
output_attrs=[["FullName","Ethnic_Code_Text"]]
for idx in range(len(datasets)):
    # if datasets[idx] in ["Musicbrainz-persons", "Musicbrainz-groups"]:
    #     blocking_attr = "artist"
    # print("=================", datasets[idx], "=================")
    # print("=================OverlapBlocker=================")
    # preprocess_OverlapBlocker(
    #     datasets[idx],
    #     key=key_attr,
    #     blocking_attr=blocking_attr,
    #     output_attrs=output_attrs[idx],
    # )
    print("=================Autoencoder=================")
    preprocess_Autoencoder(datasets[idx], blocking_attr=output_attrs[idx])
