import pathlib
import requests
import json
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt


def download_cellxgene_collections_metadata(
    save_path="../data/cellxgene/collections_metadata.json",
    domain_name="cellxgene.cziscience.com",
    collection_path="/curation/v1/collections/",
):
    """Download a cellxgene collection to given folder
    Args:
        save_path (str, optional):
            path to save file. Defaults to '../data/cellxgene/collections_metadata.json'.
            If None, returns the dataframe without saving
        domain_name (str, optional):
            cellxgene domain name. Defaults to "cellxgene.cziscience.com".
        collection_path (str, optional):
            cellxgene collection path. Defaults to "/curation/v1/collections/"
    Returns:
        cellxgene collections metadata saved in {folder_path}/collections_metadata.json
    """
    # https://github.com/chanzuckerberg/single-cell-curation/blob/main/notebooks/curation_api/python_raw/get_collection.ipynb

    api_url_base = f"https://api.{domain_name}"
    collection_url = f"{api_url_base}{collection_path}"
    response = requests.get(url=collection_url)
    response.raise_for_status()
    collections = response.json()
    response.close()

    for i, collection in enumerate(collections):
        print(i)
        df_ = pd.DataFrame(collection["datasets"])
        for k, v in collection.items():
            if "datasets" != k:
                if type(v) in [list, dict]:
                    df_[k] = [v for i in df_.index]
                else:
                    df_[k] = v
        if i == 0:
            df = df_
        else:
            df = pd.concat([df, df_], axis=0)
    df.index = range(len(df))
    if save_path is None:
        return df
    else:
        df.to_json(save_path)


def download_cellxgene_collection(
    collection_id,
    titles_contain=[],
    assays_contain=[],
    overwrite=False,
    folder_path="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/jbac/projects/data/cellxgene",
    domain_name="cellxgene.cziscience.com",
):
    """Download a cellxgene collection to given folder
    Args:
        collection_id (str):
            cellxgene collection_id
        titles_contain (list, optional):
            list of strings. Datasets are kept if they contain any of these strings in their title. Defaults to [] which returns all datasets
        overwrite (bool, optional):
            overwrite existing files. Defaults to False.
        folder_path (str, optional):
            folder to save files. Defaults to '/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/jbac/projects/data/cellxgene'.
        domain_name (str, optional):
            cellxgene domain name. Defaults to "cellxgene.cziscience.com".

    Returns:
        cellxgene datasets saved in folder_path
    """
    # https://github.com/chanzuckerberg/single-cell-curation/blob/main/notebooks/curation_api/python_raw/get_collection.ipynb

    api_url_base = f"https://api.{domain_name}"
    collection_path = f"/curation/v1/collections/{collection_id}"
    collection_url = f"{api_url_base}{collection_path}"
    response = requests.get(url=collection_url)
    response.raise_for_status()
    response_content = response.json()
    response.close()

    # Check if the folder exists, create it if not
    folder = pathlib.Path(folder_path) / collection_id
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "contents.json", "w", encoding="utf-8") as f:
        json.dump(response_content, f, ensure_ascii=False, indent=4)
    print("Collection information downloaded and saved to:", folder / "contents.json")

    for i in range(len(response_content["datasets"])):
        if len(titles_contain) and not any(
            t in response_content["datasets"][i]["title"] for t in titles_contain
        ):
            continue
        if len(assays_contain) and not any(
            t in response_content["datasets"][i]["assay"][0]["label"]
            for t in assays_contain
        ):
            continue
        else:
            for data_dict in response_content["datasets"][i]["assets"]:
                # Extract the download URL from the data dictionary
                dataset_url = data_dict["url"]

                # Extract the filename from the URL
                filename = dataset_url.split("/")[-1]

                # Construct the full path where the file will be saved
                file_path = folder / filename

                if file_path.exists() and not overwrite:
                    print(
                        f"File exists and overwrite is set to False, skipping: {file_path}"
                    )
                    continue
                else:
                    response_data = requests.get(url=dataset_url)
                    response_data.raise_for_status()
                    # Write the file to the specified folder
                    with open(file_path, "wb") as f:
                        for chunk in response_data.iter_content(chunk_size=10240):
                            if chunk:
                                f.write(chunk)

                    print(f"File downloaded and saved to: {file_path}")
                    response_data.close()


def _explode_list_columns(
    df,
    list_columns=[
        "assay",
        "cell_type",
        "development_stage",
        "disease",
        "self_reported_ethnicity",
        "sex",
        "suspension_type",
        "tissue",
    ],
):
    for col in list_columns:
        exploded = (
            df[col]
            .apply(lambda x: pd.DataFrame(x).to_dict("list"))
            .apply(pd.Series)
            .add_prefix(f"{col}_")
        )
        df = pd.concat([df, exploded], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df


def load_cellxgene_collections_metadata(
    file_path="../data/cellxgene/collections_metadata.json",
    explode_columns=True,
    list_columns=["assay", "disease", "organism", "suspension_type", "tissue", "links"],
):
    """Load a cellxgene collection to given folder

    Args:
        file_path (str): The path to the JSON file containing the collection metadata.
            Defaults to '../data/cellxgene/collections_metadata.json'.
        explode_columns (bool): Whether to explode the list columns into separate columns.
            Defaults to True.
        list_columns (List[str]): The list columns to explode. Defaults to the list of columns
            specified in the function definition.

    Returns:
        pandas.DataFrame: The cellxgene datasets saved in the folder_path.
    """

    # Read the JSON file containing the collection metadata
    df = pd.read_json(file_path)

    # Explode the list columns into separate columns, if specified
    if explode_columns:
        df = _explode_list_columns(df, list_columns=list_columns)

    return df


def load_cellxgene_collection_contents(
    collection_id,
    folder_path="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/jbac/projects/data/cellxgene",
):
    """Load a cellxgene collection in given folder
    Parameters:
        collection_id (str):
            cellxgene collection_id
        domain_name (str, optional):
            cellxgene domain name. Defaults to "cellxgene.cziscience.com".

    Returns:
        cellxgene datasets saved in folder_path
    """

    with open(
        pathlib.Path(folder_path) / f"{collection_id}/contents.json",
        "r",
        encoding="utf-8",
    ) as f:
        collection_contents = json.load(f)

    df = _explode_list_columns(pd.DataFrame(collection_contents["datasets"]))
    return collection_contents, df


def adata_X_sanity_check(adata, n=3):
    print(".X row 0 unique values", np.unique(adata.X[0].toarray().flatten()))

    if adata.raw is not None:
        print(
            ".raw.X row 0 unique values", np.unique(adata.raw.X[0].toarray().flatten())
        )
        for i in np.random.randint(0, adata.shape[0], n):
            x = adata.raw.X[i].toarray().flatten().copy()
            x = 1e4 * x / np.sum(x)
            x = np.log1p(x)
            plt.scatter(x, adata.X[i].toarray(), s=1)


def sparse_mean_var(adata):
    """
    Calculate mean and variance of the sparse adata.X and store the results in the adata object.
    """
    scalar = sklearn.preprocessing.StandardScaler(with_mean=False).fit(adata.X)
    adata.var["means"] = scalar.mean_
    adata.var["variances"] = scalar.var_
    adata.var["log1p_means"] = np.log1p(scalar.mean_)
    adata.var["log1p_variances"] = np.log1p(scalar.var_)
