import pathlib
import requests
import json
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt


def download_cellxgene_collection(
    collection_id,
    titles_contain=[],
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
    folder = (
        pathlib.Path(
            "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/jbac/projects/data/cellxgene"
        )
        / collection_id
    )
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "contents.json", "w", encoding="utf-8") as f:
        json.dump(response_content, f, ensure_ascii=False, indent=4)
    print("Collection information downloaded and saved to:", folder / "contents.json")

    for i in range(len(response_content["datasets"])):
        if len(titles_contain) and not any(
            t in response_content["datasets"][i]["title"] for t in titles_contain
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


def load_cellxgene_collection_contents(
    collection_id,
    folder_path="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/jbac/projects/data/cellxgene",
):
    """Download a cellxgene collection to given folder
    Parameters:
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

    def _explode_list_columns(
        df,
        list_columns=[
            "assay",
            "assets",
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

    with open(pathlib.Path(folder_path) / f"{collection_id}/contents.json", "r") as f:
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
