import scanpy as sc
import scvi
import scarches as sca
import ot
import os
import pathlib
import importlib

if importlib.util.find_spec("rapids_singlecell") is not None:
    # Import gpu libraries, Initialize rmm and cupy
    import rapids_singlecell as rsc
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator


def check_gpu_availability():
    # Check if CUDA_VISIBLE_DEVICES environment variable is set
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        # GPUs are available
        return True
    else:
        # GPUs are not available
        return False


def preprocess(
    adata,
    normalize=False,  # Normalize total counts
    log1p=False,  # Log1p transform
    pca=False,  # Perform PCA
    scale=False,  # Scale data
    umap=False,  # Perform UMAP
    n_comps=30,  # Number of PCA components
    n_neighbors=15,  # Number of neighbors for kNN
    metric="cosine",  # Metric for kNN
    backend="gpu",  # "gpu" or "cpu"
    device=0,  # Device ID for GPU backend
    save_raw=True,  # Whether to save raw data
    verbose=True,  # Optional verbose output
):
    """
    Preprocesses an AnnData object.

    Args:
        adata: AnnData object to preprocess.
        normalize (bool): Whether to normalize total counts.
        log1p (bool): Whether to log1p transform.
        pca (bool): Whether to perform PCA.
        scale (bool): Whether to scale data.
        umap (bool): Whether to perform UMAP.
        n_comps (int): Number of PCA components.
        n_neighbors (int): Number of neighbors for kNN.
        metric (str): Metric for kNN.
        backend (str): "gpu" or "cpu".
        device (int): Device ID for GPU backend.
        save_raw (bool): Whether to save raw data in layers['counts']
        verbose (bool): Whether to print verbose output.

    Returns:
        None
    """
    if "preprocess" in adata.uns:
        print("Warning: preprocess key already found in adata.uns")

    if save_raw:
        if verbose:
            print("Saving raw counts in layers['counts']...")
        adata.layers["counts"] = adata.X

    ### optional switch to GPU backend ###
    if backend == "gpu":
        if not check_gpu_availability():
            print("GPU not available. Switching to CPU backend...")
            xsc = sc
            backend = "cpu"
        else:
            # allow memory oversubscription, transfer data to GPU
            rmm.reinitialize(managed_memory=True, devices=device)
            cp.cuda.set_allocator(rmm_cupy_allocator)
            if verbose:
                print("Transferring data to GPU...")
            rsc.get.anndata_to_GPU(adata)
            xsc = rsc
    else:
        xsc = sc

    ### preprocessing ###
    if normalize:
        if verbose:
            print("Normalizing total counts...")
        xsc.pp.normalize_total(adata, target_sum=1e4)
    if log1p:
        if verbose:
            print("Applying log1p transformation...")
        xsc.pp.log1p(adata)
    if scale:
        if verbose:
            print("Scaling data...")
        xsc.pp.scale(adata)
    if pca:
        if verbose:
            print("Performing PCA...")
        xsc.tl.pca(adata, n_comps=n_comps)
    if umap:
        if verbose:
            print("Performing UMAP...")
        xsc.pp.neighbors(adata, n_pcs=n_comps, n_neighbors=n_neighbors, metric=metric)
        xsc.tl.umap(adata)

    # Transfer data back to CPU if using GPU backend
    if backend == "gpu":
        if verbose:
            print("Transferring data back to CPU...")
        rsc.get.anndata_to_CPU(adata)
    adata.uns["preprocess"] = dict(
        normalize=normalize,
        log1p=log1p,
        pca=pca,
        scale=scale,
        umap=umap,
        n_comps=n_comps,
        n_neighbors=n_neighbors,
        metric=metric,
        backend=backend,
        device=device,
    )


def prepare_adatas_hvg_split(ads, path=None, label="dataset_merge_id", overwrite=False):
    """
    Prepare data by finding HVGs for each AnnData object and
    creating a joint AnnData with union, intersection, and strict_intersection of HVGs.

    Args:
        ads (dict): Dictionary of AnnData objects.
        path (str): Path to save the processed data.
        label (str): Label column of dataset ids in the joint AnnData object.
    Returns:
        None
    """

    # Get common genes across all AnnData objects
    common_genes = list(
        set.intersection(*map(set, [a.var_names for a in ads.values()]))
    )
    assert len(common_genes) > 0

    # Concatenate AnnData objects and save as ad_all
    ad_all = sc.concat(ads, label=label, join="outer")[:, common_genes]
    ad_all.obs["obs_names"] = ad_all.obs_names
    ad_all.obs_names_make_unique()
    ad_all.write(f"{path}_all.h5ad")

    # Loop over HVG modes
    for hvg_mode in ["union", "strict_intersection", "intersection"]:
        print(hvg_mode)

        # Check if file exists and skip if so
        if (pathlib.Path(f"{path}_{hvg_mode}.h5ad").is_file()) and (not overwrite):
            print("\tfound saved file")
            continue
        else:
            hvgs = []  # List to store HVGs for each AnnData object

            # Find HVGs for each AnnData object
            for k in ads.keys():
                ad = ads[k][:, common_genes]
                sc.pp.highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=2000)
                hvgs.append(ad.var["highly_variable"][ad.var["highly_variable"]].index)

            # Create joint HVG and save
            if hvg_mode == "union":
                union_hvgs = list(set.union(*map(set, hvgs)))
                ad_joint = ad_all[:, union_hvgs].copy()
            elif hvg_mode == "strict_intersection":
                intersection_hvgs = list(set.intersection(*map(set, hvgs)))
                ad_joint = ad_all[:, intersection_hvgs].copy()
            elif hvg_mode == "intersection":
                sc.pp.highly_variable_genes(
                    ad_all,
                    flavor="seurat_v3",
                    batch_key="dataset_merge_id",
                    n_top_genes=2000,
                )
                ad_joint = ad_all[
                    :,
                    ad_all.var["highly_variable"][ad_all.var["highly_variable"]].index,
                ].copy()

            if path is None:
                return ad_joint
            else:
                # Save joint HVG
                ad_joint.write(f"{path}_{hvg_mode}.h5ad")
                del ad_joint


def prepare_adatas_hvg(ads, path=None, label="dataset_merge_id"):
    """
    Prepare data by finding HVGs for each AnnData object and
    creating a joint AnnData.

    Args:
        ads (dict): Dictionary of AnnData objects.
        path (str): Path to save the processed data.
        label (str): Label column of dataset ids in the joint AnnData object.
    Returns:
        None
    """

    # Find common genes across all AnnData objects
    common_genes = list(
        set.intersection(*map(set, [a.var_names for a in ads.values()]))
    )
    assert len(common_genes) > 0

    # Concatenate AnnData objects and save as ad_all
    ad_all = sc.concat(ads, label=label, join="outer")[:, common_genes]
    ad_all.obs["obs_names"] = ad_all.obs_names
    ad_all.obs_names_make_unique()
    ad_all.layers["counts"] = ad_all.X

    # Loop over HVG modes
    for hvg_mode in ["union", "strict_intersection", "intersection"]:
        print(hvg_mode)

        # Find HVGs for each AnnData object
        hvgs_batch = []
        for k in ads.keys():
            ad = ads[k][:, common_genes]
            sc.pp.highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=2000)
            hvgs_batch.append(
                ad.var["highly_variable"][ad.var["highly_variable"]].index
            )

        # Create joint HVG and save
        if hvg_mode == "union":
            hvgs = list(set.union(*map(set, hvgs_batch)))
        elif hvg_mode == "strict_intersection":
            hvgs = list(set.intersection(*map(set, hvgs_batch)))
        elif hvg_mode == "intersection":
            hvgs = (
                sc.pp.highly_variable_genes(
                    ad_all,
                    flavor="seurat_v3",
                    batch_key="dataset_merge_id",
                    n_top_genes=2000,
                    inplace=False,
                )
                .query("highly_variable==True")
                .index
            )

        # Add joint HVG to ad_all and save
        ad_all.var[f"highly_variable_{hvg_mode}"] = False
        ad_all.var.loc[hvgs, f"highly_variable_{hvg_mode}"] = True

    if path is None:
        return ad_all
    else:
        # Save joint HVG
        ad_all.write(path)
        del ad_all


def embed_adata_cellxgene_scvi(
    adata,
    batch_key="sample_name",
    cxg_scvi_retrain=False,
    n_latent=None,
    model_filename="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/jbac/projects/data/cellxgene/models/2023-12-15-scvi-homo-sapiens/scvi.model",
):
    """
    Embeds adata using scvi and cellxgene retrained model.

    Args:
        adata : AnnData
            Annotated data matrix.
        cxg_scvi_retrain : bool, optional
            Whether to use the cellxgene retrained model, by default False.
        model_filename : str, optional
            File path to the scvi model
    Returns:
        np.ndarray
            Latent representation of the input data.
    """
    adata_ = sc.AnnData(adata.layers["counts"], obs=adata.obs, var=adata.var)
    adata_.var_names = adata.var_names
    adata_.obs_names = adata.obs_names
    adata_.var["ensembl_id"] = adata_.var.index
    adata_.obs["n_counts"] = adata_.X.sum(axis=1)
    adata_.obs["joinid"] = list(range(adata_.n_obs))
    adata_.obs["batch"] = adata_.obs[batch_key]

    scvi.model.SCVI.prepare_query_anndata(adata_, model_filename)
    if cxg_scvi_retrain:
        model = scvi.model.SCVI.load_query_data(
            adata_, model_filename, freeze_dropout=True
        )
        model.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0))
    else:
        model = scvi.model.SCVI.load_query_data(adata_, model_filename)
        # This allows for a simple forward pass
        model.is_trained = True
    return model.get_latent_representation()


def ot_mapping(xs, xt, mode="ot_bw"):
    """
    Compute optimal transport mapping between source and target distributions.

    Args:
        xs: numpy array, source distribution
        xt: numpy array, target distribution
        mode: string, optional, mode of optimal transport, default is 'ot_bw'
    Returns:
        xst: numpy array, optimal transport mapping from source to target distribution
    """
    if mode == "ot_bw":
        Ae, be = ot.gaussian.empirical_bures_wasserstein_mapping(xs, xt)
        xst = xs.dot(Ae) + be
    elif mode == "ot_gw":
        Ae, be = ot.gaussian.empirical_gaussian_gromov_wasserstein_mapping(xs, xt)
        xst = xs.dot(Ae) + be
    elif mode == "ot_emd":
        xst = ot.da.EMDTransport().fit_transform(Xs=xs, Xt=xt)
    return xst


def integrate(
    adata,
    integration_key,
    mode,
    ref=None,
    n_latent=10,
    compute_embeddings=["mde", "umap"],
    model_save_path=None,
):
    """Integrates the joint dataset according to integration_key using mode as integration method.

    Args:
        adata (anndata.AnnData):
            adata object with joint data
        integration_key (str):
            adata.obs column to use for integration
        mode (str): 'ot_bw','ot_gw','ot_emd','scvi', 'cxg_scvi', 'cxg_scvi_retrain', 'scanorama', 'harmony',
            Integration method to use
            obsm['X_pca'] is used as input to OT methods
            layers['counts'] is used as raw counts input to scvi models
        ref (str, optional):
            For OT based integration only, the dataset to use as the reference dataset. Defaults to None.
        compute_embeddings (bool, optional):
            Whether to compute UMAP and MDE on the integrated data. Defaults to True.

    Returns:
        new keys in adata.obs:
            LATENT_KEY = f"X_{mode}"
            MDE_KEY = f"{LATENT_KEY}_MDE"
            UMAP_KEY = f"{LATENT_KEY}_UMAP"
    """

    if "counts" in adata.layers:
        adata.layers["X"] = adata.X
        adata.X = adata.layers["counts"]
    else:
        print(
            'WARNING: adata.layers["counts"] does not exist. Using adata.X with assumption it contains raw counts'
        )

    LATENT_KEY = f"X_{mode}"
    MDE_KEY = f"{LATENT_KEY}_mde"
    UMAP_KEY = f"{LATENT_KEY}_umap"

    if mode in ["ot_bw", "ot_gw", "ot_emd"]:
        if ref is None:
            ref = adata.obs[integration_key].value_counts().idxmax()
            print(
                "ref parameter is None. Using dataset with most cells as reference:",
                ref,
            )
        ref_idx = adata.obs[integration_key] == ref

        adata.obsm[LATENT_KEY] = adata.obsm["X_pca"].copy()
        for sample in adata.obs[integration_key].unique():
            if sample != ref:
                xs_idx = adata.obs[integration_key] == sample
                adata.obsm[LATENT_KEY][xs_idx] = ot_mapping(
                    adata.obsm[LATENT_KEY][xs_idx],
                    adata.obsm["X_pca"][ref_idx],
                    mode=mode,
                )

    elif mode == "scanorama":
        sc.external.pp.scanorama_integrate(adata, key=integration_key, basis="X_pca")
    elif mode == "harmony":
        sc.external.pp.harmony_integrate(
            adata, key=integration_key, basis="X_pca", adjusted_basis=LATENT_KEY
        )

    elif mode == "scvi":
        sca.models.SCVI.setup_anndata(adata, batch_key=integration_key)
        model = sca.models.SCVI(
            adata, n_hidden=128, n_layers=2, n_latent=n_latent, gene_likelihood="nb"
        )
        model.train()
        if model_save_path is not None:
            model.save(model_save_path, overwrite=True)
        adata.obsm[LATENT_KEY] = model.get_latent_representation()

    elif mode == "trvae":
        model = sca.models.TRVAE(
            adata=adata,
            condition_key=integration_key,
            conditions=adata.obs[integration_key].unique().tolist(),
            latent_dim=n_latent,
            hidden_layer_sizes=[128, 128],
            recon_loss="nb",
        )
        model.train()
        if model_save_path is not None:
            model.save(model_save_path, overwrite=True)
        adata.obsm[LATENT_KEY] = model.get_latent()

    elif mode == "cxg_scvi":
        adata.obsm[LATENT_KEY] = embed_adata_cellxgene_scvi(
            adata, cxg_scvi_retrain=False
        )

    elif mode == "cxg_scvi_retrain":
        adata.obsm[LATENT_KEY] = embed_adata_cellxgene_scvi(
            adata, cxg_scvi_retrain=True
        )

    if "mde" in compute_embeddings:
        adata.obsm[MDE_KEY] = scvi.model.utils.mde(adata.obsm[LATENT_KEY])
    if "umap" in compute_embeddings:
        rsc.get.anndata_to_GPU(adata)
        rsc.pp.neighbors(adata, metric="cosine", use_rep=LATENT_KEY)
        adata.obsm[UMAP_KEY] = rsc.tl.umap(adata, copy=True, min_dist=0.2).obsm[
            "X_umap"
        ]
        rsc.get.anndata_to_CPU(adata)

    if "counts" in adata.layers:
        adata.X = adata.layers["X"]
        del adata.layers["X"]


def sca_transfer_labels(ad_ref, ad_query, latent_key, label_key, n_neighbors=30):
    """
    Transfer labels from a reference dataset to a query dataset using scanvi weighted k-nearest neighbors (KNN) utility function.

    Parameters:
        ad_ref (AnnData): The reference dataset.
        ad_query (AnnData): The query dataset.
        latent_key (str): The location of the joint embedding.
        n_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 30.

    Returns:
        tuple: A tuple containing the transferred labels and their uncertainties.
    """
    knn_transformer = sca.utils.knn.weighted_knn_trainer(
        train_adata=ad_ref,
        train_adata_emb=latent_key,  # location of our joint embedding
        n_neighbors=n_neighbors,
    )

    labels, uncert = sca.utils.knn.weighted_knn_transfer(
        query_adata=ad_query,
        query_adata_emb=latent_key,  # location of our embedding, query_adata.X in this case
        label_keys=label_key,  # (start of) obs column name(s) for which to transfer labels
        knn_model=knn_transformer,
        ref_adata_obs=ad_ref.obs,
    )
    return labels, uncert
