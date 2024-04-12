def umap_gpu(adata, use_rep="X_pca", n_neighbors=15, metric="euclidean", min_dist=0.5):
    import rapids_singlecell as rsc

    rsc.utils.anndata_to_GPU(adata)
    rsc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, metric=metric)
    adata.obsm[f"{use_rep}_umap"] = rsc.tl.umap(
        adata, copy=True, min_dist=min_dist
    ).obsm["X_umap"]
    rsc.utils.anndata_to_CPU(adata)


def mde_gpu(adata, use_rep="X_pca",):
    import scvi

    adata.obsm[f'{use_rep}_mde'] = scvi.model.utils.mde(adata.obsm[use_rep])
