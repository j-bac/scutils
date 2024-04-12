import numpy as np
import scipy.sparse
import scipy.stats
import scanpy as sc
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def annotate_adatas(
    names,
    adatas,
    detailed_markers,
    permutations=100,
    percentile=99,
    ct_key="cell_type",
    symbol_key="gene_symbol",
):
    cell_types = detailed_markers[ct_key].unique()

    for name, _anndata in zip(names, adatas):
        print(name)

        # ---cells x markers boolean matrix
        gs = sc.AnnData(np.zeros((_anndata.shape[1], len(cell_types))))
        gs.obs_names = _anndata.var_names
        gs.var_names = cell_types
        for ct in cell_types:
            found_markers = np.isin(
                gs.obs_names,
                detailed_markers[detailed_markers[ct_key] == ct][symbol_key],
            )
            gs[found_markers, ct] = 1

        # ---gene score df from gs boolean matrix
        detailed_gene_set_scores_df = pd.DataFrame(index=_anndata.obs.index)
        for j in range(gs.shape[1]):
            gene_set_name = str(gs.var.index.values[j])
            result = score_gene_sets(
                ds=_anndata,
                gs=gs[:, [j]],
                permutations=permutations,
                method="mean_z_score",
            )
            detailed_gene_set_scores_df[gene_set_name + "_score"] = result["score"]
            if permutations > 0:
                detailed_gene_set_scores_df[gene_set_name + "_p_value"] = result[
                    "p_value"
                ]
                detailed_gene_set_scores_df[gene_set_name + "_fdr_bh"] = result["fdr"]
                detailed_gene_set_scores_df[gene_set_name + "_k"] = result["k"]

        detailed_gene_set_scores_adata = sc.AnnData(
            X=detailed_gene_set_scores_df.values,
            obs=_anndata.obs,
            var=pd.DataFrame(index=detailed_gene_set_scores_df.columns),
        )
        column_filter = list(
            filter(lambda x: "_score" in x, detailed_gene_set_scores_adata.var_names)
        )
        _anndata.uns["detailed_gene_score_df"] = detailed_gene_set_scores_df[
            column_filter
        ]

        # ---gene scores to annotation from max gene score
        _anndata.obs["cell_ids_from_max"] = _anndata.uns[
            "detailed_gene_score_df"
        ].idxmax(
            axis=1
        )  # .map(lambda s: s[:-6])

        # ---gene scores to annotation from 99% percentile and significant p value
        cell_set_name_to_ids = {}
        gs_adata = detailed_gene_set_scores_adata[:, column_filter]
        percentiles = np.percentile(gs_adata.X, axis=0, q=percentile)

        _anndata.uns["cell_ids"] = detailed_gene_set_scores_df[column_filter].copy()
        _anndata.uns["cell_ids"][:] = False
        if len(percentiles.shape) == 0:
            percentiles = [percentiles]

        for j in range(gs_adata.shape[1]):  # each gene set
            x = gs_adata[:, j].X
            selected = x >= percentiles[j]
            if permutations > 0:
                selected_pval = (
                    detailed_gene_set_scores_adata[
                        :, gs_adata.var_names[j][:-6] + "_p_value"
                    ].X
                    < 0.05
                )
                cell_ids = gs_adata[selected & selected_pval].obs.index
            else:
                cell_ids = gs_adata[selected].obs.index
            if len(cell_ids) > 0:
                cell_set_name_to_ids[gs_adata.var.index[j]] = cell_ids
                _anndata.uns["cell_ids"].loc[
                    selected.flat, gs_adata.var.index[j]
                ] = True
        _anndata.uns["cell_ids"].columns = gs.var.index

        idx_dup = _anndata.uns["cell_ids"].sum(axis=1) > 1
        idx_nan = _anndata.uns["cell_ids"].sum(axis=1) == 0
        _anndata.obs["cell_ids_from_quantile"] = _anndata.uns["cell_ids"].idxmax(axis=1)
        _anndata.obs["cell_ids_from_quantile"][idx_dup] = _anndata.obs[
            "cell_ids_from_max"
        ][idx_dup]
        _anndata.obs["cell_ids_from_quantile"][idx_nan] = "undefined"


def _ecdf(x):
    """no frills empirical cdf used in fdrcorrection"""
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


# from http://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
def fdr(pvals, is_sorted=False, method="indep"):
    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    if method in ["i", "indep", "p", "poscorr"]:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ["n", "negcorr"]:
        cm = np.sum(1.0 / np.arange(1, len(pvals_sorted) + 1))  # corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
    ##    elif method in ['n', 'negcorr']:
    ##        cm = np.sum(np.arange(len(pvals)))
    ##        ecdffactor = ecdf(pvals_sorted)/cm
    else:
        raise ValueError("only indep and negcorr implemented")

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected > 1] = 1
    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        return pvals_corrected_
    else:
        return pvals_corrected


def get_p_value_ci(n, n_s, z):
    # smooth
    n = n + 2
    n_s = n_s + 1
    n_f = n - n_s
    ci = (z / n) * np.sqrt((n_s * n_f) / n)
    return ci


def score_gene_sets(
    ds,
    gs,
    method="mean_z_score",
    max_z_score=5,
    permutations=None,
    random_state=0,
    smooth_p_values=True,
):
    """Score gene sets.

    Note that datasets and gene sets must be aligned prior to invoking this method. No check is done.

    mean_z_score: Compute the z-score for each gene in the set. Truncate these z-scores at 5 or -5,
    and define the signature of the cell to be the mean z-score over all genes in the gene set.

    Parameters
    ----------

    random_state : `int`, optional (default: 0)
        The random seed for sampling.

    Returns
    -------
    Observed scores and permuted p-values if permutations > 0

    """

    if permutations is None:
        permutations = 0
    x = ds.X
    gs_1_0 = gs.X
    if not scipy.sparse.issparse(gs.X) and len(gs.X.shape) == 1:
        gs_1_0 = np.array([gs_1_0]).T

    if not scipy.sparse.issparse(gs_1_0):
        gs_1_0 = scipy.sparse.csr_matrix(gs_1_0)

    gs_indices = gs_1_0 > 0
    if hasattr(gs_indices, "toarray"):
        gs_indices = gs_indices.toarray()
    gs_indices = gs_indices.flatten()

    if len(x.shape) == 1:
        x = np.array([x])
    # preprocess the dataset
    if method == "mean_z_score":
        x = x[:, gs_indices]  # only include genes in gene set
        if scipy.sparse.issparse(x):
            x = x.toarray()
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        std = np.sqrt(var)
        # std[std == 0] = 1e-12 # avoid divide by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (x - mean) / std
        x[np.isnan(x)] = 0
        x[x < -max_z_score] = -max_z_score
        x[x > max_z_score] = max_z_score
    elif method == "mean_rank":  # need all genes for ranking
        ranks = np.zeros(x.shape)
        is_sparse = scipy.sparse.issparse(x)
        for i in range(x.shape[0]):  # rank each cell separately
            row = x[i, :]
            if is_sparse:
                row = row.toarray()
            ranks[i] = scipy.stats.rankdata(row, method="min")
        x = ranks
        x = x[:, gs_indices]
    else:
        x = x[:, gs_indices]
        if scipy.sparse.issparse(x):
            x = x.toarray()
    observed_scores = x.mean(axis=1)
    if hasattr(observed_scores, "toarray"):
        observed_scores = observed_scores.toarray()

    # gene sets has genes on rows, sets on columns
    # ds has cells on rows, genes on columns
    # scores contains cells on rows, gene sets on columns

    if permutations is not None and permutations > 0:
        if random_state:
            np.random.seed(random_state)
        p_values = np.zeros(x.shape[0])
        permuted_X = x.T.copy()  # put genes on rows to shuffle each row indendently
        for i in range(permutations):
            for _x in permuted_X:
                np.random.shuffle(_x)
            # count number of times permuted score is >= than observed score
            p_values += permuted_X.mean(axis=0) >= observed_scores

        k = p_values
        if smooth_p_values:
            p_values = (p_values + 1) / (permutations + 2)
        else:
            p_values = p_values / permutations
        return {
            "score": observed_scores,
            "p_value": p_values,
            "fdr": fdr(p_values),
            "k": k,
        }

    return {"score": observed_scores}


def dict_to_newick(tree, name):
    """
    Recursively converts a dictionary representation of a tree into Newick format.

    Parameters:
    - tree (dict): A dictionary representing the tree structure.
    - name (str): The name of the current node in the tree.

    Returns:
    - str: The Newick format representation of the tree rooted at the specified node.
    """

    if len(tree[name]) == 0:
        return f"{name}"

    else:
        child_strings = [dict_to_newick(tree[name], child) for child in tree[name]]
        return f'({", ".join(child_strings)}){name}'


def dataframe_to_nested_dicts(df):
    """
    Generate nested dictionaries from a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to convert to nested dictionaries.

    Returns:
    dict: A nested dictionary structure created from the DataFrame.
    """
    nested_dict = {}
    for _, row in df.iterrows():
        current_dict = nested_dict
        for level in df.columns:
            value = row[level]

            if value not in current_dict and not pd.isna(value):
                current_dict[value] = {}
            if pd.isna(value):
                break
            else:
                current_dict = current_dict[value]
    return nested_dict


def prepare_adata_scHPL(adata, ann_key_prefix="ann_level_", n_levels=5):
    """
    Prepare the adata object for scHPL by processing the annotation keys
    and creating a tree (nested dictionary of unique levels).

    Parameters:
    - adata: Anndata object
    - ann_key_prefix: Prefix for the annotation keys (default: 'ann_level_')
    - n_levels: Number of annotation levels (default: 5)

    Returns:
    - updated adata.obs ann_keys
    - dict_unique_levels: Nested dictionary of unique levels
    """

    assert 'root' not in adata.obs[f"{ann_key_prefix}1"].values,  \
        "root category name found in values is reserved for the tree root. Please rename this category"

    # prepare anndata.obs
    ann_keys = [f"{ann_key_prefix}{i}" for i in range(1, n_levels + 1)]
    for k in ann_keys:
        adata.obs[k] = adata.obs[k].pipe(np.array)
    adata.obs.loc[
        adata.obs.query(f"{ann_key_prefix}1.isna()").index, f"{ann_key_prefix}1"
    ] = "root"

    # convert to nested dict
    df_unique_levels = adata.obs[ann_keys].drop_duplicates()
    dict_unique_levels = dataframe_to_nested_dicts(df_unique_levels)
    dict_unique_levels = {"root": dict_unique_levels}
    # rm artificial root key due to filling nan values if present
    if 'root' in dict_unique_levels['root']:
        del dict_unique_levels['root']['root']

    # fill anndata.obs missing values by propageting coarse labels
    for i in range(1, n_levels):
        k, km1 = ann_keys[i], ann_keys[i - 1]
        idx_ = adata.obs.query(f"{k}.isna()").index
        adata.obs.loc[idx_, k] = adata.obs.loc[idx_][km1].values

    return dict_unique_levels



def get_depth_first_order(tree):
    """
    Get the depth-first order of a nested dictionary tree.

    Parameters:
    - tree: Nested dictionary tree

    Returns:
    - pd.DataFrame: DataFrame with 'name' and 'depth' columns
    """

    # Initialize list to store labels and their depth
    label_order = []

    # Initialize depth to -1
    depth = -1

    def depth_first_order(node, label_order, depth):
        """
        Recursively traverse the tree and append the node name and its depth to the list.

        Parameters:
        - node: Current node in the tree
        - label_order: List to store labels and their depth
        - depth: Current depth of the tree
        """

        # Increase depth
        depth += 1

        # Append node name and depth to the list
        label_order.append({'name': node.name[0], 'depth': depth})

        # Recursively traverse the child nodes
        for child in node.descendants:
            depth_first_order(child, label_order, depth)

        # Decrease depth
        depth -= 1

    # Start the traversal from the first node
    depth_first_order(tree[0], label_order, depth)

    # Convert the list to a DataFrame and reset the index
    return pd.DataFrame(label_order).reset_index().rename(columns={'index': 'order'})

def confusion_matrix(true_labels, pred_labels):
    '''Construct a confusion matrix (taken from scArches).
    
        Parameters
        ----------
        true_labels: array_like 
            True labels of the dataset
        pred_labels: array_like
            Predicted labels
            
        Returns
        -------
        conf: confusion matrix
    '''
    
    true_labels = pd.DataFrame(true_labels).reset_index(drop=True)
    pred_labels = pd.DataFrame(pred_labels).reset_index(drop=True)
    yall = pd.concat([true_labels, pred_labels], axis=1)
    yall.columns = ['ytrue', 'ypred']
    conf = pd.crosstab(yall['ytrue'], yall['ypred'])

    return conf

def heatmap(true_labels, 
            pred_labels, 
            order_rows: list = None, 
            order_cols: list = None, 
            transpose: bool = False, 
            cmap: str = 'Reds', 
            title: str = None, 
            xlabel: str = 'Predicted labels', 
            ylabel: str = 'True labels', 
            row_mode: bool = 'mask',
            **kwargs):
    """
    Generate a heatmap of the confusion mat
    Parameters
    ----------
    true_labels : array_like
        True labels of the dataset
    pred_labels : array_like
        Predicted labels
    order_rows : list, optional
        Order of rows in the heatmap
    order_cols : list, optional
        Order of columns in the heatmap
    transpose : bool, optional
        Transpose the confusion matrix
    cmap : str, optional
        Colormap for the heatmap
    title : str, optional
        Title of the heatmap
    annot : bool, optional
        Annotate the heatmap with values
    xlabel : str, optional
        Label for the x-axis
    ylabel : str, optional
        Label for the y-axis
    row_mode : {'mask', 'delete'}, optional
        Method to handle rows with zero values ('mask' or 'delete')
    kwargs : dict
        Additional keyword arguments for the heatm
    Returns
    -------
    p : matplotlib.axes._subplots.AxesSubplot
        Heatmap object
    conf : pandas.DataFrame
        Confusion matrix
    conf2 : pandas.DataFrame
        Normalized confusion matrix
    """
    #Get confusion matrix & normalize
    conf = confusion_matrix(true_labels, pred_labels) 

    if transpose:
        conf = np.transpose(conf)

    conf2 = np.divide(conf,np.sum(conf.values, axis = 1, keepdims=True))   

    if order_rows is None:
        num_rows = np.shape(conf2)[0]
        order_rows = np.linspace(0, num_rows-1, num=num_rows, dtype=int)
        order_rows = np.asarray(conf2.index)
    else:
        xx = np.setdiff1d(order_rows, conf2.index)
        test = pd.DataFrame(np.zeros((len(xx), np.shape(conf2)[1])), index = xx, columns=conf2.columns)
        conf2 = pd.concat([conf2,test], axis=0)    
    
    if order_cols is None:
        num_cols = np.shape(conf2)[1]
        order_cols = np.linspace(0, num_cols-1, num=num_cols, dtype=int)
        order_cols = np.asarray(conf2.columns)
    else:
        xx = np.setdiff1d(order_cols, conf2.columns)
        test = pd.DataFrame(np.zeros((np.shape(conf2)[0], len(xx))), index = conf2.index, columns=xx)
        conf2 = pd.concat([conf2,test], axis=1)    

    conf2 = conf2.loc[order_rows,order_cols]

    if row_mode == 'mask':
        mask=pd.DataFrame(np.zeros_like(conf2,dtype=bool),index=conf2.index,columns=conf2.columns)
        mask.loc[conf2[conf2.sum(1)==0].index] = True
        cmap = mpl.colormaps[cmap]
        cmap.set_bad('lightgrey',.3)
    elif row_mode == 'delete':
        conf = conf.loc[conf2[conf2.sum(1)!=0].index]
        conf2 = conf2.loc[conf2[conf2.sum(1)!=0].index]
        mask = None
    else:
        mask = None

    p=sns.heatmap(conf2, vmin = 0, vmax = 1,  mask=mask,
            xticklabels=True,yticklabels=True,
            cbar_kws={'label': 'Fraction'}, cmap=cmap, **kwargs)
    
    if title is not None:
        plt.title(title)
        
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize = 14)
    
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize = 14)
        
    return p, conf, conf2


def tree_heatmap(
            tree_scHPL,
            true_labels, 
            pred_labels, 
            order_rows: list = None, 
            order_cols: list = None, 
            transpose: bool = False, 
            cmap: str = 'Reds', 
            title: str = None, 
            xlabel: str = 'Predicted labels', 
            ylabel: str = 'True labels', 
            row_mode: bool = 'mask',
            xblocksize: float = None,
            yblocksize: float = None,
            edgecolor: str = 'white',
            ax = None,
            figsize = None,
            **kwargs):
    """Plot a heatmap with tree structured blocks.

    Args:
        tree_scHPL: scHPL Tree object representing the tree structure.
        true_labels: The true labels.
        pred_labels: The predicted labels.
        order_rows: The order of rows.
        order_cols: The order of columns.
        transpose: Transpose the heatmap.
        cmap: The color map.
        title: The title of the plot.
        annot: Annotate the heatmap.
        xlabel: The x label.
        ylabel: The y label.
        row_mode: The mode of rows.
        xblocksize: The size of x blocks.
        yblocksize: The size of y blocks.
        **kwargs: Additional arguments.
    """

    # Get the depth-first order of the tree and the colors for each depth
    label_order = get_depth_first_order(tree_scHPL)
    order_rows = label_order['name'].to_list() # + ['Unknown'],
    order_cols = label_order['name'].to_list() + ['Rejection (dist)']
    depths_colors = dict(zip(label_order['depth'].unique(), plt.get_cmap('magma', len(label_order['depth'].unique())).colors))
    max_depth = label_order['depth'].max()

    # Create a subplot if ax is None
    if ax is None:
        if figsize is None:
            figsize=(26, 4) if row_mode == 'delete' else (28, 25)
            fig = plt.figure(figsize=figsize)
        ax = plt.subplot()
    # Call the heatmap function
    p, conf, conf2 = heatmap(true_labels=true_labels, 
                              pred_labels=pred_labels, 
                              order_rows=order_rows, 
                              order_cols=order_cols, 
                              transpose=transpose, 
                              cmap=cmap, 
                              title=title, 
                              xlabel=xlabel, 
                              ylabel=ylabel, 
                              row_mode=row_mode,
                              ax=ax,
                              **kwargs)

    # Set the block size if not provided
    if xblocksize is None:
        xblocksize = 0.05 if row_mode == 'delete' else 0.01
    if yblocksize is None:
        yblocksize = 0.01

    # Add blocks for each label
    for i, tick_label in enumerate(p.axes.get_yticklabels()):
        txt = tick_label.get_text()
        depth = label_order['depth'][label_order['name'] == txt].values[0]
        for d in depths_colors.keys():
            if d != depth:
                ax.add_patch(plt.Rectangle(xy=(-yblocksize*(max_depth-d+1), i), width=yblocksize, height=1, color='white', lw=0, alpha=.0,
                                           transform=ax.get_yaxis_transform(), clip_on=False))
            else:
                ax.add_patch(plt.Rectangle(xy=(-yblocksize*(max_depth-d+1), i), width=yblocksize, height=1, color=depths_colors[depth], lw=0, alpha=.3,
                                           transform=ax.get_yaxis_transform(), clip_on=False))

    for i, tick_label in enumerate(p.axes.get_xticklabels()):
        txt = tick_label.get_text()
        if txt == 'Rejection (dist)':
            continue
        depth = label_order['depth'][label_order['name'] == txt].values[0]
        for d in depths_colors.keys():
            if d != depth:
                ax.add_patch(plt.Rectangle(xy=(i, -xblocksize*(max_depth-d+1)), width=1, height=xblocksize, color='white', lw=0,
                                           transform=ax.get_xaxis_transform(), clip_on=False))
            else:
                ax.add_patch(plt.Rectangle(xy=(i, -xblocksize*(max_depth-d+1)), width=1, height=xblocksize, color=depths_colors[depth], lw=0, alpha=.3,
                                           transform=ax.get_xaxis_transform(), clip_on=False))

    for i in label_order['order'].values:
        ax.axhline(i, color=edgecolor, lw=.75)
        ax.axvline(i, color=edgecolor, lw=.75)

    for i in label_order.query('depth == 1')['order'].values:
        if row_mode != 'delete':
            ax.axhline(i, color=depths_colors[1], lw=1.5, xmin=-.15, clip_on=False)
        ax.axvline(i, color=depths_colors[1], lw=1.5, ymin=-.15, clip_on=False)

    if row_mode != 'delete':
        ax.plot(ax.get_xlim(), ax.get_ylim()[::-1], color='black', linewidth=.5)

    ax.grid(False)
    return p, conf, conf2
