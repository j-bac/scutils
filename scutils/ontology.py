import networkx as nx
import time
import matplotlib.pyplot as plt
import pronto
import ccf_openapi_client
import pickle
import pandas as pd
import numpy as np
from ccf_openapi_client.api import default_api
from nxontology.imports import pronto_to_multidigraph, multidigraph_to_digraph


def all_single_source_shortest_paths(graph, source, targets=None):
    shortest_paths_dict = {}
    pred = nx.predecessor(graph, source)
    if targets is None:
        targets = graph.nodes
    for node in targets:
        shortest_paths_dict[node] = list(
            nx.algorithms.shortest_paths.generic._build_paths_from_predecessors(
                [source], node, pred
            )
        )
    return shortest_paths_dict


def all_single_source_shortest_paths_nodes(graph, source, targets=None):
    shortest_paths_dict = {}
    pred = nx.predecessor(graph, source)
    if targets is None:
        targets = graph.nodes
    for node in targets:
        shortest_paths_dict[node] = list(
            nx.algorithms.shortest_paths.generic._build_paths_from_predecessors(
                [source], node, pred
            )
        )
        shortest_paths_dict[node] = list(
            set(ct for path in shortest_paths_dict[node] for ct in path)
        )
    return shortest_paths_dict


def all_shortest_paths(graph, sources, targets=None):
    for source in sources:
        yield source, all_single_source_shortest_paths(graph, source, targets=targets)


def all_shortest_paths_nodes(graph, sources, targets=None):
    for source in sources:
        yield source, all_single_source_shortest_paths_nodes(
            graph, source, targets=targets
        )


def subset_graph(
    graph,
    nodes_of_interest,
    parent=True,
    parent_child=False,
    child=True,
    all_paths=True,
):
    """Take a subset of a directed graph based on a list of nodes

    Args:
        graph (networkx.DiGraph): Original directed graph.
        nodes_of_interest (list[str]): List of nodes.
        parent (bool, optional): Whether to add parents. Defaults to False.
        parent_child (bool, optional): Whether to add parents' children. Defaults to False.
        child (bool, optional): Whether to add children. Defaults to False.
        strict (bool, optional): Enforce children or parents to be in nodes_of_interest. Defaults to False.

    Returns:
        networkx.DiGraph: Subgraph containing relevant nodes and their edges.
    """

    if all_paths:
        all_paths_dict = dict(
            all_shortest_paths_nodes(
                graph.to_undirected(),
                sources=nodes_of_interest,
                targets=nodes_of_interest,
            )
        )
        all_paths_nodes = list(
            set(
                ct
                for paths in all_paths_dict.values()
                for path in paths.values()
                for ct in path
            )
        )
    else:
        all_paths_nodes = []

    # Create subgraph
    subgraph = graph.subgraph(list(nodes_of_interest) + all_paths_nodes).copy()

    # Add parents if specified
    if parent:
        parents = set()
        for node in nodes_of_interest:
            parents.update(graph.predecessors(node))
        subgraph.add_nodes_from(parents)

    # Add parent's children if specified
    if parent_child:
        parent_children = set()
        for parent in parents:
            parent_children.update(graph.successors(parent))
        subgraph.add_nodes_from(parent_children)

    # Add children if specified
    if child:
        children = set()
        for node in nodes_of_interest:
            children.update(graph.successors(node))
        subgraph.add_nodes_from(children)

    # Add edges to the subgraph
    if parent or parent_child or child:
        for node in subgraph.nodes():
            for successor in graph.successors(node):
                if successor in subgraph:
                    subgraph.add_edge(node, successor)

    return subgraph


def ontology_graph_obophenotype(
    cell_ontology_url="https://github.com/obophenotype/cell-ontology/releases/download/v2024-02-13/cl.owl",
):
    """Create a directed graph of the cell ontology

    Args:
        cell_ontology_url (str, optional): From 'https://github.com/obophenotype/cell-ontology/releases/'

    Returns:
        G: networkx DiGraph
    """
    G_pronto = pronto.Ontology(cell_ontology_url)
    G_multidigraph = pronto_to_multidigraph(G_pronto)
    G_digraph = multidigraph_to_digraph(
        G_multidigraph, reverse=True
    )  # reverse to edges go from superterm to subterm (whereas edges are of type 'is_a')
    return G_pronto, G_multidigraph, G_digraph


def ontology_graph_bionty(
    cell_types, add_parent=True, add_parent_child=False, add_child=True, strict=False
):
    """Create a directed graph of the cell ontology based on a list of bionty cell types

    Args:
        cell_types (list): list of bionty.CellType
        add_parent (bool, optional): whether to add parents. Defaults to True.
        add_parent_child (bool, optional): whether to add parents' children. Defaults to False.
        add_child (bool, optional): whether to add children. Defaults to True.
        strict (bool, optional): enforce children or parents to be in cell_types. Defaults to True.

    Returns:
        G: networkx DiGraph
    """

    # Create an empty graph
    G = nx.DiGraph()

    cell_types_names = [ct.name for ct in cell_types]
    # Add nodes of interest and their parent's children to the G
    for node in cell_types:
        # Add the node itself
        G.add_node(node.name)

        # Add the node's parent (if it exists) and its children
        for parent in node.parents.all():
            if add_parent or add_parent_child:
                G.add_node(parent.name)
            if add_parent:
                G.add_edge(parent.name, node.name)
            if add_parent_child:
                for parent_child in parent.children.all():
                    if strict and parent_child.name not in cell_types_names:
                        continue
                    else:
                        G.add_edge(parent.name, parent_child.name)

        # Add the node's children
        if add_child:
            for child in node.children.all():
                if strict and child.name not in cell_types_names:
                    continue
                else:
                    G.add_node(child.name)
                    G.add_edge(node.name, child.name)

    return G


def ontology_graph_bionty_all(save_path=None, source="laminlabs/cellxgene"):
    """
    Create a networkx graph representing the ontology using the bionty library.

    Args:
        save_path (str, optional): Path to save the graph. Defaults to None.
        source (str, optional): The ontology source. Defaults to "laminlabs/cellxgene".

    Returns:
        graph (nx.DiGraph): The ontology graph.
    """
    import bionty as bt

    # Create a new directed graph
    graph = nx.DiGraph()

    # Get the root cell type from the specified source
    root = bt.CellType.using(source).filter(ontology_id="CL:0000000").one()

    # Recursive function to traverse the ontology tree
    def add_nodes_and_edges(node):
        """
        Recursively adds nodes and edges to the graph representing the ontology tree.

        Args:
            node (bionty.CellType): The current node in the ontology tree.
        """
        for child in node.children.all():
            graph.add_node(child.name, info=child)
            graph.add_edge(node.name, child.name)
            add_nodes_and_edges(child)

    # Add root node to the graph
    graph.add_node(root.name, info=root)

    # Traverse the ontology tree and add nodes and edges
    add_nodes_and_edges(root)

    # Save the graph if a path is provided
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(graph, f)

    return graph


def ontology_tree_hra(save_path="../data/ontology/"):
    """
    Retrieves the HRA ontology tree from the API and saves it to a pickle file.

    Args:
        save_path (str): The path to save the pickle file. Defaults to '../data/ontology/'.

    Returns:
        dict: The HRA ontology tree.

    Raises:
        ccf_openapi_client.ApiException: If there is an exception when calling the API.
    """
    # Create API client configuration and instantiate API client
    configuration = ccf_openapi_client.Configuration(
        host="https://apps.humanatlas.io/hra-api/v1"
    )
    api_client = ccf_openapi_client.ApiClient(configuration)
    api_instance = default_api.DefaultApi(api_client)

    # Wait until the database is ready
    db_ready = False
    result = None
    while not db_ready:
        result = api_instance.db_status()
        if result["status"] == "Ready":
            db_ready = True
        else:
            print("Database not ready yet! Retrying...", result)
            time.sleep(2)
    print("Database ready!\n", result)

    try:
        # Retrieve the HRA ontology tree and save it to a pickle file
        api_response = api_instance.cell_type_tree_model(cache=False)
        with open(save_path + "/hra_ontology_tree.pkl", "wb") as f:
            pickle.dump(api_response, f)
        return api_response
    except ccf_openapi_client.ApiException as e:
        # Print the exception message if there is an exception
        print("Exception when calling DefaultApi->aggregate_results: %s\n" % e)


def ontology_tree_hra_to_networkx_digraph(ontology_tree):
    """
    Convert the HRA ontology tree to a networkx DiGraph.

    Args:
        ontology_tree (dict): The HRA ontology tree.

    Returns:
        graph (nx.DiGraph): The networkx DiGraph representing the ontology tree.
    """
    # Create a new directed graph
    graph = nx.DiGraph()

    # Extract the root node from the ontology tree
    root = ontology_tree["root"]

    # Convert the ontology tree to a dictionary
    ontology_tree = ontology_tree.to_dict()["nodes"]

    # Recursive function to traverse the ontology tree
    def add_nodes_and_edges(node, tree):
        """
        Recursively add nodes and edges to the graph.

        Args:
            node (dict): The current node in the ontology tree.
            tree (dict): The ontology tree.
        """
        if "children" in node:
            for child_ in node["children"]:
                child = tree[child_]
                graph.add_edge(node["id"], child["id"])
                add_nodes_and_edges(child, tree)

    # Add the root node to the graph
    graph.add_node(root)

    # Traverse the ontology tree and add nodes and edges
    add_nodes_and_edges(ontology_tree[root], ontology_tree)

    return graph


def plot_graph_with_highlighted_nodes(G, selected_nodes, figsize=(15, 40), font_size=5):
    """Plot a graph with highlighted nodes

    Args:
        G: networkx DiGraph
        selected_nodes (list[str]): list of nodes in G

    Returns:
        pos, pos_text: position of nodes and text
    """

    plt.figure(figsize=figsize)
    # Create a layout for the graph
    pos_ = nx.nx_agraph.graphviz_layout(G, prog="dot")

    pos = {}
    pos_text = {}
    for k in pos_.keys():
        pos[k] = (-pos_[k][1], pos_[k][0])
        pos_text[k] = (-pos_[k][1], pos_[k][0] + 10)

    # Draw the G
    nx.draw(G, pos, with_labels=False, node_color="lightblue", node_size=50)

    # Draw node labels
    nx.draw_networkx_labels(
        G,
        pos_text,
        labels={node: node for node in G.nodes},
        font_size=font_size,
        font_color="black",
        font_weight="bold",
        verticalalignment="bottom",
    )

    # Highlight selected nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=selected_nodes, node_color="red", node_size=100
    )

    # Show the plot
    plt.show()
    return pos, pos_text

def ontology_paths_from_digraph(G,root='cell'):
    """Convert nx.DiGraph of cell types to dataframe of paths to each leaf
    Args:
        G (nx.DiGraph )

    Returns:
        df_leaves_paths (pd.DataFrame): leaf x max paths depth table
        cell_types_depth
        cell_types_paths
    """    
    leaves = [n for n,d in G.out_degree if d==0]
    target = list(G.nodes)
    target.remove(root)
    list(nx.all_simple_paths(G, source=root,target=target))
    cell_types_shortest_paths = nx.shortest_path(G, source=root)
    cell_types_all_paths = list(nx.all_simple_paths(G, source=root,target=target))
    df_leaves_shortest_paths = pd.DataFrame({k:pd.Series(v) for k,v in cell_types_shortest_paths.items() if k in leaves}).T
    df_leaves_all_paths = pd.DataFrame({v[-1]+'_'+str(i):pd.Series(v) for i,v in enumerate(cell_types_all_paths)}).T
    return df_leaves_shortest_paths, df_leaves_all_paths, cell_types_shortest_paths, cell_types_all_paths

def ontology_df_from_cell_annotation(cell_types_all_paths,cell_types):
    cell_types_u = np.unique(cell_types)
    longest_path = lambda node, paths: max([p for p in paths if p[-1]==node],key=len)
    shortest_path = lambda node, paths: min([p for p in paths if p[-1]==node],key=len)

    df_ct_longest = pd.DataFrame({ct: pd.Series(longest_path(ct,cell_types_all_paths)) for ct in cell_types_u}).T
    df_ct_shortest = pd.DataFrame({ct: pd.Series(shortest_path(ct,cell_types_all_paths)) for ct in cell_types_u}).T
    return df_ct_longest, df_ct_shortest