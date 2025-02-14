# raptor_recursive.py
"""
This module implements the core RAPTOR recursive summarization and clustering process.
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds a hierarchical tree
of summaries from raw texts. It performs:
  1. Text embedding.
  2. Global clustering via UMAP and GMM.
  3. Local clustering for fine-grained grouping.
  4. LLM-based summarization of each cluster.
  5. Recursion to generate higher-level abstractions.
"""

from typing import List, Tuple, Dict, Union
import random
import numpy as np
import pandas as pd
import umap
from sklearn.mixture import GaussianMixture

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_random_seed() -> int:
    """
    Get a random seed for algorithm that requires it.
    If you need to reproduce the same results, you can set the seed to a fixed value.
    """
    return random.randint(0, 2**32 - 1)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dimension: int,
    n_neighbors: Union[int, None] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Globally reduce the dimensionality of embeddings using UMAP.
    Further clustering will be performed on the reduced embeddings.

    UMAP is applied to the entire set of embeddings to capture their overall structure, making the clustering process computationally efficient and potentially revealing global patterns.

    Args:
        embeddings (np.ndarray): The high-dimensional embeddings to cluster.
        dimension (int): The dimension to reduce to.
        n_neighbors (Union[int, None]): The number of neighbors to consider for UMAP. If None, n_neighbors value will be set automatically.
        metric (str): The metric to use for UMAP.

    Returns:
        np.ndarray: The embedding object with reduced dimensions.
    """
    if n_neighbors is None:
        # Calculate the number of neighbors based on the number of embeddings
        n_neighbors = int((len(embeddings) - 1) ** 0.5)

    dimensionality_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dimension,
        metric=metric,
        random_state=get_random_seed(),
    )

    result = dimensionality_reducer.fit_transform(embeddings)
    assert isinstance(result, np.ndarray)  # Just to pass the type checker

    return result


def local_cluster_embeddings(
    embeddings: np.ndarray,
    dimension: int,
    num_neighbors: int = 10,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Locally reduce dimensionality using UAMP for a subset of embeddings.
    This "zoom in" helps reveal finer-grained cluster structure within a global cluster.

    It's similar to global_cluster_embeddings, but focuses on a subset of embeddings.
    Thus, is's helpful to analyze the structure of a specific cluster(fine-grained).

    Args:
        embeddings (np.ndarray): The high-dimensional embeddings to cluster.
        dimension (int): The dimension to reduce to.
        num_neighbors (int): The number of neighbors to consider for UMAP.
        metric (str): The metric to use for UMAP.

    Returns:
        np.ndarray: The embedding object with reduced dimensions.
    """
    dimensionality_reducer = umap.UMAP(
        n_neighbors=num_neighbors,
        n_components=dimension,
        metric=metric,
        random_state=get_random_seed(),
    )

    result = dimensionality_reducer.fit_transform(embeddings)
    assert isinstance(result, np.ndarray)  # Just to pass the type checker

    return result


def get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = get_random_seed(),
) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC)
    on a Gaussian Mixture Model (GMM).

    BIC is a criterion for model selection among a finite set of models. It balances the model's complexity and goodness of fit(accuracy).

    GMM is a probabilistic model that assumes all data points are generated from a mixture of several Gaussian distributions with unknown parameters.

    Args:
        embeddings (np.ndarray): The embeddings to cluster.
        max_clusters (int): The maximum number of clusters to consider.
        random_state (int): The random seed for reproducibility.

    Returns:
        int: The optimal number of clusters.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters_range = np.arange(1, max_clusters)
    bics = []  # To store BIC values for each number of clusters

    for n_clusters in n_clusters_range:
        # Create a GMM(Gaussian Mixture Model) with n_clusters components
        gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gmm.fit(embeddings)
        bics.append(gmm.bic(embeddings))

    optimal_number_of_clusters = n_clusters_range[np.argmin(bics)]
    return optimal_number_of_clusters


def GMM_cluster(
    embeddings: np.ndarray,
    threshold: float,
    random_state: int = get_random_seed(),
) -> Tuple[List[np.ndarray], int]:
    """
    Cluster the embeddings using a Gaussian Mixture Model (GMM) and return the cluster centers.

    Args:
        embeddings (np.ndarray): The embeddings to cluster.
        threshold (float): Minimum probability threshold to assign a point to a cluster.
        random_state (int): The random seed.

    Returns:
        Tuple[List[np.ndarray], int]: A list of arrays (each containing the indices where the membership probability exceeds the threshold) and the number of clusters.
    """
    # Find out the optimal number of clusters using BIC
    n_clusters = get_optimal_clusters(embeddings, random_state=random_state)

    # Learn the GMM model with the optimal number of clusters
    gaussian_mixture = GaussianMixture(
        n_components=n_clusters, random_state=random_state
    )
    gaussian_mixture.fit(embeddings)

    # Calculate the probabilities of each point belonging to each cluster
    probabilities = gaussian_mixture.predict_proba(embeddings)

    # If the probability is greater than the threshold, assign the point to the cluster
    labels = [np.where(probability > threshold)[0] for probability in probabilities]

    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dimension: int, threshold: float
) -> List[np.ndarray]:
    """
    Perform hierarchical clustering on embeddings using UMAP and GMM.
    This function carries out the pipeline of global clustering and local clustering.

    First, it conducts global clustering to group similar embeddings.
    Then, it performs local clustering on each global cluster to reveal finer-grained structure.

    Args:
        embeddings (np.ndarray): The embeddings to cluster.
        dimension (int): The dimension to reduce to.
        threshold (float): Minimum probability threshold to assign a point to a cluster.

    Returns:
        List[np.ndarray]: A list of arrays containing the indices of embeddings in each cluster.
    """
    if len(embeddings) <= dimension + 1:
        # If the data is insufficient, avoid clustering and return each point as a cluster
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction and clustering
    reduced_global_embedding = global_cluster_embeddings(
        embeddings=embeddings, dimension=dimension, metric="cosine"
    )
    global_clusters, n_global_clusters = GMM_cluster(
        embeddings=reduced_global_embedding, threshold=threshold
    )

    # Prepare(initialize) container for local clusters
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # For each global cluster, perform local clustering
    for global_cluster_index in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        indices = [
            index
            for index, gc in enumerate(global_clusters)
            if global_cluster_index in gc
        ]
        if not indices:
            continue
        cluster_embeddings = embeddings[indices]

        if len(cluster_embeddings) <= dimension + 1:
            # For small clusters, avoid local clustering and return each point as a cluster
            local_clusters = [np.array([0]) for _ in cluster_embeddings]
            n_local_clusters = 1
        else:
            # For larger clusters, perform local dimensionality reduction and clustering
            reduced_local_embedding = local_cluster_embeddings(
                embeddings=cluster_embeddings, dimension=dimension, metric="cosine"
            )
            local_clusters, n_local_clusters = GMM_cluster(
                embeddings=reduced_local_embedding, threshold=threshold
            )

        # Map local cluster indices to global cluster indices
        for local_cluster_index in range(n_local_clusters):
            # Collect indices of embeddings in the local cluster
            local_indices = [
                indices[k]
                for k, lc in enumerate(local_clusters)
                if local_cluster_index in lc
            ]
            # For each local cluster, append the global cluster index
            for index in local_indices:
                all_local_clusters[index] = np.append(
                    all_local_clusters[index], index + total_clusters
                )
        total_clusters += n_local_clusters

    return all_local_clusters


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert a list of texts into embeddings using LLM model.

    Args:
        texts (List[str]): The texts to embed.

    Returns:
        np.ndarray: The embeddings of the texts.
    """
    embedding_model = OpenAIEmbeddings()
    embeddings = embedding_model.embed_documents(texts)

    return np.array(embeddings)


def embed_cluster_texts(texts: List[str]) -> pd.DataFrame:
    """
    Embed texts and perform clustering on the embeddings.
    Returns a DataFrame with columns: text, embedding, and cluster assignment.
    (Combines embedding and clustering into a single process.)

    Args:
        texts (List[str]): The texts to embed.

    Returns:
        np.ndarray: The embeddings of the texts.
    """
    embeddings_np = embed_texts(texts)
    cluster_labels = perform_clustering(
        embeddings=embeddings_np, dimension=10, threshold=0.1
    )

    df = pd.DataFrame()
    df["text"] = texts
    df["embedding"] = list(embeddings_np)
    df["cluster"] = cluster_labels

    return df


def format_text(df: pd.DataFrame) -> str:
    """
    Format texts from a DataFrame into a single string for summarization.

    Args:
        df (pd.DataFrame): DataFrame with a "text" column.

    Returns:
        str: Concatenated text separated by a delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


def embed_cluster_summarize_texts(
    texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embed, cluster, and then summarize texts within each cluster.
    For each cluster, the texts are concatenated and summarized using an LLM.
    (Integrates embedding, clustering, and summarization into a single process.)

    Args:
        texts (List[str]): List of texts to process.
        level (int): Current recursion level (for tracking purposes).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
          - df_clusters: DataFrame with original texts, embeddings, and cluster assignments.
          - df_summary: DataFrame with cluster summaries, level info, and cluster IDs.
    """
    df_clusters = embed_cluster_texts(texts)

    # Expand each row into multiple rows for each cluster assignment.
    expanded_list = []
    for _, row in df_clusters.iterrows():
        for cluster_id in row["cluster"]:
            # Print the current progress
            print(f"[Level {level}] Processing cluster #{int(cluster_id)}...")

            # Expand the row with the cluster assignment
            expanded_list.append(
                {
                    "text": row["text"],
                    "embedding": row["embedding"],
                    "cluster": cluster_id,
                }
            )

    expanded_df = pd.DataFrame(expanded_list)
    all_cluster_ids = expanded_df["cluster"].unique()

    # Prepare the LLM for summarization.
    template = """
    Here is a subset of the LangChain Expression Language documentation.
    The LangChain Expression Language provides a way to organize chains in LangChain.
    Please provide a detailed summary of the provided documentation.

    Documents:
    {context}
    """

    # Build the prompt chain
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm | StrOutputParser()

    # Summarize each cluster and store the summaries
    summaries: List[str] = []
    for cluster_id in all_cluster_ids:
        # Print the current progress
        print(f"[Level {level}] Summarizing cluster #{int(cluster_id)}...")

        # Extract texts for the current cluster
        df_cluster = expanded_df[expanded_df["cluster"] == cluster_id]
        assert isinstance(df_cluster, pd.DataFrame)  # Just to pass the type checker
        formatted_text = format_text(df_cluster)
        summary = chain.invoke({"context": formatted_text})
        summaries.append(summary)

    # For each cluster, create a summary DataFrame
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_cluster_ids),
        }
    )

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively perform embedding, clustering, and summarization.

    At each recursion level, the summaries from the previous level become the new input texts.
    This allows the model to build an abstraction tree of the document's content.

    Args:
        texts (List[str]): Input texts for the current recursion level.
        level (int): Current recursion level.
        n_levels (int): Maximum recursion depth.

    Returns:
        Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]: A dictionary mapping each level to a tuple of DataFrames
        (df_clusters, df_summary).
    """
    # Print current status.
    print(f"\n[Level {level}] Processing {len(texts)} texts.")
    results: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    # Perform embedding, clustering, and summarization at this level.
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)
    results[level] = (df_clusters, df_summary)

    # Continue recursion if there is more than one cluster and we haven't reached the max level.
    if level < n_levels and df_summary["cluster"].nunique() > 1:
        next_texts = df_summary["summaries"].tolist()
        print(
            f"[Level {level}] Recursing to level {level+1} with {len(next_texts)} summaries."
        )
        next_level_results = recursive_embed_cluster_summarize(
            texts=next_texts, level=level + 1, n_levels=n_levels
        )
        results.update(next_level_results)
    else:
        print(
            f"[Level {level}] Recursion terminated (either reached max levels or only one cluster exists)."
        )

    return results
