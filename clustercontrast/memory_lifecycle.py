from __future__ import absolute_import


def consume_precomputed_ephemeral(distance_builder, consumer):
    """Consume one distance matrix without retaining it in the caller scope."""
    distances = distance_builder()
    try:
        return consumer(distances)
    finally:
        del distances


def cluster_precomputed_ephemeral(distance_builder, clusterer, name,
                                  require_nonempty=False):
    """Build, consume, and release exactly one precomputed distance matrix."""
    labels = consume_precomputed_ephemeral(
        distance_builder, clusterer.fit_predict)
    num_clusters = len(set(int(label) for label in labels)) - (
        1 if any(int(label) == -1 for label in labels) else 0)
    if require_nonempty and num_clusters == 0:
        eps = getattr(clusterer, "eps", "unknown")
        raise RuntimeError(
            "{} clustering produced zero clusters at fixed eps={}".format(name, eps))
    return labels, num_clusters
