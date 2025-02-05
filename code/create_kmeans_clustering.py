import torch
import numpy as np
import argparse
import time

def kmeans_torch(points, num_clusters, tol=1e-4, verbose=False):
    """
    Perform KMeans clustering using PyTorch.

    Args:
        points (numpy.ndarray): Data points of shape (N, D).
        num_clusters (int): Number of clusters.
        tol (float): Convergence tolerance for stopping criteria.
        verbose (bool): Whether to print iteration progress.

    Returns:
        centroids (torch.Tensor): Final cluster centroids.
        labels (numpy.ndarray): Cluster labels for each point.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    points = torch.tensor(points, dtype=torch.float32).to(device)
    centroids = points[torch.randperm(points.size(0))[:num_clusters]]  # Random initial centroids

    prev_centroids = centroids.clone()
    for iteration in range(1000):  # Maximum iterations safeguard
        # Calculate distances from points to centroids
        distances = torch.cdist(points, centroids, p=2)
        labels = distances.argmin(dim=1)

        # Update centroids
        for i in range(num_clusters):
            cluster_points = points[labels == i]
            if cluster_points.size(0) > 0:
                centroids[i] = cluster_points.mean(dim=0)

        # Check for convergence
        centroid_shift = torch.norm(centroids - prev_centroids, p=2, dim=1).max()
        if verbose:
            print(f"Iteration {iteration + 1}, Centroid shift: {centroid_shift:.6f}")

        if centroid_shift < tol:
            if verbose:
                print("Convergence reached.")
            break

        prev_centroids = centroids.clone()

    return centroids, labels.cpu().numpy()


def main(args):
    # Load the data
    P = np.load(args.point_file)
    V = np.load(args.vocab_file)

    useable_count = int(float(args.count) * len(V)) if args.count != -1 else -1
    P = P[:useable_count, :] if useable_count != -1 else P
    V = V[:useable_count] if useable_count != -1 else V

    K = int(args.cluster)
    start_time = time.time()

    # Run KMeans clustering
    centroids, labels = kmeans_torch(P, K, tol=1e-4, verbose=True)

    # Save the clustering results
    clusters = {i: [] for i in range(K)}
    for v, l in zip(V, labels):
        clusters[l].append(f'{v}|||{l}')

    out_file = f"{args.output_path}/{args.prefix}-clusters-kmeans-{K}.txt"
    with open(out_file, 'w') as f:
        for k, v in clusters.items():
            f.write('\n'.join(v) + '\n')

    end_time = time.time()
    print(f"Clustering completed in {end_time - start_time:.2f} seconds.")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-file", "-v", help="Path to the vocab file")
    parser.add_argument("--point-file", "-p", help="Path to the point file")
    parser.add_argument("--output-path", "-o", help="Output path for clustering results")
    parser.add_argument("--prefix", "-pf", help="Prefix for the output file")
    parser.add_argument("--cluster", "-k", help="Number of clusters")
    parser.add_argument("--count", "-c", help="Point count ratio", default=-1)

    args = parser.parse_args()
    main(args)
