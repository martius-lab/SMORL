import numpy as np
import sklearn


def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h


def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
  return m

def _histogram_discretize(target, num_bins=30):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized


def make_discretizer(target, num_bins=30,
                     discretizer_fn=_histogram_discretize):
  """Wrapper that creates discretizers."""
  return discretizer_fn(target, num_bins)


def compute_mig(representation, gt_factors, k=1): 
    """Computes score based on both training and testing codes and factors.
    k is the size of subspace that should be disentangled from other parts.
    """
    representation = representation.cpu().detach().numpy().T 
    gt_factors = gt_factors.cpu().detach().numpy().T
    discretized_representation = make_discretizer(representation)
    discretized_gt_factors = make_discretizer(gt_factors)
    m = discrete_mutual_info(discretized_representation, discretized_gt_factors)
    z_dim = m.shape[0]
    if k > 1:
        m_comp = np.row_stack([m[i:i+k, :].mean(axis=0) for i in range(0,z_dim, k)])
    else: 
        m_comp = m
    entropy = discrete_entropy(discretized_gt_factors)
    sorted_m = np.sort(m_comp, axis=0)[::-1]
    score = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    return score, m_comp