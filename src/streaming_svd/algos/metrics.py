"""Metrics for evaluating SVD approximation quality."""

import torch


def rel_fro_error(A, U, s, Vt):
    """
    Compute relative Frobenius norm error of SVD approximation.
    
    Computes ||A - U @ diag(s) @ Vt||_F / ||A||_F
    
    Parameters
    ----------
    A : torch.Tensor
        Original matrix of shape (m, n).
    U : torch.Tensor
        Left singular vectors of shape (m, k).
    s : torch.Tensor
        Singular values of shape (k,).
    Vt : torch.Tensor
        Right singular vectors of shape (k, n).
    
    Returns
    -------
    error : float
        Relative Frobenius norm error.
    """
    # Compute approximation: A_approx = U @ diag(s) @ Vt
    A_approx = U @ torch.diag(s) @ Vt
    
    # Compute norms
    error_norm = torch.linalg.norm(A - A_approx, ord='fro')
    A_norm = torch.linalg.norm(A, ord='fro')
    
    return (error_norm / A_norm).item()


def rel_spec_error_est(A, U, n_iter=3):
    """
    Estimate relative spectral norm error of subspace approximation.
    
    Estimates ||(I - U @ U.T) @ A||_2 / ||A||_2 using power iteration
    on the residual operator.
    
    Parameters
    ----------
    A : torch.Tensor
        Original matrix of shape (m, n).
    U : torch.Tensor
        Orthonormal basis of shape (m, k).
    n_iter : int, optional
        Number of power iterations for estimation. Default is 3.
    
    Returns
    -------
    error : float
        Estimated relative spectral norm error.
    """
    m, n = A.shape
    device = A.device
    dtype = A.dtype
    
    # Random starting vector
    v = torch.randn(n, 1, dtype=dtype, device=device)
    v = v / torch.linalg.norm(v)
    
    # Power iteration on residual: (I - U @ U.T) @ A @ v
    for _ in range(n_iter):
        w = A @ v
        w = w - U @ (U.T @ w)  # Project out U
        v = A.T @ w
        v_norm = torch.linalg.norm(v)
        if v_norm > 1e-10:
            v = v / v_norm
        else:
            break
    
    # Estimate ||(I - UU^T) A||_2
    residual_norm = torch.linalg.norm(A @ v - U @ (U.T @ (A @ v)))
    
    # Compute ||A||_2 (or estimate it)
    A_norm = torch.linalg.norm(A, ord=2)
    
    return (residual_norm / A_norm).item()


def subspace_sin_theta(U1, U2):
    """
    Compute subspace distance using principal angles (sine of angles).
    
    Computes ||sin(Theta)||_2 where Theta are the principal angles
    between subspaces spanned by U1 and U2.
    
    This is computed as ||I - (U1.T @ U2) @ (U2.T @ U1)||_2^(1/2).
    Alternatively, using SVD of U1.T @ U2.
    
    Parameters
    ----------
    U1 : torch.Tensor
        Orthonormal basis of shape (m, k1).
    U2 : torch.Tensor
        Orthonormal basis of shape (m, k2).
    
    Returns
    -------
    dist : float
        Subspace distance ||sin(Theta)||_2.
    
    Notes
    -----
    If k1 != k2, the result measures the distance between the
    k-dimensional subspaces where k = min(k1, k2).
    """
    # Compute overlap matrix
    M = U1.T @ U2
    
    # SVD of M gives cosines of principal angles
    _, sigma, _ = torch.linalg.svd(M, full_matrices=False)
    
    # Clamp to [0, 1] for numerical stability
    sigma = torch.clamp(sigma, 0.0, 1.0)
    
    # sin(theta) = sqrt(1 - cos^2(theta))
    sin_theta = torch.sqrt(1.0 - sigma**2)
    
    # Return spectral norm (maximum sine)
    return sin_theta.max().item()


def subspace_sin_theta_fro(U1, U2):
    """
    Compute Frobenius norm of sine of principal angles.
    
    Computes ||sin(Theta)||_F where Theta are the principal angles
    between subspaces spanned by U1 and U2.
    
    Parameters
    ----------
    U1 : torch.Tensor
        Orthonormal basis of shape (m, k1).
    U2 : torch.Tensor
        Orthonormal basis of shape (m, k2).
    
    Returns
    -------
    dist : float
        Subspace distance ||sin(Theta)||_F.
    """
    # Compute overlap matrix
    M = U1.T @ U2
    
    # SVD of M gives cosines of principal angles
    _, sigma, _ = torch.linalg.svd(M, full_matrices=False)
    
    # Clamp to [0, 1] for numerical stability
    sigma = torch.clamp(sigma, 0.0, 1.0)
    
    # sin(theta) = sqrt(1 - cos^2(theta))
    sin_theta = torch.sqrt(1.0 - sigma**2)
    
    # Return Frobenius norm
    return torch.linalg.norm(sin_theta).item()
