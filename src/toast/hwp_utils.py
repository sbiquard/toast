# Copyright (c) 2023-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.linalg import eigvalsh, lu_factor, lu_solve

from .dist import distribute_uniform
from .mpi import MPI


def hwpss_samples(n_samp, comm):
    """Helper function to distribute samples.

    This distributes slices of samples uniformly among the processes
    of an MPI communicator.

    Args:
        n_samp (int): The number of samples
        comm (MPI.Comm): The communicator.

    Returns:
        (slice):  The local slice on this process.

    """
    if comm is None:
        slc = slice(0, n_samp, 1)
        return slc
    dist = distribute_uniform(n_samp, comm.size)
    # The sample counts and displacement for each process
    sample_count = [x.n_elem for x in dist]
    sample_displ = [x.offset for x in dist]

    # The local sample range as a slice
    slc = slice(
        sample_displ[comm.rank],
        sample_displ[comm.rank] + sample_count[comm.rank],
        1,
    )
    return slc


def hwpss_sincos_buffer(angles, flags, n_harmonics, comm=None):
    """Precompute sin / cos terms.

    Compute the harmonic terms involving sin / cos of the HWP angle.
    This is done once in parallel and used by each process for its
    local detectors.

    Args:
        angles (array):  The HWP angle at each time stamp
        flags (array):  The flags indicating bad angle samples.
        n_harmonics (ing):  The number of harmonics in the model.
        comm (MPI.Comm):  The optional communicator to parallelize
            the calculation.

    Returns:
        (array):  The full buffer of sin/cos factors on all processes.

    """
    slc = hwpss_samples(len(angles), comm)
    ang = np.copy(angles[slc])
    good = flags[slc] == 0
    my_samp = len(ang)
    sample_vals = 2 * n_harmonics
    my_sincos = np.zeros((my_samp, sample_vals), dtype=np.float32)
    for h in range(n_harmonics):
        my_sincos[good, 2 * h] = np.sin((h + 1) * ang[good])
        my_sincos[good, 2 * h + 1] = np.cos((h + 1) * ang[good])
    if comm is None:
        return my_sincos
    sincos = np.vstack(comm.allgather(my_sincos))
    return sincos


def hwpss_compute_coeff_covariance(times, flags, sincos, comm=None):
    """Build covariance of HWPSS model coefficients.

    The HWPSS model for this function is the one used by the Maxipol
    and EBEX experiments.  See for example Joy Didier's thesis, equation
    8.17:  https://academiccommons.columbia.edu/doi/10.7916/D8MW2HCG

    The model with N harmonics and HWP angle H can be written as:

    h(t) = Sum(i=1...N) { [ C_i0 + C_i1 * t ] cos(i * H(t)) +
                          [ C_i2 + C_i3 * t ] sin(i * H(t)) ] }

    Writing this in terms of the vector of coefficients and the matrix of
    of sin / cos factors:

    h(t) = M x C
    h(t) =
    [ cos(H(t_0))  t_0 cos(H(t_0))  sin(H(t_0))  t_0 sin(H(t_0))  cos(2H(t_0)) ...]
    [ cos(H(t_1))  t_1 cos(H(t_1))  sin(H(t_1))  t_1 sin(H(t_1))  cos(2H(t_1)) ...]
    [ ... ]  X  [ C_10 C_11 C_12 C_13 C_20 C_21 C_22 C_23 C_30 ... ]^T

    The least squares solution for the coefficients is then

    C = (M^T M)^-1 M^T h(t)

    We then assume that h(t) is just the input data and that it is dominated by the
    HWPSS.  This function computes the covariance matrix and factors it for later
    use.

    Args:
        times (array):  The **relative** timestamps of the samples from the start
            of the observation.
        flags (array):  The flags indicating bad angle samples
        sincos (array):  The pre-computed sin / cos terms.
        comm (MPI.Comm):  The optional communicator to parallelize
            the calculation.

    Returns:
        (tuple):  The LU factorization and pivots.

    """
    slc = hwpss_samples(len(times), comm)
    my_sincos = sincos[slc, :]
    my_times = times[slc]
    my_flags = flags[slc]

    n_harmonics = sincos.shape[1] // 2
    my_cov = np.zeros((4 * n_harmonics, 4 * n_harmonics), dtype=np.float64)
    good = my_flags == 0
    # Compute local upper triangle
    for hr in range(0, n_harmonics):
        for hc in range(hr, n_harmonics):
            my_cov[4 * hr + 0, 4 * hc + 0] = np.dot(
                my_sincos[good, 2 * hr + 0], my_sincos[good, 2 * hc + 0]
            )
            my_cov[4 * hr + 0, 4 * hc + 1] = np.dot(
                my_sincos[good, 2 * hr + 0],
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 0]),
            )
            my_cov[4 * hr + 0, 4 * hc + 2] = np.dot(
                my_sincos[good, 2 * hr + 0], my_sincos[good, 2 * hc + 1]
            )
            my_cov[4 * hr + 0, 4 * hc + 3] = np.dot(
                my_sincos[good, 2 * hr + 0],
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 1]),
            )

            my_cov[4 * hr + 1, 4 * hc + 0] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 0]),
                my_sincos[good, 2 * hc + 0],
            )
            my_cov[4 * hr + 1, 4 * hc + 1] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 0]),
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 0]),
            )
            my_cov[4 * hr + 1, 4 * hc + 2] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 0]),
                my_sincos[good, 2 * hc + 1],
            )
            my_cov[4 * hr + 1, 4 * hc + 3] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 0]),
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 1]),
            )

            my_cov[4 * hr + 2, 4 * hc + 0] = np.dot(
                my_sincos[good, 2 * hr + 1], my_sincos[good, 2 * hc + 0]
            )
            my_cov[4 * hr + 2, 4 * hc + 1] = np.dot(
                my_sincos[good, 2 * hr + 1],
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 0]),
            )
            my_cov[4 * hr + 2, 4 * hc + 2] = np.dot(
                my_sincos[good, 2 * hr + 1], my_sincos[good, 2 * hc + 1]
            )
            my_cov[4 * hr + 2, 4 * hc + 3] = np.dot(
                my_sincos[good, 2 * hr + 1],
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 1]),
            )

            my_cov[4 * hr + 3, 4 * hc + 0] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 1]),
                my_sincos[good, 2 * hc + 0],
            )
            my_cov[4 * hr + 3, 4 * hc + 1] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 1]),
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 0]),
            )
            my_cov[4 * hr + 3, 4 * hc + 2] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 1]),
                my_sincos[good, 2 * hc + 1],
            )
            my_cov[4 * hr + 3, 4 * hc + 3] = np.dot(
                np.multiply(my_times[good], my_sincos[good, 2 * hr + 1]),
                np.multiply(my_times[good], my_sincos[good, 2 * hc + 1]),
            )
    # Accumulate across processes
    if comm is None:
        cov = my_cov
    else:
        cov = np.zeros((4 * n_harmonics, 4 * n_harmonics), dtype=np.float64)
        comm.Allreduce(my_cov, cov, op=MPI.SUM)

    # Fill in lower triangle
    for hr in range(0, 4 * n_harmonics):
        for hc in range(0, hr):
            cov[hr, hc] = cov[hc, hr]
    # Check that condition number is reasonable
    evals = eigvalsh(cov)
    rcond = np.min(evals) / np.max(evals)
    if rcond < 1.0e-8:
        return None
    # LU factorization for later solve
    cov_lu, cov_piv = lu_factor(cov)
    return cov_lu, cov_piv


def hwpss_compute_coeff(detdata, flags, times, sincos, cov_lu, cov_piv):
    """Compute the HWPSS model coefficients.

    See docstring for `hwpss_compute_coeff_covariance`.  This function computes
    the expression M^T h(t) and then uses the LU factorization of the covariance
    to solve for the coefficients.

    Args:
        detdata (array):  The detector data for one detector.
        flags (array):  The detector flags.
        times (array):  The **relative** timestamps of the samples from the start
            of the observation.
        sincos (array):  The pre-computed sin / cos terms.
        cov_lu (array):  The covariance LU factorization.
        cov_piv (array):  The covariance pivots.

    Returns:
        (array):  The coefficients.

    """
    n_samp = len(times)
    n_harmonics = sincos.shape[1] // 2
    good = flags == 0
    bad = np.logical_not(good)
    input = np.copy(detdata)
    dc = np.mean(input[good])
    input[:] -= dc
    input[bad] = 0
    rhs = np.zeros(4 * n_harmonics, dtype=np.float64)
    for h in range(n_harmonics):
        rhs[4 * h + 0] = np.dot(input[good], sincos[good, 2 * h])
        rhs[4 * h + 1] = np.dot(
            input[good], np.multiply(sincos[good, 2 * h], times[good])
        )
        rhs[4 * h + 2] = np.dot(input[good], sincos[good, 2 * h + 1])
        rhs[4 * h + 3] = np.dot(
            input[good], np.multiply(sincos[good, 2 * h + 1], times[good])
        )
    coeff = lu_solve((cov_lu, cov_piv), rhs)
    return coeff


def hwpss_build_model(times, flags, sincos, coeff):
    """Construct the HWPSS template from coefficients.

    Args:
        times (array):  The **relative** timestamps of the samples from the start
            of the observation.
        flags (array):  The flags indicating bad angle samples
        sincos (array):  The pre-computed sin / cos terms.
        coeff (array):  The model coefficents for this detector.

    Returns:
        (array):  The template.

    """
    n_samp = len(times)
    n_harmonics = sincos.shape[1] // 2
    good = flags == 0
    model = np.zeros(n_samp, dtype=np.float64)
    for h in range(n_harmonics):
        model[good] += coeff[4 * h + 0] * sincos[good, 2 * h]
        model[good] += coeff[4 * h + 1] * np.multiply(sincos[good, 2 * h], times[good])
        model[good] += coeff[4 * h + 2] * sincos[good, 2 * h + 1]
        model[good] += coeff[4 * h + 3] * np.multiply(
            sincos[good, 2 * h + 1], times[good]
        )
    return model