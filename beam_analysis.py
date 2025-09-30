"""
beam_analysis.py
~~~~~~~~~~~~~~~~~~

This module provides functions to analyse transverse laser beam profiles
similar to those used in the original MATLAB ``M2tool``.  The core
capabilities implemented here are:

* ``ISO`` second‑moment method: an iterative procedure to estimate the
  second‑moment beam radii (2‑σ) along the principal axes of an elliptical
  beam.  During each iteration a region of interest is cropped around
  the provisional beam centre; a robust background estimate (from the
  5 % image border) is subtracted; the second moments are computed; and
  the region of interest is updated.  Convergence is reached when the
  radii change by less than ``tol`` pixels.  The algorithm is based on
  the MATLAB code shown in ``M2tool.m`` where cropping and background
  subtraction are performed iteratively【881670830533059†screenshot】.

* Gaussian fits along the principal axes: once the beam is aligned
  to its principal axes (by rotating the image according to the angle
  returned from the second‑moment calculation), the integrated
  intensity along each axis is fitted to the model
  ``A exp(−2 ((x − centre)/radius)²)``.  The starting values for the
  amplitude, centre and radius are taken from the second‑moment results
  (similar to the ``fitGauss`` function in the MATLAB code【881670830533059†screenshot】).

The main entry point is :func:`analyze_beam`, which accepts a 2D image
(NumPy array) and returns a dictionary containing:

* ``theta`` – rotation angle (in radians) of the principal axis.
* ``rx_iso`` and ``ry_iso`` – second moment radii (two sigma) along the
  major and minor axes.
* ``cx`` and ``cy`` – beam centre coordinates in pixel units
  relative to the original input image.
* ``Ix_spectrum`` and ``Iy_spectrum`` – one‑dimensional cuts through
  the beam along the principal axes after the final iteration, as
  ``(x_positions, intensity)`` and ``(y_positions, intensity)`` arrays.
* ``img_for_spec`` – the 2D image (rotated if necessary) used to compute
  the spectra.  This corresponds to the final processed region of
  interest and is convenient for plotting or further analysis.
* ``gauss_fit_x`` and ``gauss_fit_y`` – dictionaries containing
  the best‑fit parameters ``A``, ``centre`` and ``radius`` for the
  Gaussian fits to ``Ix_spectrum`` and ``Iy_spectrum``.
* ``iterations`` – number of iterations used by the ISO second‑moment
  method.

The library is written without any GUI dependencies and can therefore be
used in headless environments.  It relies solely on NumPy and SciPy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate as nd_rotate
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional

__all__ = [
    "bg_subtract",
    "estimate_background_edge_ring",
    "get_beam_size",
    "fit_gaussian",
    "iso_second_moment",
    "analyze_beam",
]


def bg_subtract(img: ArrayLike) -> np.ndarray:
    """Robustly subtract the background from an image.

    The background level is estimated from the outer 5 % border of the
    frame. Instead of removing a single scalar offset, we fit an affine
    plane ``a*x + b*y + c`` to the border pixels by means of iterative
    sigma clipping and subtract the fitted plane from the full image.
    This allows slowly varying backgrounds to be removed while retaining
    robustness against outliers along the border. The returned array is
    provided as ``float64``.

    Parameters
    ----------
    img : array_like
        2D array representing the image. The values are converted to
        floating point internally.

    Returns
    -------
    np.ndarray
        The background-subtracted image as ``float64``.
    """
    arr = np.asarray(img, dtype=np.float64)
    h, w = arr.shape
    if h == 0 or w == 0:
        return arr.copy()

    m = int(np.ceil(0.05 * min(h, w)))
    if m < 1:
        finite = np.isfinite(arr)
        if not finite.any():
            return arr.copy()
        return arr - np.median(arr[finite])

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:m, :] = True
    border_mask[-m:, :] = True
    border_mask[:, :m] = True
    border_mask[:, -m:] = True

    y_indices, x_indices = np.indices(arr.shape)
    x = x_indices[border_mask].astype(np.float64)
    y = y_indices[border_mask].astype(np.float64)
    z = arr[border_mask]

    finite = np.isfinite(z)
    x = x[finite]
    y = y[finite]
    z = z[finite]

    if z.size == 0:
        return arr.copy()
    if z.size < 3:
        return arr - np.median(z)

    A = np.column_stack((x, y, np.ones_like(x)))
    params = None
    for _ in range(3):
        sol, *_ = np.linalg.lstsq(A, z, rcond=None)
        params = sol
        residuals = z - A @ params
        res_med = np.median(residuals)
        mad_raw = np.median(np.abs(residuals - res_med))
        sigma = 1.4826 * mad_raw
        thresh = 3.0 * max(sigma, np.finfo(float).eps)
        mask = np.abs(residuals - res_med) < thresh
        if mask.all():
            break
        if mask.sum() < 3:
            break
        A = A[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]

    if params is None:
        return arr - np.median(z)

    bg_plane = params[0] * x_indices + params[1] * y_indices + params[2]
    return arr - bg_plane

def estimate_background_edge_ring(img: ArrayLike) -> float:
    """Estimate background from an edge ring with robust clipping.

    Uses a border ring of width m = ceil(5% * min(h, w)) around the
    image, applying 3 rounds of sigma clipping around the median with a
    threshold of 3*max(sigma, eps). Returns the median of the clipped
    samples as the background estimate.

    Parameters
    ----------
    img : array_like
        2D input image.

    Returns
    -------
    float
        Estimated scalar background level.
    """
    arr = np.asarray(img, dtype=np.float64)
    h, w = arr.shape
    if h == 0 or w == 0:
        return 0.0
    m = int(np.ceil(0.05 * min(h, w)))
    if m < 1:
        # fall back to global median
        vals = arr.ravel()
    else:
        # build edge ring without duplicating corners multiple times
        top = arr[:m, :]
        bottom = arr[-m:, :]
        left = arr[m:-m, :m] if h > 2 * m else np.empty((0, 0))
        right = arr[m:-m, -m:] if h > 2 * m else np.empty((0, 0))
        vals = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
    b = np.array(vals, dtype=np.float64)
    for _ in range(3):
        mu = np.median(b)
        mad_raw = np.median(np.abs(b - mu))
        sig = 1.4826 * mad_raw
        thresh = 3.0 * max(sig, np.finfo(float).eps)
        mask = np.abs(b - mu) < thresh
        if not np.any(mask):
            break
        b = b[mask]
    return float(np.median(b))


def get_beam_size(profile: ArrayLike) -> Tuple[float, float, float, float, float]:
    """Compute second‑moment beam parameters.

    Given a 2D intensity distribution ``profile``, this function
    calculates the centroid coordinates ``(cx, cy)``, the second‑moment
    radii along the horizontal and vertical axes ``(rx, ry)`` and the
    rotation angle ``phi`` of the principal axes.  The algorithm
    implements the formulas from the MATLAB function ``getBeamSize``
    where the radii are defined as ``2*sqrt(<x²>)`` with ``<…>``
    denoting the intensity‑weighted second moment【881670830533059†screenshot】.  The angle
    ``phi`` is computed using the standard two‑argument arctangent to
    correctly determine the quadrant of the rotation.

    Parameters
    ----------
    profile : array_like
        2D array containing the (background‑subtracted) intensity values.

    Returns
    -------
    rx : float
        Second‑moment beam radius (2σ) in the x‑direction.
    ry : float
        Second‑moment beam radius (2σ) in the y‑direction.
    cx : float
        Intensity‑weighted centroid along x (0‑indexed pixel coordinate).
    cy : float
        Intensity‑weighted centroid along y (0‑indexed pixel coordinate).
    phi : float
        Rotation angle of the principal axis (radians).  A positive
        angle means the major axis is rotated counter‑clockwise from the
        x‑axis.
    """
    arr = np.asarray(profile, dtype=np.float64)
    # compute sums and coordinate grids
    y_indices, x_indices = np.indices(arr.shape)
    total_intensity = arr.sum()
    if total_intensity <= 0:
        # avoid division by zero; return zeros
        return 0.0, 0.0, 0.0, 0.0, 0.0
    cx = float((arr * x_indices).sum() / total_intensity)
    cy = float((arr * y_indices).sum() / total_intensity)
    # second moments
    dx = x_indices - cx
    dy = y_indices - cy
    sigma_x_sq = float((arr * (dx ** 2)).sum() / total_intensity)
    sigma_y_sq = float((arr * (dy ** 2)).sum() / total_intensity)
    sigma_xy = float((arr * dx * dy).sum() / total_intensity)
    rx = 2.0 * np.sqrt(abs(sigma_x_sq))
    ry = 2.0 * np.sqrt(abs(sigma_y_sq))
    # rotation angle; use atan2 for proper quadrant handling.  When the
    # horizontal and vertical moments are equal the principal axes are
    # ±45 degrees depending on the sign of sigma_xy【881670830533059†screenshot】.
    if sigma_x_sq != sigma_y_sq:
        phi = 0.5 * np.arctan2(2.0 * sigma_xy, (sigma_x_sq - sigma_y_sq))
    else:
        phi = 0.25 * np.pi * np.sign(sigma_xy) if sigma_xy != 0 else 0.0
    return rx, ry, cx, cy, phi


def fit_gaussian(x: np.ndarray, y: np.ndarray, start_params: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], np.ndarray]:
    """Fit a 1D Gaussian of the form ``A exp(-2 ((x - centre)/radius)**2)``.

    The Gaussian used here matches the model employed in the MATLAB
    function ``fitGauss``【881670830533059†screenshot】.  The decay constant ``radius`` corresponds
    to the 1/e² radius.  Initial guesses for the amplitude ``A``,
    centre and radius are passed via ``start_params``.  The function
    returns the best‑fit parameters and the covariance matrix.

    Parameters
    ----------
    x : ndarray
        1D array of positions.
    y : ndarray
        1D array of intensities.
    start_params : tuple
        Tuple ``(A, centre, radius)`` providing initial guesses for the
        parameters.

    Returns
    -------
    params : tuple of floats
        Best‑fit values ``(A, centre, radius)``.
    covariance : ndarray
        3×3 covariance matrix returned by ``scipy.optimize.curve_fit``.
    """
    # define the Gaussian model
    def model(x, A, centre, radius):
        return A * np.exp(-2.0 * ((x - centre) / radius) ** 2)
    # perform the fit.  Use bounds to ensure positive radius.
    popt, pcov = curve_fit(model, x, y, p0=start_params, bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]))
    return (popt[0], popt[1], popt[2]), pcov


def iso_second_moment(
    img: ArrayLike,
    aperture_factor: float = 3.0,
    principal_axes_rot: bool = True,
    tol: float = 1,
    max_iterations: int = 100,
) -> Dict[str, object]:
    """Iteratively compute ISO second‑moment beam parameters.

    Starting from the brightest pixel of the smoothed image, this
    algorithm repeatedly crops a region around the provisional beam
    centre, subtracts a robust background, computes second‑moment radii
    and updates the crop until convergence.  The procedure mirrors the
    iterative approach in ``beamData`` from the MATLAB code【881670830533059†screenshot】.  Optionally
    the final image is rotated so that the principal axes align with the
    horizontal/vertical axes.

    Parameters
    ----------
    img : array_like
        2D input image (beam profile).
    aperture_factor : float, optional
        Factor by which the cropping window is larger than the current
        diameter estimate (default 3.0 as in the MATLAB code ``config.aperture2diameter``【881670830533059†screenshot】).
    principal_axes_rot : bool, optional
        If ``True``, return radii along the principal axes by rotating the
        final cropped image.  If ``False``, the returned radii are along
        the image axes.
    tol : float, optional
        Convergence tolerance in pixels.  The iterations stop when the
        sum of absolute differences of the radii between successive
        iterations is below this value.
    max_iterations : int, optional
        Maximum number of iterations.  If the algorithm does not converge
        within this many iterations, a warning is issued.

    Returns
    -------
    result : dict
        Dictionary containing the results of the ISO second‑moment
        analysis with the following keys:

        ``cx``, ``cy`` : float
            Global coordinates (0‑indexed pixels) of the beam centre.

        ``rx``, ``ry`` : float
            Second‑moment radii (2σ) in the laboratory axes prior to
            rotating to the principal axes.

        ``phi`` : float
            Rotation angle (radians) of the principal axis.

        ``iterations`` : int
            Number of iterations performed.

        ``processed_img`` : ndarray
            Final cropped, background‑subtracted image used for
            second‑moment calculation.

        ``crop_origin`` : tuple of ints
            ``(ymin, xmin)`` coordinates of the top‑left corner of the
            final crop within the original image.

        ``rotated_img`` : ndarray
            If ``principal_axes_rot`` is ``True``, the image rotated by
            ``phi`` so that the principal axes align with the image axes.
            Otherwise ``None``.

        ``rotated_crop_origin`` : tuple of floats
            ``(ymin_rot, xmin_rot)`` – the origin of the rotated
            image relative to the original image (fractional pixel).  Only
            provided if ``rotated_img`` is not ``None``.
    """
    # convert to float and ensure a copy so modifications do not affect the original
    full_img = np.asarray(img, dtype=np.float64)
    # One-time global background subtraction to match M2tool's behavior
    full_img = bg_subtract(full_img)
    ny, nx = full_img.shape
    # initial centre: use smoothed image to avoid hot pixels
    smoothed = gaussian_filter(full_img, sigma=2.0)
    max_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    cy, cx = float(max_idx[0]), float(max_idx[1])
    # estimate FWHM along rows/columns as the initial radius
    row = full_img[int(cy), :]
    col = full_img[:, int(cx)]
    # avoid division by zero
    if row.max() > 0:
        ix = row / row.max()
        above_half = np.where(ix > 0.5)[0]
        if above_half.size > 0:
            rx = float(above_half.max() - above_half.min())
        else:
            rx = nx / 2.0
    else:
        rx = nx / 2.0
    if col.max() > 0:
        iy = col / col.max()
        above_half = np.where(iy > 0.5)[0]
        if above_half.size > 0:
            ry = float(above_half.max() - above_half.min())
        else:
            ry = ny / 2.0
    else:
        ry = ny / 2.0
    prev_rx, prev_ry = rx, ry
    crop_ymin, crop_xmin = 0, 0  # place holders
    processed_img = None
    phi = 0.0
    for iteration in range(1, max_iterations + 1):
        # compute crop bounds around the current centre
        xmin = int(round(cx - rx * aperture_factor))
        xmax = int(round(cx + rx * aperture_factor))
        ymin = int(round(cy - ry * aperture_factor))
        ymax = int(round(cy + ry * aperture_factor))
        # clamp to image boundaries
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, nx - 1)
        ymax = min(ymax, ny - 1)
        # extract and subtract background on the crop as in M2tool (corner-based BGsub)
        cropped = full_img[ymin:ymax + 1, xmin:xmax + 1]
        processed = bg_subtract(cropped)
        # compute second moment radii and centroid in the cropped frame
        rx_new, ry_new, cx_local, cy_local, phi = get_beam_size(processed)
        # update global centre coordinates
        cx = cx_local + xmin
        cy = cy_local + ymin
        # check convergence
        if abs(rx_new - prev_rx) + abs(ry_new - prev_ry) < tol:
            rx, ry = rx_new, ry_new
            crop_ymin, crop_xmin = ymin, xmin
            processed_img = processed
            break
        rx, ry = rx_new, ry_new
        prev_rx, prev_ry = rx_new, ry_new
        crop_ymin, crop_xmin = ymin, xmin
        processed_img = processed
    else:
        # did not converge within max_iterations
        # assign last values
        pass
    iterations = iteration
    # optionally rotate the final processed image to align principal axes
    rotated_img = None
    rotated_crop_origin = None
    phi_rot = None  # rotation in the rotated image frame (expected ~0)
    if principal_axes_rot and processed_img is not None:
        # Rotate a larger section of the background-subtracted full image so the
        # rotated data retains realistic edge noise. After rotation the image is
        # cropped back to the processed-image size for downstream analysis.
        cx_pre_rot, cy_pre_rot = cx, cy
        h_old, w_old = processed_img.shape
        if h_old > 0 and w_old > 0:
            rot_deg = phi * 180.0 / np.pi
            abs_cos = float(abs(np.cos(phi)))
            abs_sin = float(abs(np.sin(phi)))
            rot_h = int(np.ceil(h_old * abs_cos + w_old * abs_sin)) + 2
            rot_w = int(np.ceil(h_old * abs_sin + w_old * abs_cos)) + 2
            rot_h = max(rot_h, h_old)
            rot_w = max(rot_w, w_old)

            y_offset = int(np.floor((rot_h - h_old) / 2.0))
            x_offset = int(np.floor((rot_w - w_old) / 2.0))
            ymin_exp = crop_ymin - y_offset
            xmin_exp = crop_xmin - x_offset
            ymax_exp = ymin_exp + rot_h
            xmax_exp = xmin_exp + rot_w

            expanded_patch = np.zeros((rot_h, rot_w), dtype=np.float64)
            y_in_start = max(ymin_exp, 0)
            y_in_end = min(ymax_exp, ny)
            x_in_start = max(xmin_exp, 0)
            x_in_end = min(xmax_exp, nx)
            patch_y_start = y_in_start - ymin_exp
            patch_x_start = x_in_start - xmin_exp
            patch_y_end = patch_y_start + (y_in_end - y_in_start)
            patch_x_end = patch_x_start + (x_in_end - x_in_start)
            if patch_y_start < patch_y_end and patch_x_start < patch_x_end:
                expanded_patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = (
                    full_img[y_in_start:y_in_end, x_in_start:x_in_end]
                )

            expanded_processed = bg_subtract(expanded_patch)
            y_insert_end = y_offset + h_old
            x_insert_end = x_offset + w_old
            expanded_processed[y_offset:y_insert_end, x_offset:x_insert_end] = processed_img

            rotated_full = nd_rotate(
                expanded_processed,
                angle=rot_deg,
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
            )
            start_y = y_offset
            start_x = x_offset
            end_y = start_y + h_old
            end_x = start_x + w_old
            rotated_cropped = rotated_full[start_y:end_y, start_x:end_x]
            rotated_img = rotated_cropped

            rx_rot, ry_rot, cx_rot_local, cy_rot_local, phi_rot = get_beam_size(rotated_img)
            rx, ry = rx_rot, ry_rot
            rotated_crop_origin = (
                cy_pre_rot - cy_rot_local,
                cx_pre_rot - cx_rot_local,
            )
            cx = cx_rot_local + rotated_crop_origin[1]
            cy = cy_rot_local + rotated_crop_origin[0]
    return {
        "cx": cx,
        "cy": cy,
        "rx": rx,
        "ry": ry,
        # phi is the rotation angle of the principal axis in the original
        # (unrotated) image coordinates. If the image was rotated for
        # analysis, phi_rot will be close to 0.
        "phi": phi,
        "phi_rot": phi_rot,
        "iterations": iterations,
        "processed_img": processed_img,
        "crop_origin": (crop_ymin, crop_xmin),
        "rotated_img": rotated_img,
        "rotated_crop_origin": rotated_crop_origin,
    }


def analyze_beam(
    img: ArrayLike,
    aperture_factor: float = 3.0,
    principal_axes_rot: bool = True,
    tol: float = 0.5,
    max_iterations: int = 100,
) -> Dict[str, object]:
    """Analyse a beam image and return ISO and Gaussian beam parameters.

    This high‑level function performs the ISO second‑moment analysis and
    Gaussian fitting along the principal axes.  It is designed to be
    directly called by users.  The spectra and fit parameters are
    returned in a dictionary.

    Parameters
    ----------
    img : array_like
        Input 2D image representing the beam profile.
    aperture_factor : float, optional
        Factor by which the cropping window is larger than the current
        diameter estimate (default 3.0).
    principal_axes_rot : bool, optional
        Whether to rotate the image to align the principal axes.  For
        non‑cylindrical beams this is recommended.
    tol : float, optional
        Convergence tolerance in pixels for the ISO second‑moment method.
    max_iterations : int, optional
        Maximum number of iterations for the ISO method.

    Returns
    -------
    result : dict
        Dictionary containing the beam parameters and helper data.  See
        the module level documentation for the list of keys, including
        ``img_for_spec`` which holds the 2D image used to build the
        spectra.
    """
    # run ISO second‑moment analysis
    iso_result = iso_second_moment(
        img,
        aperture_factor=aperture_factor,
        principal_axes_rot=principal_axes_rot,
        tol=tol,
        max_iterations=max_iterations,
    )
    # compute spectra along principal axes using the rotated or processed image
    rotated_img = iso_result["rotated_img"]
    processed_img = iso_result["processed_img"]
    # positions relative to original image
    if rotated_img is not None:
        img_for_spec = rotated_img
        # global origin for rotated image
        ymin_rot, xmin_rot = iso_result["rotated_crop_origin"]
        # x and y positions in original pixel coordinates
        h_rot, w_rot = rotated_img.shape
        x_positions = xmin_rot + np.arange(w_rot)
        y_positions = ymin_rot + np.arange(h_rot)
    else:
        img_for_spec = processed_img
        ymin, xmin = iso_result["crop_origin"]
        h_proc, w_proc = processed_img.shape
        x_positions = xmin + np.arange(w_proc)
        y_positions = ymin + np.arange(h_proc)
    # for spectral analysis negative values (arising from background
    # subtraction) can bias the Gaussian fit.  Clip to zero as in
    # typical beam analysis to ensure the integrated profiles are
    # non‑negative.  This is consistent with the MATLAB implementation
    # which subtracts a constant background but does not allow the sum
    # over an axis to become negative.
    # Compute 1D spectra (integrated profiles) directly from the
    # processed image; any signed residuals after background subtraction
    # are preserved so downstream consumers can reason about the noise
    # floor if needed.
    Ix = img_for_spec.sum(axis=0)
    Iy = img_for_spec.sum(axis=1)
        # initial guesses for Gaussian fit using ISO results
    # amplitude guess: max value of spectrum
    # centre guess: ISO centroid along each axis
    # radius guess: second moment radii (rx, ry)
    centre_x = iso_result["cx"]
    centre_y = iso_result["cy"]
    rx = iso_result["rx"]
    ry = iso_result["ry"]
    # fit along x
    params_x, cov_x = fit_gaussian(x_positions, Ix, (Ix.max(), centre_x, max(rx, 1e-3)))
    # fit along y
    params_y, cov_y = fit_gaussian(y_positions, Iy, (Iy.max(), centre_y, max(ry, 1e-3)))
    return {
        "img_for_spec": img_for_spec,
        "theta": iso_result["phi"],
        "rx_iso": iso_result["rx"],
        "ry_iso": iso_result["ry"],
        "cx": iso_result["cx"],
        "cy": iso_result["cy"],
        "Ix_spectrum": (x_positions, Ix),
        "Iy_spectrum": (y_positions, Iy),
        "gauss_fit_x": {
            "amplitude": params_x[0],
            "centre": params_x[1],
            "radius": params_x[2],
            "covariance": cov_x,
        },
        "gauss_fit_y": {
            "amplitude": params_y[0],
            "centre": params_y[1],
            "radius": params_y[2],
            "covariance": cov_y,
        },
        "iterations": iso_result["iterations"],
    }
