"""
beam_plot_example.py
---------------------

This example script demonstrates how to use the :mod:`beam_analysis`
library to analyse a beam profile stored as a BMP image and plot the
integrated spectra along the principal axes together with the Gaussian
fits returned by :func:`beam_analysis.analyze_beam`.

The script expects the path to a BMP image file as its first command line
argument.  It then loads the image, performs the analysis and creates
two plots: one for the spectrum along the x‑axis (Ix) and one for the
y‑axis (Iy).  Each plot shows both the measured spectrum and the best
fit Gaussian curve.  A legend indicates the fitted 1/e² radius for
convenience.

Example usage::

    python beam_plot_example.py path/to/beam_profile.bmp

An optional ``--pixel-size`` argument lets you provide the physical size
of a pixel (together with ``--pixel-unit``).  When supplied, the script
reports ISO and Gaussian radii in both pixels and physical units and
adds secondary axes with the converted scales.

Requirements:

* numpy
* matplotlib
* Pillow (for BMP loading)
* SciPy (beam_analysis dependencies)
* beam_analysis.py located in the same directory or installed on the
  Python path

"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from typing import Optional

# import the analyze_beam function from the beam_analysis module
from beam_analysis import analyze_beam


def main(image_path: str, pixel_size: Optional[float] = None, pixel_unit: str = "µm") -> None:
    """Analyse a beam image and plot the Ix and Iy spectra with fits.

    Parameters
    ----------
    image_path : str
        Path to the BMP image containing the beam profile.
    """
    # Load the BMP image and convert to grayscale
    with Image.open(image_path) as img:
        data = np.asarray(img.convert("L"), dtype=float)

    # Run the beam analysis; adjust aperture_factor here if necessary
    result = analyze_beam(data)

    # Unpack the spectra and fit parameters
    x_positions, Ix = result["Ix_spectrum"]
    y_positions, Iy = result["Iy_spectrum"]
    fit_x = result["gauss_fit_x"]
    fit_y = result["gauss_fit_y"]

    cx = result["cx"]
    cy = result["cy"]
    rx_iso = result["rx_iso"]
    ry_iso = result["ry_iso"]
    is_major_x = rx_iso >= ry_iso

    if pixel_size is not None:
        print(
            "Pixel size set to "
            f"{pixel_size:.6g} {pixel_unit}\n"
            f"ISO radii: rₓ = {rx_iso:.3f} px ({rx_iso * pixel_size:.3f} {pixel_unit}), "
            f"rᵧ = {ry_iso:.3f} px ({ry_iso * pixel_size:.3f} {pixel_unit})\n"
            f"Gaussian radii: wₓ = {fit_x['radius']:.3f} px ({fit_x['radius'] * pixel_size:.3f} {pixel_unit}), "
            f"wᵧ = {fit_y['radius']:.3f} px ({fit_y['radius'] * pixel_size:.3f} {pixel_unit})"
        )

    # Build fitted curves for plotting
    def gauss_curve(x: np.ndarray, params: dict) -> np.ndarray:
        return params["amplitude"] * np.exp(-2.0 * ((x - params["centre"]) / params["radius"]) ** 2)

    Ix_fit = gauss_curve(x_positions, fit_x)
    Iy_fit = gauss_curve(y_positions, fit_y)

    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    # Plot Ix
    axes[0].plot(x_positions, Ix, label="Ix (integrated)", color="C0")
    if pixel_size is not None:
        gauss_x_label = (
            f"Gaussian fit (w={fit_x['radius']:.2f} px / "
            f"{fit_x['radius'] * pixel_size:.2f} {pixel_unit})"
        )
    else:
        gauss_x_label = f"Gaussian fit (w={fit_x['radius']:.2f} px)"
    axes[0].plot(x_positions, Ix_fit, label=gauss_x_label, color="C1")
    # Indicate second moment radius (2σ) on the x spectrum
    x_label = "ISO rₓ (major)" if is_major_x else "ISO rₓ (minor)"
    if pixel_size is not None:
        iso_rx_label = f"{x_label} = {rx_iso:.2f} px ({rx_iso * pixel_size:.2f} {pixel_unit})"
    else:
        iso_rx_label = f"{x_label} = {rx_iso:.2f} px"
    axes[0].axvline(cx - rx_iso, linestyle="--", color="C2", alpha=0.7, label=iso_rx_label)
    axes[0].axvline(cx + rx_iso, linestyle="--", color="C2", alpha=0.7)
    axes[0].set_xlabel("X position (pixels)")
    axes[0].set_ylabel("Integrated intensity")
    axes[0].set_title("Spectrum along principal x‑axis")
    axes[0].legend()
    if pixel_size is not None:
        def px_to_phys_x(x_value: float) -> float:
            return (x_value - cx) * pixel_size

        def phys_to_px_x(x_value: float) -> float:
            return (x_value / pixel_size) + cx

        secax_x = axes[0].secondary_xaxis('top', functions=(px_to_phys_x, phys_to_px_x))
        secax_x.set_xlabel(f"X position relative to centre ({pixel_unit})")

    # Plot Iy
    axes[1].plot(y_positions, Iy, label="Iy (integrated)", color="C0")
    if pixel_size is not None:
        gauss_y_label = (
            f"Gaussian fit (w={fit_y['radius']:.2f} px / "
            f"{fit_y['radius'] * pixel_size:.2f} {pixel_unit})"
        )
    else:
        gauss_y_label = f"Gaussian fit (w={fit_y['radius']:.2f} px)"
    axes[1].plot(y_positions, Iy_fit, label=gauss_y_label, color="C1")
    # Indicate second moment radius (2σ) on the y spectrum
    cy = result["cy"]
    y_label = "ISO rᵧ (major)" if not is_major_x else "ISO rᵧ (minor)"
    if pixel_size is not None:
        iso_ry_label = f"{y_label} = {ry_iso:.2f} px ({ry_iso * pixel_size:.2f} {pixel_unit})"
    else:
        iso_ry_label = f"{y_label} = {ry_iso:.2f} px"
    axes[1].axvline(cy - ry_iso, linestyle="--", color="C2", alpha=0.7, label=iso_ry_label)
    axes[1].axvline(cy + ry_iso, linestyle="--", color="C2", alpha=0.7)
    axes[1].set_xlabel("Y position (pixels)")
    axes[1].set_ylabel("Integrated intensity")
    axes[1].set_title("Spectrum along principal y‑axis")
    axes[1].legend()
    if pixel_size is not None:
        def px_to_phys_y(x_value: float) -> float:
            return (x_value - cy) * pixel_size

        def phys_to_px_y(x_value: float) -> float:
            return (x_value / pixel_size) + cy

        secax_y = axes[1].secondary_xaxis('top', functions=(px_to_phys_y, phys_to_px_y))
        secax_y.set_xlabel(f"Y position relative to centre ({pixel_unit})")

    fig.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Plot the input image with ellipses and principal axes
    # ------------------------------------------------------------------
    # Display the original image in a new figure
    fig2, ax_img = plt.subplots(figsize=(6, 6))
    display_cmap = plt.cm.get_cmap("jet", 256).copy()
    display_cmap.set_under("white")
    vmin = 0.0 if data.min() < 0 else None
    ax_img.imshow(data, cmap=display_cmap, origin="upper", vmin=vmin)
    ax_img.set_title("Beam profile with ISO and Gaussian ellipses")
    ax_img.set_xlabel("X (pixels)")
    ax_img.set_ylabel("Y (pixels)")

    from matplotlib.patches import Ellipse
    # ISO ellipse: second moment radii (major/minor) and orientation
    # theta is the principal axis angle in the original image frame
    theta = result["theta"]  # rotation angle in radians
    # width and height are diameters (2*radius)
    iso_width = 2.0 * rx_iso
    iso_height = 2.0 * ry_iso
    iso_ellipse = Ellipse(
        (cx, cy),
        width=iso_width,
        height=iso_height,
        angle=np.degrees(theta),
        edgecolor="white",
        facecolor="none",
        linestyle="-",
        linewidth=2,
        label="ISO ellipse",
    )
    ax_img.add_patch(iso_ellipse)

    # Gaussian ellipse: 1/e^2 radii from Gaussian fits
    gauss_rx = fit_x["radius"]
    gauss_ry = fit_y["radius"]
    gauss_width = 2.0 * gauss_rx
    gauss_height = 2.0 * gauss_ry
    gauss_ellipse = Ellipse(
        (cx, cy),
        width=gauss_width,
        height=gauss_height,
        angle=np.degrees(theta),
        edgecolor="yellow",
        facecolor="none",
        linestyle="--",
        linewidth=2,
        label="Gaussian ellipse",
    )
    ax_img.add_patch(gauss_ellipse)

    # Principal axes lines (use correct major/minor lengths)
    major_len = max(rx_iso, ry_iso)
    minor_len = min(rx_iso, ry_iso)
    # Major axis direction: along theta
    x0 = cx - major_len * np.cos(theta)
    x1 = cx + major_len * np.cos(theta)
    y0 = cy - major_len * np.sin(theta)
    y1 = cy + major_len * np.sin(theta)
    ax_img.plot([x0, x1], [y0, y1], color="cyan", linewidth=1.5, label="Major axis")
    # Minor axis direction: orthogonal to theta
    x0m = cx - minor_len * np.sin(theta)
    x1m = cx + minor_len * np.sin(theta)
    y0m = cy + minor_len * np.cos(theta)
    y1m = cy - minor_len * np.cos(theta)
    ax_img.plot([x0m, x1m], [y0m, y1m], color="magenta", linewidth=1.5, label="Minor axis")

    ax_img.legend(loc="upper right")
    ax_img.set_xlim(0, data.shape[1] - 1)
    # origin='upper' already places (0,0) at the top-left; no ylim flip needed
    ax_img.set_aspect('equal')
    if pixel_size is not None:
        def px_to_phys_img_x(x_value: float) -> float:
            return (x_value - cx) * pixel_size

        def phys_to_px_img_x(x_value: float) -> float:
            return (x_value / pixel_size) + cx

        def px_to_phys_img_y(y_value: float) -> float:
            return (y_value - cy) * pixel_size

        def phys_to_px_img_y(y_value: float) -> float:
            return (y_value / pixel_size) + cy

        secax_img_x = ax_img.secondary_xaxis('top', functions=(px_to_phys_img_x, phys_to_px_img_x))
        secax_img_x.set_xlabel(f"X relative to centre ({pixel_unit})")
        secax_img_y = ax_img.secondary_yaxis('right', functions=(px_to_phys_img_y, phys_to_px_img_y))
        secax_img_y.set_ylabel(f"Y relative to centre ({pixel_unit})")
    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyse a beam-profile BMP image and plot the ISO/gaussian cuts."
        )
    )
    parser.add_argument(
        "image_path",
        help="Path to the BMP image containing the beam profile.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Physical size of one pixel (same units as --pixel-unit).",
    )
    parser.add_argument(
        "--pixel-unit",
        default="µm",
        help="Unit string used when displaying physical radii (default: µm).",
    )
    args = parser.parse_args()
    image_file = Path(args.image_path)
    if not image_file.is_file():
        print(f"Error: {image_file} does not exist or is not a file.")
        sys.exit(1)
    if args.pixel_size is not None and args.pixel_size <= 0:
        print("Error: --pixel-size must be positive if provided.")
        sys.exit(1)
    main(str(image_file), pixel_size=args.pixel_size, pixel_unit=args.pixel_unit)
