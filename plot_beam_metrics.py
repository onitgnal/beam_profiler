#!/usr/bin/env python3
"""Generate beam radius plots across propagation distance."""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import OptimizeWarning, curve_fit

from beam_analysis import analyze_beam_batch
from roi_utils import trim_to_fwhm_roi

PIXEL_SIZE_MICROMETER = 15.0
Z_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")
Z_UNIT_TO_MM = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "um": 0.001,
    "µm": 0.001,
    "micrometer": 0.001,
    "micrometers": 0.001,
}


@dataclass
class BeamMeasurement:
    z_mm: float
    gaussian_major_um: float
    gaussian_minor_um: float
    iso_major_um: float
    iso_minor_um: float
    rotation_deg: float


def extract_z(path: Path) -> float:
    match = Z_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Could not extract z position from filename '{path.name}'")
    return float(match.group(1))


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=float)


def classify_major_minor(value_a: float, value_b: float) -> tuple[float, float]:
    if value_a >= value_b:
        return value_a, value_b
    return value_b, value_a


def discover_images(directory: Path) -> List[Path]:
    candidates = sorted(directory.glob("*.bmp"))
    if not candidates:
        raise SystemExit(f"No BMP files found in {directory}")
    return candidates


def analyze_directory(directory: Path, *, roi_scale: float, z_unit_scale: float) -> List[BeamMeasurement]:
    paths = discover_images(directory)

    trimmed_images: List[np.ndarray] = []
    z_positions: List[float] = []
    for path in paths:
        image = load_image(path)
        trimmed, _ = trim_to_fwhm_roi(image, scale=roi_scale)
        trimmed_images.append(trimmed)
        z_positions.append(extract_z(path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        results = analyze_beam_batch(trimmed_images, background_subtraction=True)

    measurements: List[BeamMeasurement] = []
    for z_mm, result in zip(z_positions, results):
        theta_deg = math.degrees(result["theta"])

        rx_iso = float(result["rx_iso"])
        ry_iso = float(result["ry_iso"])
        iso_major_px, iso_minor_px = classify_major_minor(rx_iso, ry_iso)

        fit_x = result.get("gauss_fit_x")
        fit_y = result.get("gauss_fit_y")

        w_x = float(fit_x["radius"]) if fit_x is not None else float("nan")
        w_y = float(fit_y["radius"]) if fit_y is not None else float("nan")
        gaussian_major_px, gaussian_minor_px = classify_major_minor(w_x, w_y)

        measurements.append(
            BeamMeasurement(
                z_mm=z_mm * z_unit_scale,
                gaussian_major_um=gaussian_major_px * PIXEL_SIZE_MICROMETER,
                gaussian_minor_um=gaussian_minor_px * PIXEL_SIZE_MICROMETER,
                iso_major_um=iso_major_px * PIXEL_SIZE_MICROMETER,
                iso_minor_um=iso_minor_px * PIXEL_SIZE_MICROMETER,
                rotation_deg=theta_deg,
            )
        )

    measurements.sort(key=lambda item: item.z_mm)
    return measurements


def _fit_caustic(z: np.ndarray, radii: np.ndarray) -> Optional[Tuple[float, float, float]]:
    def _model(z_vals: np.ndarray, w0: float, z0: float, zR: float) -> np.ndarray:
        return w0 * np.sqrt(1.0 + ((z_vals - z0) / zR) ** 2)

    mask = np.isfinite(z) & np.isfinite(radii)
    if mask.sum() < 3:
        return None

    z_data = z[mask]
    r_data = radii[mask]

    idx_min = int(np.argmin(r_data))
    w0_guess = r_data[idx_min]
    z0_guess = z_data[idx_min]
    z_span = max(z_data.max() - z_data.min(), 1e-6)
    zR_guess = max(z_span / 2.0, 1e-3)

    try:
        popt, _ = curve_fit(
            _model,
            z_data,
            r_data,
            p0=(w0_guess, z0_guess, zR_guess),
            bounds=(
                [0.0, z_data.min() - 10 * z_span, 1e-6],
                [np.inf, z_data.max() + 10 * z_span, np.inf],
            ),
        )
    except Exception:  # noqa: BLE001
        return None
    return tuple(popt)


def build_plot(
    z_positions: np.ndarray,
    major: np.ndarray,
    minor: np.ndarray,
    rotation: np.ndarray,
    *,
    title: str,
    filename: Path,
    major_fit: Optional[Tuple[float, float, float]] = None,
    minor_fit: Optional[Tuple[float, float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(z_positions, major, label="Major axis", color="C0", marker="o")
    ax.plot(z_positions, minor, label="Minor axis", color="C1", marker="s")

    if major_fit is not None or minor_fit is not None:
        z_dense = np.linspace(z_positions.min(), z_positions.max(), 400)
        def _model(z_vals: np.ndarray, w0: float, z0: float, zR: float) -> np.ndarray:
            return w0 * np.sqrt(1.0 + ((z_vals - z0) / zR) ** 2)
        if major_fit is not None:
            ax.plot(
                z_dense,
                _model(z_dense, *major_fit),
                color="C0",
                linestyle="--",
                label=(
                    f"Major fit (w0={major_fit[0]:.1f} µm, z0={major_fit[1]:.1f} mm, "
                    f"zR={major_fit[2]:.1f} mm)"
                ),
            )
        if minor_fit is not None:
            ax.plot(
                z_dense,
                _model(z_dense, *minor_fit),
                color="C1",
                linestyle="--",
                label=(
                    f"Minor fit (w0={minor_fit[0]:.1f} µm, z0={minor_fit[1]:.1f} mm, "
                    f"zR={minor_fit[2]:.1f} mm)"
                ),
            )

    ax.set_xlabel("z position (mm)")
    ax.set_ylabel("Radius (µm)")

    ax2 = ax.twinx()
    ax2.plot(z_positions, rotation, label="Rotation angle", color="C2", linestyle="--")
    ax2.set_ylabel("Principal axis rotation (deg)")

    handles: List = []
    labels: List[str] = []
    for axis in (ax, ax2):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    ax.legend(handles, labels, loc="best")

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot beam radii vs z position")
    parser.add_argument(
        "directory",
        nargs="?",
        default=Path("750_500_mm_lens"),
        type=Path,
        help="Directory containing beam profile BMP files",
    )
    parser.add_argument(
        "--output-prefix",
        default="beam_metrics",
        help="Prefix for output plot filenames",
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=3.0,
        help="Multiplier applied to the FWHM-derived half-width when trimming (default: 3)",
    )
    parser.add_argument(
        "--z-unit",
        default="mm",
        help="Unit encoded in filenames (mm, cm, m, um, ...).",
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    z_unit_scale = Z_UNIT_TO_MM.get(args.z_unit.lower())
    if z_unit_scale is None:
        raise SystemExit(f"Unsupported z unit '{args.z_unit}'.")

    measurements = analyze_directory(directory, roi_scale=args.roi_scale, z_unit_scale=z_unit_scale)

    z = np.array([m.z_mm for m in measurements])
    gaussian_major = np.array([m.gaussian_major_um for m in measurements])
    gaussian_minor = np.array([m.gaussian_minor_um for m in measurements])
    iso_major = np.array([m.iso_major_um for m in measurements])
    iso_minor = np.array([m.iso_minor_um for m in measurements])
    rotation = np.array([m.rotation_deg for m in measurements])

    gauss_major_fit = _fit_caustic(z, gaussian_major)
    gauss_minor_fit = _fit_caustic(z, gaussian_minor)
    iso_major_fit = _fit_caustic(z, iso_major)
    iso_minor_fit = _fit_caustic(z, iso_minor)

    base = Path(args.output_prefix)
    build_plot(
        z,
        gaussian_major,
        gaussian_minor,
        rotation,
        title="Gaussian fit radii vs z",
        filename=base.with_name(f"{base.name}_gaussian.png"),
        major_fit=gauss_major_fit,
        minor_fit=gauss_minor_fit,
    )
    build_plot(
        z,
        iso_major,
        iso_minor,
        rotation,
        title="ISO second-moment radii vs z",
        filename=base.with_name(f"{base.name}_iso.png"),
        major_fit=iso_major_fit,
        minor_fit=iso_minor_fit,
    )


if __name__ == "__main__":
    main()
