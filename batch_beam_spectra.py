#!/usr/bin/env python3
"""Generate per-image beam analysis plots akin to beam_plot_example."""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from PIL import Image
from scipy.optimize import OptimizeWarning

from beam_analysis import analyze_beam_batch
from roi_utils import trim_to_fwhm_roi
from dataclasses import dataclass

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
class Measurement:
    z_mm: float
    gaussian_major_px: float
    gaussian_minor_px: float
    iso_major_px: float
    iso_minor_px: float
    rotation_deg: float


def extract_z(path: Path) -> Optional[float]:
    match = Z_PATTERN.search(path.stem)
    return float(match.group(1)) if match else None


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=float)


def gaussian_curve(x: np.ndarray, params: dict) -> np.ndarray:
    return params["amplitude"] * np.exp(-2.0 * ((x - params["centre"]) / params["radius"]) ** 2)


def physical_label(value_px: float, pixel_size_um: Optional[float], unit: str) -> str:
    if pixel_size_um is None:
        return f"{value_px:.2f} px"
    return f"{value_px:.2f} px / {value_px * pixel_size_um:.2f} {unit}"


def iso_label(prefix: str, value_px: float, pixel_size_um: Optional[float], unit: str) -> str:
    if pixel_size_um is None:
        return f"{prefix} = {value_px:.2f} px"
    return f"{prefix} = {value_px:.2f} px ({value_px * pixel_size_um:.2f} {unit})"


def add_secondary_axis(axis, centre: float, pixel_size_um: float, unit: str, orientation: str) -> None:
    def px_to_phys(value: float) -> float:
        return (value - centre) * pixel_size_um

    def phys_to_px(value: float) -> float:
        return (value / pixel_size_um) + centre

    if orientation == "x":
        sec = axis.secondary_xaxis('top', functions=(px_to_phys, phys_to_px))
        sec.set_xlabel(f"Position relative to centre ({unit})")
    else:
        sec = axis.secondary_yaxis('right', functions=(px_to_phys, phys_to_px))
        sec.set_ylabel(f"Position relative to centre ({unit})")


def classify_major_minor(value_a: float, value_b: float) -> tuple[float, float]:
    if np.isnan(value_a) or np.isnan(value_b):
        return value_a, value_b
    return (value_a, value_b) if value_a >= value_b else (value_b, value_a)


def plot_result(
    image_path: Path,
    result: dict,
    *,
    pixel_size_um: Optional[float],
    pixel_unit: str,
    output_dir: Path,
) -> None:
    x_positions, Ix = result["Ix_spectrum"]
    y_positions, Iy = result["Iy_spectrum"]
    fit_x = result.get("gauss_fit_x")
    fit_y = result.get("gauss_fit_y")

    cx = float(result["cx"])
    cy = float(result["cy"])
    rx_iso = float(result["rx_iso"])
    ry_iso = float(result["ry_iso"])
    theta_rad = float(result["theta"])
    theta_deg = np.degrees(theta_rad)

    processed_img = result.get("img_for_spec")
    crop_origin = result.get("img_for_spec_origin", (0.0, 0.0))
    crop_y0, crop_x0 = map(float, crop_origin)
    cx_local = cx - crop_x0
    cy_local = cy - crop_y0

    Ix_fit = gaussian_curve(x_positions, fit_x) if fit_x is not None else None
    Iy_fit = gaussian_curve(y_positions, fit_y) if fit_y is not None else None

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2, 1, 1])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1:])
    ax_y = fig.add_subplot(gs[1, 1:])

    if processed_img is not None and processed_img.size:
        im = ax_img.imshow(processed_img, cmap="jet", origin="upper")

        length = 0.5 * max(processed_img.shape)

        def draw_axis(angle_rad: float, color: str, label: str) -> None:
            dx = length * np.cos(angle_rad)
            dy = length * np.sin(angle_rad)
            ax_img.plot(
                [cx_local - dx, cx_local + dx],
                [cy_local - dy, cy_local + dy],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=label,
            )

        draw_axis(theta_rad, "C0", "Principal x-axis")
        draw_axis(theta_rad + np.pi / 2.0, "C1", "Principal y-axis")

        ellipse = Ellipse(
            (cx_local, cy_local),
            width=2 * rx_iso,
            height=2 * ry_iso,
            angle=theta_deg,
            fill=False,
            color="C1",
            linewidth=1.5,
            alpha=0.8,
        )
        ax_img.add_patch(ellipse)
        ax_img.set_title("Processed beam ROI (background-subtracted)")
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.02)
    else:
        ax_img.text(0.5, 0.5, "No processed image available", ha="center", va="center")
        ax_img.set_axis_off()

    ax_x.plot(x_positions, Ix, label="Ix (integrated)", color="C0")
    if Ix_fit is not None:
        gauss_label = f"Gaussian fit (w={physical_label(fit_x['radius'], pixel_size_um, pixel_unit)})"
        ax_x.plot(x_positions, Ix_fit, label=gauss_label, color="C1")
    iso_rx_label = iso_label("ISO r_x", rx_iso, pixel_size_um, pixel_unit)
    ax_x.axvline(cx - rx_iso, linestyle="--", color="C2", alpha=0.7, label=iso_rx_label)
    ax_x.axvline(cx + rx_iso, linestyle="--", color="C2", alpha=0.7)
    ax_x.set_xlabel("X position (pixels)")
    ax_x.set_ylabel("Integrated intensity")
    ax_x.legend()
    if pixel_size_um is not None:
        add_secondary_axis(ax_x, cx, pixel_size_um, pixel_unit, orientation="x")

    ax_y.plot(y_positions, Iy, label="Iy (integrated)", color="C0")
    if Iy_fit is not None:
        gauss_label = f"Gaussian fit (w={physical_label(fit_y['radius'], pixel_size_um, pixel_unit)})"
        ax_y.plot(y_positions, Iy_fit, label=gauss_label, color="C1")
    iso_ry_label = iso_label("ISO r_y", ry_iso, pixel_size_um, pixel_unit)
    ax_y.axvline(cy - ry_iso, linestyle="--", color="C2", alpha=0.7, label=iso_ry_label)
    ax_y.axvline(cy + ry_iso, linestyle="--", color="C2", alpha=0.7)
    ax_y.set_xlabel("Y position (pixels)")
    ax_y.set_ylabel("Integrated intensity")
    ax_y.legend()
    if pixel_size_um is not None:
        add_secondary_axis(ax_y, cy, pixel_size_um, pixel_unit, orientation="y")

    z_mm = extract_z(image_path)
    if z_mm is not None:
        title = f"{image_path.name} (z = {z_mm:.2f} mm) — rotation {theta_deg:.2f} deg"
    else:
        title = f"{image_path.name} — rotation {theta_deg:.2f} deg"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = output_dir / f"{image_path.stem}_analysis.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)




def plot_caustic_curves(
    measurements: List[Measurement],
    *,
    pixel_size_um: Optional[float],
    pixel_unit: str,
    output_dir: Path,
) -> None:
    measurements = sorted(measurements, key=lambda m: m.z_mm)
    z = np.array([m.z_mm for m in measurements])
    gaussian_major = np.array([m.gaussian_major_px for m in measurements])
    gaussian_minor = np.array([m.gaussian_minor_px for m in measurements])
    iso_major = np.array([m.iso_major_px for m in measurements])
    iso_minor = np.array([m.iso_minor_px for m in measurements])
    rotation = np.array([m.rotation_deg for m in measurements])

    unit = pixel_unit if pixel_size_um is not None else 'px'
    scale = pixel_size_um if pixel_size_um is not None else 1.0

    def _plot(major: np.ndarray, minor: np.ndarray, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(z, major * scale, label='Major axis', color='C0', marker='o')
        ax.plot(z, minor * scale, label='Minor axis', color='C1', marker='s')
        ax.set_xlabel('z position (mm)')
        ax.set_ylabel(f'Radius ({unit})')

        ax2 = ax.twinx()
        ax2.plot(z, rotation, label='Rotation angle', color='C2', linestyle='--')
        ax2.set_ylabel('Principal axis rotation (deg)')

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc='best')
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    _plot(gaussian_major, gaussian_minor, 'Gaussian radii vs z', 'caustic_gaussian.png')
    _plot(iso_major, iso_minor, 'ISO second-moment radii vs z', 'caustic_iso.png')
def analyze_directory(
    directory: Path,
    *,
    pixel_size_um: Optional[float],
    pixel_unit: str,
    output_dir: Path,
    roi_scale: float,
    principal_axes_rot: bool,
    rotation_angle: Optional[float],
    z_unit_scale: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(directory.glob("*.bmp"))
    if not image_paths:
        raise SystemExit(f"No BMP files found in {directory}")

    measurements: List[Measurement] = []
    trimmed_images: List[np.ndarray] = []
    z_positions: List[Optional[float]] = []
    for image_path in image_paths:
        image = load_grayscale(image_path)
        trimmed, _ = trim_to_fwhm_roi(image, scale=roi_scale)
        trimmed_images.append(trimmed)
        z_positions.append(extract_z(image_path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        results = analyze_beam_batch(
            trimmed_images,
            background_subtraction=True,
            principal_axes_rot=principal_axes_rot,
            rotation_angle=rotation_angle,
        )

    for image_path, z_mm_raw, result in zip(image_paths, z_positions, results):
        plot_result(
            image_path,
            result,
            pixel_size_um=pixel_size_um,
            pixel_unit=pixel_unit,
            output_dir=output_dir,
        )
        fit_x = result.get("gauss_fit_x")
        fit_y = result.get("gauss_fit_y")
        w_x = float(fit_x["radius"]) if fit_x is not None else float("nan")
        w_y = float(fit_y["radius"]) if fit_y is not None else float("nan")
        g_major, g_minor = classify_major_minor(w_x, w_y)
        rx_iso = float(result["rx_iso"])
        ry_iso = float(result["ry_iso"])
        iso_major, iso_minor = classify_major_minor(rx_iso, ry_iso)
        theta_deg = float(np.degrees(result["theta"]))
        if z_mm_raw is not None:
            measurements.append(
                Measurement(
                    z_mm=z_mm_raw * z_unit_scale,
                    gaussian_major_px=g_major,
                    gaussian_minor_px=g_minor,
                    iso_major_px=iso_major,
                    iso_minor_px=iso_minor,
                    rotation_deg=theta_deg,
                )
            )
        radius_text_x = (
            physical_label(w_x, pixel_size_um, pixel_unit) if fit_x is not None else "NA"
        )
        radius_text_y = (
            physical_label(w_y, pixel_size_um, pixel_unit) if fit_y is not None else "NA"
        )
        print(
            f"Processed {image_path.name}: Gaussian radius x = {radius_text_x}, "
            f"Gaussian radius y = {radius_text_y}"
        )

    if measurements:
        plot_caustic_curves(
            measurements,
            pixel_size_um=pixel_size_um,
            pixel_unit=pixel_unit,
            output_dir=output_dir,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create beam analysis plots for each profile")
    parser.add_argument(
        "directory",
        nargs="?",
        default=Path("750_500_mm_lens"),
        type=Path,
        help="Directory containing beam profile BMP files",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("beam_profile_plots"),
        type=Path,
        help="Directory where analysis figures are saved",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=15.0,
        help="Pixel size in micrometers (set to 0 to disable physical units)",
    )
    parser.add_argument(
        "--pixel-unit",
        default="um",
        help="Unit label for physical distances when pixel-size is provided",
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=3.0,
        help="Multiplier applied to FWHM-derived half-width when trimming (default: 3)",
    )
    parser.add_argument(
        "--z-unit",
        default="mm",
        help="Unit encoded in filenames (mm, cm, m, um, ...).",
    )
    parser.add_argument(
        "--no-principal-axis-rot",
        dest="principal_axes_rot",
        action="store_false",
        help="Disable rotation of the ROI to principal axes before spectra (default: enabled)",
    )
    parser.add_argument(
        "--fixed-rotation-deg",
        type=float,
        default=None,
        help="Fix the principal-axis rotation angle in degrees; overrides auto estimation",
    )
    parser.set_defaults(principal_axes_rot=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pixel_size_um = None if args.pixel_size <= 0 else args.pixel_size
    rotation_angle = None if args.fixed_rotation_deg is None else np.deg2rad(args.fixed_rotation_deg)
    z_unit_scale = Z_UNIT_TO_MM.get(args.z_unit.lower())
    if z_unit_scale is None:
        raise SystemExit(f"Unsupported z unit '{args.z_unit}'.")

    analyze_directory(
        args.directory.resolve(),
        pixel_size_um=pixel_size_um,
        pixel_unit=args.pixel_unit,
        output_dir=args.output_dir.resolve(),
        roi_scale=args.roi_scale,
        principal_axes_rot=args.principal_axes_rot,
        rotation_angle=rotation_angle,
        z_unit_scale=z_unit_scale,
    )


if __name__ == "__main__":
    main()
