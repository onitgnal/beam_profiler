#!/usr/bin/env python3
"""Fit beam caustic curves to extract waist, Rayleigh range, and M^2."""
from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from PIL import Image
from scipy.optimize import OptimizeWarning, curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from beam_analysis import analyze_beam_batch
from roi_utils import trim_to_fwhm_roi

UNIT_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "micrometer": 1e-6,
    "micrometers": 1e-6,
    "nm": 1e-9,
    "nanometer": 1e-9,
    "nanometers": 1e-9,
}

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
    img: Optional[np.ndarray]
    path: Path


@dataclass
class FitResult:
    axis_label: str
    model_label: str
    w0: float  # waist in pixel unit (e.g. µm)
    z0: float  # waist location in mm
    zR: float  # Rayleigh range in mm
    M2: float
    w0_m: float
    z0_m: float
    zR_m: float
    cov: Optional[np.ndarray]


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=float)


def classify_major_minor(val_a: float, val_b: float) -> Tuple[float, float]:
    if np.isnan(val_a) or np.isnan(val_b):
        return val_a, val_b
    return (val_a, val_b) if val_a >= val_b else (val_b, val_a)


def render_beam_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    positives = arr > 0
    rgb = np.ones((*arr.shape, 3), dtype=np.uint8) * 255  # default white
    if positives.any():
        pos_values = arr[positives]
        vmin = pos_values.min()
        vmax = pos_values.max()
        if vmax <= vmin:
            norm = np.zeros_like(pos_values)
        else:
            norm = (pos_values - vmin) / (vmax - vmin)
        colors = colormaps["jet"](norm)[:, :3]
        rgb_vals = (colors * 255).astype(np.uint8)
        rgb[positives] = rgb_vals
    return rgb


def save_measurement_images(measurements: List[Measurement], output_dir: Path) -> None:
    for meas in measurements:
        if meas.img is None:
            continue
        Image.fromarray(render_beam_image(meas.img)).save(
            output_dir / f"{meas.path.stem}_img_for_spec.png"
        )


def gather_measurements(
    directory: Path,
    *,
    roi_scale: float,
    principal_axes_rot: bool,
    rotation_angle: Optional[float],
    z_unit_scale: float,
) -> List[Measurement]:
    paths = sorted(directory.glob("*.bmp"))
    if not paths:
        raise SystemExit(f"No BMP files found in {directory}")

    trimmed_images: List[np.ndarray] = []
    z_positions: List[Optional[float]] = []
    for path in paths:
        image = load_grayscale(path)
        trimmed, _ = trim_to_fwhm_roi(image, scale=roi_scale)
        trimmed_images.append(trimmed)
        z_positions.append(extract_z(path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        results = analyze_beam_batch(
            trimmed_images,
            background_subtraction=True,
            principal_axes_rot=principal_axes_rot,
            rotation_angle=rotation_angle,
        )

    measurements: List[Measurement] = []
    for path, z, trimmed, result in zip(paths, z_positions, trimmed_images, results):
        if z is None:
            continue

        fit_x = result.get("gauss_fit_x")
        fit_y = result.get("gauss_fit_y")
        w_x = float(fit_x["radius"]) if fit_x is not None else float("nan")
        w_y = float(fit_y["radius"]) if fit_y is not None else float("nan")
        g_major, g_minor = classify_major_minor(w_x, w_y)

        rx_iso = float(result["rx_iso"])
        ry_iso = float(result["ry_iso"])
        iso_major, iso_minor = classify_major_minor(rx_iso, ry_iso)

        rotation_deg = float(np.degrees(result["theta"]))
        img = result.get("img_for_spec")
        if img is None:
            img = trimmed

        measurements.append(
            Measurement(
                z_mm=z * z_unit_scale,
                gaussian_major_px=g_major,
                gaussian_minor_px=g_minor,
                iso_major_px=iso_major,
                iso_minor_px=iso_minor,
                rotation_deg=rotation_deg,
                img=img,
                path=path,
            )
        )
    if not measurements:
        raise SystemExit("No usable measurements collected (check filenames contain z positions).")
    return sorted(measurements, key=lambda m: m.z_mm)


def extract_z(path: Path) -> Optional[float]:
    stem = path.stem
    digits = ""
    for ch in stem:
        if ch.isdigit() or ch == ".":
            digits += ch
    try:
        return float(digits)
    except ValueError:
        return None


def caustic_model(
    z_mm: np.ndarray,
    w0_m: float,
    z0_mm: float,
    M2: float,
    wavelength_m: float,
) -> np.ndarray:
    """Theoretical beam radius (in meters) along propagation for given M^2."""
    z_m = z_mm * 1e-3
    z0_m = z0_mm * 1e-3
    denom = math.pi * w0_m**2
    if denom <= 0.0:
        raise ValueError("Invalid waist radius for caustic model.")
    term = (M2 * wavelength_m * (z_m - z0_m)) / denom
    return w0_m * np.sqrt(1.0 + term**2)


def fit_axis(
    z_mm: np.ndarray,
    radii_px: np.ndarray,
    *,
    axis_label: str,
    model_label: str,
    pixel_size: float,
    pixel_unit_scale: float,
    wavelength_m: float,
) -> FitResult:
    mask = np.isfinite(z_mm) & np.isfinite(radii_px)
    z_data = z_mm[mask]
    r_data_px = radii_px[mask]
    if z_data.size < 3:
        raise RuntimeError(f"Not enough data points for {model_label} {axis_label} fit (need >=3).")

    radius_scale_m = pixel_size * pixel_unit_scale
    r_data_m = r_data_px * radius_scale_m

    # Initial guesses
    idx_min = int(np.argmin(r_data_m))
    w0_guess_m = max(r_data_m[idx_min], 1e-9)
    z0_guess_mm = z_data[idx_min]
    M2_guess = 1.0

    try:
        popt, pcov = curve_fit(
            lambda z, w0_m, z0_mm, M2: caustic_model(z, w0_m, z0_mm, M2, wavelength_m),
            z_data,
            r_data_m,
            p0=(w0_guess_m, z0_guess_mm, M2_guess),
            bounds=(
                [1e-9, z_data.min() - 100.0, 0.1],
                [np.inf, z_data.max() + 100.0, 100.0],
            ),
            maxfev=20000,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Fit failed for {model_label} {axis_label}: {exc}")

    w0_m, z0_mm, M2 = popt
    z0_m = z0_mm * 1e-3
    zR_m = math.pi * w0_m**2 / (M2 * wavelength_m)
    zR_mm = zR_m * 1e3
    w0_display = w0_m / pixel_unit_scale

    return FitResult(
        axis_label=axis_label,
        model_label=model_label,
        w0=w0_display,
        z0=z0_mm,
        zR=zR_mm,
        M2=M2,
        w0_m=w0_m,
        z0_m=z0_m,
        zR_m=zR_m,
        cov=pcov,
    )


def plot_fit(
    z_mm: np.ndarray,
    radii: np.ndarray,
    fit: FitResult,
    *,
    pixel_unit: str,
    pixel_size: Optional[float],
    pixel_unit_scale: float,
    wavelength_m: float,
    output_dir: Path,
) -> None:
    mask = np.isfinite(z_mm) & np.isfinite(radii)
    z_data = z_mm[mask]
    r_data = radii[mask]
    z_fit = np.linspace(z_data.min(), z_data.max(), 400)
    r_fit_m = caustic_model(z_fit, fit.w0_m, fit.z0, fit.M2, wavelength_m)

    scale = pixel_size if pixel_size is not None else 1.0
    unit_label = pixel_unit if pixel_size is not None else "px"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(z_data, r_data * scale, label="Measured", color="C0", s=25)
    ax.plot(z_fit, r_fit_m / pixel_unit_scale, label="Fit", color="C1")
    ax.set_xlabel("z position (mm)")
    ax.set_ylabel(f"Radius ({unit_label})")
    title = f"{fit.model_label} {fit.axis_label} caustic fit"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    filename = output_dir / f"caustic_fit_{fit.model_label.lower()}_{fit.axis_label.lower()}.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_overview_model(
    measurements: List[Measurement],
    *,
    model_label: str,
    fit_major: FitResult,
    fit_minor: FitResult,
    pixel_size: float,
    pixel_unit: str,
    pixel_unit_scale: float,
    output_dir: Path,
) -> None:
    if not measurements:
        return

    meas_sorted = sorted(measurements, key=lambda m: m.z_mm)
    z_mm = np.array([m.z_mm for m in meas_sorted])
    if model_label.lower() == "gaussian":
        major_px = np.array([m.gaussian_major_px for m in meas_sorted])
        minor_px = np.array([m.gaussian_minor_px for m in meas_sorted])
    else:
        major_px = np.array([m.iso_major_px for m in meas_sorted])
        minor_px = np.array([m.iso_minor_px for m in meas_sorted])
    rotation_deg = np.array([m.rotation_deg for m in meas_sorted])

    unit_label = pixel_unit
    major_vals = major_px * pixel_size
    minor_vals = minor_px * pixel_size

    z_dense = np.linspace(z_mm.min(), z_mm.max(), 400)
    z_dense_m = z_dense * 1e-3
    major_fit_curve = fit_major.w0_m * np.sqrt(1.0 + ((z_dense_m - fit_major.z0_m) / fit_major.zR_m) ** 2) / pixel_unit_scale
    minor_fit_curve = fit_minor.w0_m * np.sqrt(1.0 + ((z_dense_m - fit_minor.z0_m) / fit_minor.zR_m) ** 2) / pixel_unit_scale

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z_mm, major_vals, label="Major axis", color="C0", marker="o")
    ax.plot(z_mm, minor_vals, label="Minor axis", color="C1", marker="s")
    ax.plot(z_dense, major_fit_curve, color="C0", linestyle="--", label="Major fit")
    ax.plot(z_dense, minor_fit_curve, color="C1", linestyle="--", label="Minor fit")
    ax.set_xlabel("z position (mm)")
    ax.set_ylabel(f"Radius ({unit_label})")

    ax2 = ax.twinx()
    ax2.plot(z_mm, rotation_deg, label="Rotation angle", color="C2", linestyle="-.", marker="^")
    ax2.set_ylabel("Principal axis rotation (deg)")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="best")
    ax.set_title(f"{model_label} radii and rotation vs z")

    # Insets for first, waist, last
    first_meas = meas_sorted[0]
    last_meas = meas_sorted[-1]
    waist_meas = min(meas_sorted, key=lambda m: abs(m.z_mm - fit_major.z0))

    # Largest average radius within one Rayleigh range of waist
    z0 = fit_major.z0
    zR = fit_major.zR
    within = [m for m in meas_sorted if abs(m.z_mm - z0) <= zR]
    pool = within if within else meas_sorted
    if model_label.lower() == "gaussian":
        largest = max(
            pool,
            key=lambda m: (m.gaussian_major_px + m.gaussian_minor_px) * 0.5,
        )
    else:
        largest = max(
            pool,
            key=lambda m: (m.iso_major_px + m.iso_minor_px) * 0.5,
        )

    candidates = [
        (first_meas, "First"),
        (waist_meas, "Waist"),
        (largest, "max in zR"),
        (last_meas, "Last"),
    ]

    # Deduplicate by path to avoid repeated entries
    seen = set()
    unique: List[Tuple[Measurement, str]] = []
    for meas, label in candidates:
        key = meas.path
        if key in seen:
            continue
        seen.add(key)
        unique.append((meas, label))

    unique.sort(key=lambda item: item[0].z_mm)

    inset_positions = [
        (0.05, 0.02),  # 1) lower left
        (0.3, 0.65),  # 2) upper left
        (0.55, 0.65),  # 3) upper right
        (0.80, 0.02),  # 4) lower right
    ]
    lines_info: List[Tuple[float, int, str]] = []
    for idx, ((meas, label), (x0, y0)) in enumerate(zip(unique, inset_positions), start=1):
        axins = inset_axes(ax, width="100%", height="100%", loc="upper left",
                           bbox_to_anchor=(x0, y0, 0.2, 0.2),
                           bbox_transform=ax.transAxes, borderpad=0)
        if meas.img is not None:
            axins.imshow(render_beam_image(meas.img), origin="upper")
        axins.set_title(f"{idx}) {label}\nz={meas.z_mm:.1f} mm", fontsize=8)
        axins.set_xticks([])
        axins.set_yticks([])
        lines_info.append((meas.z_mm, idx, label))

    ymax = ax.get_ylim()[1]
    for z_val, idx, label in lines_info:
        ax.axvline(z_val, color="0.3", linestyle=":", linewidth=1.2)
        tag = f" {idx}" if idx != 3 else f" {idx} ({label})"
        ax.text(z_val, ymax, tag, color="0.2", fontsize=10, va="bottom")

    fig.tight_layout()
    fig.savefig(output_dir / f"caustic_overview_{model_label.lower()}.png", dpi=150)
    plt.close(fig)


def summarize_fit(fit: FitResult, *, pixel_size: Optional[float], pixel_unit: str) -> None:
    unit_label = pixel_unit if pixel_size is not None else "px"
    message = (
        f"{fit.model_label} {fit.axis_label}: "
        f"w0 = {fit.w0:.3f} {unit_label}, "
        f"z0 = {fit.z0:.3f} mm, zR = {fit.zR:.3f} mm"
    )
    if fit.M2 is not None:
        message += f", M^2 = {fit.M2:.3f}"
    print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit beam caustic curves and extract M^2")
    parser.add_argument(
        "directory",
        nargs="?",
        default=Path("750_500_mm_lens"),
        type=Path,
        help="Directory containing beam profile BMP files",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in the specified unit (e.g. 15 for 15 µm). Required for M^2.",
    )
    parser.add_argument(
        "--pixel-unit",
        default="um",
        help="Unit for pixel size (options: um, mm, m, nm).",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=None,
        help="Beam wavelength in the same units as pixel-unit (needed for M^2).",
    )
    parser.add_argument(
        "--wavelength-unit",
        default=None,
        help="Unit for wavelength (defaults to the pixel-unit when omitted).",
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=3.0,
        help="Multiplier applied to FWHM-derived half-width when trimming (default: 3)",
    )
    parser.add_argument(
        "--no-principal-axis-rot",
        dest="principal_axes_rot",
        action="store_false",
        help="Disable rotation of ROI to principal axes before analysis",
    )
    parser.add_argument(
        "--fixed-rotation-deg",
        type=float,
        default=None,
        help="Fix principal-axis rotation angle in degrees; overrides auto estimation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("beam_caustic_fit"),
        help="Directory where fit plots are stored",
    )
    parser.add_argument(
        "--z-unit",
        default="mm",
        help="Unit encoded in filenames (mm, cm, m, um, ...).",
    )
    parser.set_defaults(principal_axes_rot=True)
    args = parser.parse_args()

    if args.pixel_size is None or args.wavelength is None:
        raise SystemExit("This fitting requires both --pixel-size and --wavelength.")

    if args.pixel_size <= 0:
        raise SystemExit("--pixel-size must be positive when provided.")
    pixel_unit_scale = UNIT_TO_METERS.get(args.pixel_unit.lower())
    if pixel_unit_scale is None:
        raise SystemExit(f"Unsupported pixel unit '{args.pixel_unit}'.")

    if args.wavelength <= 0:
        raise SystemExit("--wavelength must be positive when provided.")
    wavelength_unit = args.wavelength_unit or args.pixel_unit
    wavelength_scale = UNIT_TO_METERS.get(wavelength_unit.lower())
    if wavelength_scale is None:
        raise SystemExit(f"Unsupported wavelength unit '{wavelength_unit}'.")
    wavelength_m = args.wavelength * wavelength_scale

    rotation_angle = None if args.fixed_rotation_deg is None else np.deg2rad(args.fixed_rotation_deg)

    z_unit_scale = Z_UNIT_TO_MM.get(args.z_unit.lower())
    if z_unit_scale is None:
        raise SystemExit(f"Unsupported z unit '{args.z_unit}'.")

    measurements = gather_measurements(
        args.directory.resolve(),
        roi_scale=args.roi_scale,
        principal_axes_rot=args.principal_axes_rot,
        rotation_angle=rotation_angle,
        z_unit_scale=z_unit_scale,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_measurement_images(measurements, output_dir)

    z_mm = np.array([m.z_mm for m in measurements])
    gaussian_major_px = np.array([m.gaussian_major_px for m in measurements])
    gaussian_minor_px = np.array([m.gaussian_minor_px for m in measurements])
    iso_major_px = np.array([m.iso_major_px for m in measurements])
    iso_minor_px = np.array([m.iso_minor_px for m in measurements])

    fits: List[FitResult] = []
    for model_label, radii_major_px, radii_minor_px in (
        ("Gaussian", gaussian_major_px, gaussian_minor_px),
        ("ISO", iso_major_px, iso_minor_px),
    ):
        try:
            fit_major = fit_axis(
                z_mm,
                radii_major_px,
                axis_label="major",
                model_label=model_label,
                pixel_size=args.pixel_size,
                pixel_unit_scale=pixel_unit_scale,
                wavelength_m=wavelength_m,
            )
            fits.append(fit_major)
            plot_fit(
                z_mm,
                radii_major_px,
                fit_major,
                pixel_unit=args.pixel_unit,
                pixel_size=args.pixel_size,
                pixel_unit_scale=pixel_unit_scale,
                wavelength_m=wavelength_m,
                output_dir=output_dir,
            )
        except RuntimeError as exc:
            print(exc)

        try:
            fit_minor = fit_axis(
                z_mm,
                radii_minor_px,
                axis_label="minor",
                model_label=model_label,
                pixel_size=args.pixel_size,
                pixel_unit_scale=pixel_unit_scale,
                wavelength_m=wavelength_m,
            )
            fits.append(fit_minor)
            plot_fit(
                z_mm,
                radii_minor_px,
                fit_minor,
                pixel_unit=args.pixel_unit,
                pixel_size=args.pixel_size,
                pixel_unit_scale=pixel_unit_scale,
                wavelength_m=wavelength_m,
                output_dir=output_dir,
            )
        except RuntimeError as exc:
            print(exc)

    fit_lookup = {(fit.model_label, fit.axis_label): fit for fit in fits}
    for model in ("Gaussian", "ISO"):
        fit_major = fit_lookup.get((model, "major"))
        fit_minor = fit_lookup.get((model, "minor"))
        if fit_major is None or fit_minor is None:
            continue
        plot_overview_model(
            measurements,
            model_label=model,
            fit_major=fit_major,
            fit_minor=fit_minor,
            pixel_size=args.pixel_size,
            pixel_unit=args.pixel_unit,
            pixel_unit_scale=pixel_unit_scale,
            output_dir=output_dir,
        )

    print("\nFit results:")
    for fit in fits:
        summarize_fit(fit, pixel_size=args.pixel_size, pixel_unit=args.pixel_unit)


if __name__ == "__main__":
    main()
