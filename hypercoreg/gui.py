"""
Graphical user interface for HyperCoreg.

This module provides the tkinter-based GUI for interactive coregistration.
"""

import os
import sys
import logging
import time
import queue
import threading
import traceback
import tkinter as tk
from collections.abc import Mapping
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Any, Optional, List

from hypercoreg._version import __version__
from hypercoreg.config import (
    DEFAULT_CONFIG,
    S2_BANDS,
    DEFAULT_S2_BAND,
)
from hypercoreg.logging_config import setup_logging, log_section_header
from hypercoreg.result_status import describe_result_failure
from hypercoreg.utils import detect_hyp_type

logger = logging.getLogger("COREG_PROCESSING")
GUI_PROMPT_WAIT_TIMEOUT_S = 120
GUI_METADATA_EXTENSION_LEVEL = DEFAULT_CONFIG["metadata_extension_level"]
GUI_METADATA_LABEL_PRECISION = int(DEFAULT_CONFIG["metadata_label_precision"])
GUI_METADATA_EXTENSION_CHOICES = ("none", "stats")


def _normalize_gui_metadata_level(extension_level: Any) -> str:
    """Normalize the simplified GUI metadata level control."""
    ext = str(extension_level or "").strip().lower()
    if ext not in GUI_METADATA_EXTENSION_CHOICES:
        ext = str(DEFAULT_CONFIG["metadata_extension_level"]).strip().lower()
    return ext


def _resolve_gui_metadata_runtime_settings(extension_level: Any) -> Dict[str, Any]:
    """Map the simplified GUI metadata level onto the full runtime config surface."""
    ext = _normalize_gui_metadata_level(extension_level)
    settings = {
        "metadata_extension_level": ext,
        "metadata_stats_mode": str(DEFAULT_CONFIG["metadata_stats_mode"]).strip().lower(),
        "metadata_stats_sample_windows": int(DEFAULT_CONFIG["metadata_stats_sample_windows"]),
        "metadata_stats_seed": int(DEFAULT_CONFIG["metadata_stats_seed"]),
        "metadata_histogram_buckets": int(DEFAULT_CONFIG["metadata_histogram_buckets"]),
        "metadata_label_precision": int(DEFAULT_CONFIG["metadata_label_precision"]),
        "validation_max_windows": int(DEFAULT_CONFIG["validation_max_windows"]),
        "enmap_metadata_stats_mode": str(DEFAULT_CONFIG["enmap_metadata_stats_mode"]).strip().lower(),
    }
    if ext == "none":
        settings["metadata_stats_mode"] = "none"
        settings["enmap_metadata_stats_mode"] = "none"
    return settings


def _build_gui_runtime_config(
    *,
    input_path: str,
    output_dir: str,
    batch_mode: bool,
    config_vars: Dict[str, Any],
    strict_metadata: bool,
    metadata_extension_level: str,
    metadata_stats_mode: str,
    metadata_stats_sample_windows: int,
    metadata_stats_seed: int,
    metadata_histogram_buckets: int,
    metadata_label_precision: int,
    validation_max_windows: int,
) -> Dict[str, Any]:
    """Build runtime config from GUI state using shared defaults as source of truth."""
    return {
        'input_path': input_path,
        'output_dir': output_dir,
        'batch_mode': batch_mode,
        'days_window': config_vars['days_window'].get(),
        'min_overlap': config_vars['min_overlap'].get(),
        'max_cloud': config_vars['max_cloud'].get(),
        'max_input_cloud_cover': config_vars['max_input_cloud_cover'].get(),
        'min_accuracy': config_vars['min_accuracy'].get(),
        'max_displacement': config_vars['max_displacement'].get(),
        'residual_threshold': config_vars['residual_threshold'].get(),
        'min_tie_points': config_vars['min_tie_points'].get(),
        'max_s2_candidates': config_vars['max_s2_candidates'].get(),
        'residual_mad_factor': config_vars['residual_mad_factor'].get(),
        's2_ref_band': config_vars['s2_ref_band'].get(),
        's2_stack_cache': DEFAULT_CONFIG['s2_stack_cache'],
        's2_cache_dir': DEFAULT_CONFIG['s2_cache_dir'],
        'prefer_fixed_band_pairs': DEFAULT_CONFIG['prefer_fixed_band_pairs'],
        'fixed_band_pairs_by_sensor': DEFAULT_CONFIG['fixed_band_pairs_by_sensor'],
        'bandpair_wavelength_window_nm': DEFAULT_CONFIG['bandpair_wavelength_window_nm'],
        's2_band_subset_by_branch': DEFAULT_CONFIG['s2_band_subset_by_branch'],
        'local_tiepoint_early_stop': DEFAULT_CONFIG['local_tiepoint_early_stop'],
        'cache_hs_narrowbands': DEFAULT_CONFIG['cache_hs_narrowbands'],
        'hs_narrowband_cache_dir': DEFAULT_CONFIG['hs_narrowband_cache_dir'],
        'min_band_support': DEFAULT_CONFIG['min_band_support'],
        'allow_single_band_fallback': DEFAULT_CONFIG['allow_single_band_fallback'],
        'consensus_group_rounding_px': DEFAULT_CONFIG['consensus_group_rounding_px'],
        'spatial_stratification_grid_rows': DEFAULT_CONFIG['spatial_stratification_grid_rows'],
        'spatial_stratification_grid_cols': DEFAULT_CONFIG['spatial_stratification_grid_cols'],
        'max_points_per_cell': DEFAULT_CONFIG['max_points_per_cell'],
        'preferred_polynomial_order': DEFAULT_CONFIG['preferred_polynomial_order'],
        'auto_downgrade_polynomial_order': DEFAULT_CONFIG['auto_downgrade_polynomial_order'],
        'min_gcps_order2': DEFAULT_CONFIG['min_gcps_order2'],
        'min_cells_order2': DEFAULT_CONFIG['min_cells_order2'],
        'local_coreg_grid_res': DEFAULT_CONFIG['local_coreg_grid_res'],
        'local_coreg_window_size': DEFAULT_CONFIG['local_coreg_window_size'],
        'local_coreg_tieP_filter_level': DEFAULT_CONFIG['local_coreg_tieP_filter_level'],
        'local_coreg_max_iter': DEFAULT_CONFIG['local_coreg_max_iter'],
        'global_coreg_profiles_by_sensor': DEFAULT_CONFIG['global_coreg_profiles_by_sensor'],
        'local_max_shift_by_sensor': DEFAULT_CONFIG['local_max_shift_by_sensor'],
        'global_coreg_attempt_ladder': DEFAULT_CONFIG['global_coreg_attempt_ladder'],
        'postwarp_phasecorr_check': DEFAULT_CONFIG['postwarp_phasecorr_check'],
        'postwarp_phasecorr_warn_threshold_px': DEFAULT_CONFIG['postwarp_phasecorr_warn_threshold_px'],
        'postwarp_phasecorr_reject_threshold_px': DEFAULT_CONFIG['postwarp_phasecorr_reject_threshold_px'],
        'postwarp_phasecorr_reject_bad': DEFAULT_CONFIG['postwarp_phasecorr_reject_bad'],
        'postwarp_phasecorr_max_dim': DEFAULT_CONFIG['postwarp_phasecorr_max_dim'],
        'use_geolocation_mesh_affine': DEFAULT_CONFIG['use_geolocation_mesh_affine'],
        'geolocation_mesh_stride': DEFAULT_CONFIG['geolocation_mesh_stride'],
        'save_pre': config_vars['save_pre'].get(),
        'gen_tiepoint_pngs': config_vars['gen_tiepoint_pngs'].get(),
        'save_displacement_vectors': config_vars['save_displacement_vectors'].get(),
        'use_inmemory': config_vars['use_inmemory'].get(),
        'keep_temp_files': config_vars['keep_temp_files'].get(),
        'save_pan': config_vars['save_pan'].get(),
        'save_quality_mask': config_vars['save_quality_mask'].get(),
        'remove_detector_overlap_bands': config_vars['remove_detector_overlap_bands'].get(),
        'prisma_radiometric_mode': str(config_vars['prisma_radiometric_mode'].get()).strip().lower(),
        'normalization_mode': str(config_vars['normalization_mode'].get()).strip().lower(),
        'norm_p_low': DEFAULT_CONFIG['norm_p_low'],
        'norm_p_high': DEFAULT_CONFIG['norm_p_high'],
        'norm_clip': DEFAULT_CONFIG['norm_clip'],
        'norm_eps': DEFAULT_CONFIG['norm_eps'],
        'norm_min_valid_pixels': DEFAULT_CONFIG['norm_min_valid_pixels'],
        'norm_reservoir_size': DEFAULT_CONFIG['norm_reservoir_size'],
        'norm_seed': DEFAULT_CONFIG['norm_seed'],
        'norm_tile_size': DEFAULT_CONFIG['norm_tile_size'],
        'build_overviews': config_vars['build_overviews'].get(),
        'strict_metadata': bool(strict_metadata),
        'metadata_extension_level': metadata_extension_level,
        'metadata_stats_mode': metadata_stats_mode,
        'enmap_metadata_stats_mode': str(config_vars['enmap_metadata_stats_mode'].get()).strip().lower(),
        'metadata_stats_sample_windows': int(metadata_stats_sample_windows),
        'metadata_stats_seed': int(metadata_stats_seed),
        'metadata_histogram_buckets': metadata_histogram_buckets,
        'metadata_label_precision': int(metadata_label_precision),
        'validation_max_windows': int(validation_max_windows),
        'allow_gui_prompt': True,
        'defer_temp_cleanup_gui': True,
        'timing_logs': True,
        'batch_workers': DEFAULT_CONFIG['batch_workers'],
    }


class _PipelineProgressWindow:
    """Lightweight popup showing pipeline stage and scene progress."""

    def __init__(self, is_batch: bool = False) -> None:
        self.is_batch = bool(is_batch)
        self._closed = False
        self._timer_job = None
        self._overall_start = time.monotonic()
        self._scene_start = self._overall_start
        self._current_scene_idx = 1

        self.root = tk.Tk()
        self.root.title("HyperCoreg - Processing")
        self.root.resizable(False, False)
        width = 760
        height = 380 if self.is_batch else 360
        self.root.geometry(f"{width}x{height}")

        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)

        container = tk.Frame(self.root, padx=12, pady=12)
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text="Coregistration in progress",
            font=("Arial", 11, "bold"),
            anchor="w"
        ).pack(fill="x", pady=(0, 8))

        self.stage_var = tk.StringVar(value="Stage: Initializing")
        self.scene_var = tk.StringVar(value="Scene: 1/1")
        self.status_var = tk.StringVar(value="Status: Running")
        self.detail_var = tk.StringVar(value="")
        self.scene_elapsed_var = tk.StringVar(value="Scene elapsed: 00:00:00")
        self.overall_elapsed_var = tk.StringVar(value="Overall elapsed: 00:00:00")
        self.download_var = tk.StringVar(value="")

        tk.Label(container, textvariable=self.stage_var, anchor="w").pack(fill="x")
        tk.Label(container, textvariable=self.scene_var, anchor="w").pack(fill="x", pady=(2, 0))
        tk.Label(container, textvariable=self.status_var, anchor="w").pack(fill="x", pady=(2, 0))
        tk.Label(container, textvariable=self.detail_var, anchor="w", fg="gray").pack(fill="x", pady=(2, 0))
        tk.Label(container, textvariable=self.scene_elapsed_var, anchor="w").pack(fill="x", pady=(2, 0))
        if self.is_batch:
            tk.Label(container, textvariable=self.overall_elapsed_var, anchor="w").pack(fill="x", pady=(2, 0))
        tk.Label(container, textvariable=self.download_var, anchor="w", fg="gray").pack(fill="x", pady=(2, 0))

        self.progress = ttk.Progressbar(container, mode="indeterminate")
        self.progress.pack(fill="x", pady=(10, 0))
        self.progress.start(10)

        self._schedule_timer_tick()

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _update_timer_labels(self) -> None:
        now = time.monotonic()
        self.scene_elapsed_var.set(f"Scene elapsed: {self._fmt_elapsed(now - self._scene_start)}")
        self.overall_elapsed_var.set(f"Overall elapsed: {self._fmt_elapsed(now - self._overall_start)}")

    def _schedule_timer_tick(self) -> None:
        if self._closed:
            return
        self._update_timer_labels()
        try:
            self._timer_job = self.root.after(1000, self._schedule_timer_tick)
        except tk.TclError:
            self._timer_job = None

    def update_from_event(self, event: Dict[str, Any]) -> None:
        stage = str(event.get("stage", "Running"))
        scene_idx = int(event.get("scene_idx", 1) or 1)
        scene_total = max(1, int(event.get("scene_total", 1) or 1))
        status = str(event.get("status", "running"))

        if scene_idx != self._current_scene_idx:
            self._current_scene_idx = scene_idx
            self._scene_start = time.monotonic()

        self.stage_var.set(f"Stage: {stage}")
        self.scene_var.set(f"Scene: {scene_idx}/{scene_total}")

        detail_parts = []
        substage = event.get("substage")
        if substage is not None:
            detail_parts.append(str(substage))
        elapsed_s = event.get("elapsed_s")
        if elapsed_s is not None:
            try:
                detail_parts.append(f"Elapsed: {self._fmt_elapsed(float(elapsed_s))}")
            except (TypeError, ValueError):
                pass
        if detail_parts:
            self.detail_var.set(f"Detail: {' | '.join(detail_parts)}")
        else:
            self.detail_var.set("")

        percent = event.get("download_percent")
        mb = event.get("download_mb")
        total_mb = event.get("download_total_mb")
        if percent is not None and mb is not None and total_mb is not None:
            self.download_var.set(f"Download: {float(percent):.1f}% ({float(mb):.1f} / {float(total_mb):.1f} MB)")
        elif mb is not None:
            self.download_var.set(f"Download: {float(mb):.1f} MB")
        else:
            self.download_var.set("")

        if status == "done":
            self.status_var.set("Status: Completed")
            self.progress.stop()
        elif status == "error":
            self.status_var.set("Status: Failed")
            self.progress.stop()
        else:
            self.status_var.set("Status: Running")
            self.progress.start(10)

        self._update_timer_labels()

    def close(self) -> None:
        self._closed = True
        if self._timer_job is not None:
            try:
                self.root.after_cancel(self._timer_job)
            except Exception:
                pass
            self._timer_job = None
        try:
            self.progress.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def _show_gui_error_and_exit(message: str) -> None:
    """Show GUI error dialog and exit with message."""
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", message)
        root.destroy()
    except Exception:
        pass
    sys.exit(message)


def _looks_like_enmap_spectral_file(file_name: str) -> bool:
    """Return True when a file name matches EnMAP spectral image naming."""
    upper = os.path.basename(file_name).upper()
    return "SPECTRAL_IMAGE" in upper and upper.endswith((".TIF", ".TIFF", ".BSQ"))


def _detect_input_sensor_flags(input_path: str, batch_mode: bool) -> tuple[bool, bool]:
    """Detect whether selected input includes PRISMA and/or EnMAP scenes."""
    if not batch_mode:
        try:
            hyp_type = detect_hyp_type(input_path)
            return hyp_type == "PRISMA", hyp_type == "ENMAP"
        except Exception:
            file_name = os.path.basename(input_path)
            return file_name.upper().endswith(".HE5"), _looks_like_enmap_spectral_file(file_name)

    has_prisma = False
    has_enmap = False

    for _root, _dirs, files in os.walk(input_path):
        for name in files:
            upper = name.upper()
            if upper.endswith(".HE5"):
                has_prisma = True
            elif _looks_like_enmap_spectral_file(name):
                has_enmap = True
            if has_prisma and has_enmap:
                return True, True

    return has_prisma, has_enmap


def gui_get_inputs() -> Dict[str, Any]:
    """
    Display GUI for collecting processing parameters.

    Returns:
        dict: Configuration dictionary with all parameters
    """
    # Initialize hidden root for dialogs
    root = tk.Tk()
    root.withdraw()

    # Ask for processing mode
    mode_result = messagebox.askquestion(
        "HyperCoreg - Processing Mode",
        "Process multiple files (batch mode)?\n\n"
        "Yes = Select folder with multiple files\n"
        "No = Select a single file",
        icon='question'
    )

    batch_mode = mode_result == 'yes'

    # Select input
    if batch_mode:
        input_path = filedialog.askdirectory(
            title="Select Input Folder with PRISMA/EnMAP Files"
        )
    else:
        input_path = filedialog.askopenfilename(
            title="Select PRISMA or EnMAP File",
            filetypes=[
                ("PRISMA HE5", "*.he5"),
                (
                    "EnMAP Spectral Image",
                    "*SPECTRAL_IMAGE*.tif "
                    "*SPECTRAL_IMAGE*.tiff "
                    "*SPECTRAL_IMAGE*.bsq "
                    "*SPECTRAL_IMAGE*.TIF "
                    "*SPECTRAL_IMAGE*.TIFF "
                    "*SPECTRAL_IMAGE*.BSQ",
                ),
                ("EnMAP BSQ", "*SPECTRAL_IMAGE*.bsq *SPECTRAL_IMAGE*.BSQ"),
                (
                    "EnMAP TIFF",
                    "*SPECTRAL_IMAGE*.tif "
                    "*SPECTRAL_IMAGE*.tiff "
                    "*SPECTRAL_IMAGE*.TIF "
                    "*SPECTRAL_IMAGE*.TIFF",
                ),
                ("All Files", "*.*")
            ]
        )

    if not input_path:
        _show_gui_error_and_exit("No input selected")

    # Select output directory
    output_dir = filedialog.askdirectory(
        title="Select Output Directory"
    )

    if not output_dir:
        _show_gui_error_and_exit("No output directory selected")

    has_prisma_input, has_enmap_input = _detect_input_sensor_flags(input_path, batch_mode)

    root.destroy()

    # Create parameter configuration window
    config = _show_parameter_dialog(
        input_path,
        output_dir,
        batch_mode,
        has_prisma_input=has_prisma_input,
        has_enmap_input=has_enmap_input,
    )

    return config


def _show_parameter_dialog(
    input_path: str,
    output_dir: str,
    batch_mode: bool,
    has_prisma_input: bool,
    has_enmap_input: bool,
) -> Dict[str, Any]:
    """
    Show the parameter configuration dialog.

    Args:
        input_path: Selected input path
        output_dir: Selected output directory
        batch_mode: Whether batch mode is enabled
        has_prisma_input: True when selected input includes PRISMA scenes
        has_enmap_input: True when selected input includes EnMAP scenes

    Returns:
        dict: Configuration dictionary
    """
    param_root = tk.Tk()
    param_root.title(f"HyperCoreg v{__version__} - Configuration")

    # Window size
    window_width = 560
    window_height = 650
    param_root.geometry(f"{window_width}x{window_height}")
    param_root.resizable(True, True)
    param_root.minsize(500, 500)

    # Center window
    param_root.update_idletasks()
    x = (param_root.winfo_screenwidth() // 2) - (window_width // 2)
    y = (param_root.winfo_screenheight() // 2) - (window_height // 2)
    param_root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Variables for parameters
    config_vars = {
        'days_window': tk.IntVar(value=DEFAULT_CONFIG['days_window']),
        'min_overlap': tk.DoubleVar(value=DEFAULT_CONFIG['min_overlap']),
        'max_cloud': tk.DoubleVar(value=DEFAULT_CONFIG['max_cloud']),
        'max_input_cloud_cover': tk.DoubleVar(value=DEFAULT_CONFIG['max_input_cloud_cover']),
        'min_accuracy': tk.DoubleVar(value=DEFAULT_CONFIG['min_accuracy']),
        'max_displacement': tk.DoubleVar(value=DEFAULT_CONFIG['max_displacement']),
        'residual_threshold': tk.DoubleVar(value=DEFAULT_CONFIG['residual_threshold']),
        'min_tie_points': tk.IntVar(value=DEFAULT_CONFIG['min_tie_points']),
        'max_s2_candidates': tk.IntVar(value=DEFAULT_CONFIG['max_s2_candidates']),
        'residual_mad_factor': tk.DoubleVar(value=DEFAULT_CONFIG['residual_mad_factor']),
        's2_ref_band': tk.IntVar(value=DEFAULT_CONFIG['s2_ref_band']),
        'save_pre': tk.BooleanVar(value=DEFAULT_CONFIG['save_pre']),
        'gen_tiepoint_pngs': tk.BooleanVar(value=DEFAULT_CONFIG['gen_tiepoint_pngs']),
        'save_displacement_vectors': tk.BooleanVar(value=DEFAULT_CONFIG['save_displacement_vectors']),
        'use_inmemory': tk.BooleanVar(value=DEFAULT_CONFIG['use_inmemory']),
        'keep_temp_files': tk.BooleanVar(value=DEFAULT_CONFIG['keep_temp_files']),
        'save_pan': tk.BooleanVar(value=DEFAULT_CONFIG['save_pan']),
        'save_quality_mask': tk.BooleanVar(value=DEFAULT_CONFIG['save_quality_mask']),
        'remove_detector_overlap_bands': tk.BooleanVar(value=DEFAULT_CONFIG['remove_detector_overlap_bands']),
        'prisma_radiometric_mode': tk.StringVar(value=DEFAULT_CONFIG['prisma_radiometric_mode']),
        'normalization_mode': tk.StringVar(value=DEFAULT_CONFIG['normalization_mode']),
        'build_overviews': tk.BooleanVar(value=DEFAULT_CONFIG['build_overviews']),
        'strict_metadata': tk.BooleanVar(value=DEFAULT_CONFIG['strict_metadata']),
        'metadata_extension_level': tk.StringVar(value=GUI_METADATA_EXTENSION_LEVEL),
        'metadata_stats_mode': tk.StringVar(value=DEFAULT_CONFIG['metadata_stats_mode']),
        'metadata_stats_sample_windows': tk.IntVar(value=DEFAULT_CONFIG['metadata_stats_sample_windows']),
        'metadata_stats_seed': tk.IntVar(value=DEFAULT_CONFIG['metadata_stats_seed']),
        'metadata_histogram_buckets': tk.IntVar(value=DEFAULT_CONFIG['metadata_histogram_buckets']),
        'metadata_label_precision': tk.IntVar(value=DEFAULT_CONFIG['metadata_label_precision']),
        'validation_max_windows': tk.IntVar(value=DEFAULT_CONFIG['validation_max_windows']),
        'enmap_metadata_stats_mode': tk.StringVar(value=DEFAULT_CONFIG['enmap_metadata_stats_mode']),
    }

    result_config = {'cancelled': True}

    # Header
    header_frame = tk.Frame(param_root)
    header_frame.pack(fill='x', padx=10, pady=5)

    tk.Label(
        header_frame,
        text=f"HyperCoreg v{__version__}",
        font=('Arial', 14, 'bold')
    ).pack()

    mode_text = "Batch Mode" if batch_mode else "Single File Mode"
    tk.Label(
        header_frame,
        text=mode_text,
        font=('Arial', 10),
        fg='gray'
    ).pack()

    # Input/output display
    io_frame = tk.LabelFrame(param_root, text="Input/Output", padx=5, pady=5)
    io_frame.pack(fill='x', padx=10, pady=5)

    tk.Label(io_frame, text=f"Input: {os.path.basename(input_path)}", anchor='w').pack(fill='x')
    tk.Label(io_frame, text=f"Output: {output_dir}", anchor='w').pack(fill='x')

    # Notebook for parameter tabs
    notebook = ttk.Notebook(param_root)
    notebook.pack(fill='both', expand=True, padx=10, pady=5)

    # Tab 1: Basic Parameters
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Basic")

    basic_frame = tk.Frame(tab1, padx=10, pady=10)
    basic_frame.pack(fill='both', expand=True)

    params_basic = [
        ("Days window for S2 search:", config_vars['days_window'], "days"),
        ("Minimum overlap (0-1):", config_vars['min_overlap'], ""),
        ("Max S2 cloud cover:", config_vars['max_cloud'], "%"),
        ("Max input cloud cover:", config_vars['max_input_cloud_cover'], "%"),
        ("Min accuracy:", config_vars['min_accuracy'], "%"),
        ("Max displacement:", config_vars['max_displacement'], "m"),
    ]

    for i, (label, var, unit) in enumerate(params_basic):
        tk.Label(basic_frame, text=label, anchor='w').grid(row=i, column=0, sticky='w', pady=2)
        entry = tk.Entry(basic_frame, textvariable=var, width=10)
        entry.grid(row=i, column=1, padx=5, pady=2)
        tk.Label(basic_frame, text=unit).grid(row=i, column=2, sticky='w', pady=2)

    # Footer citation in Basic tab (kept at lower area to avoid overlap with parameter fields).
    basic_frame.grid_rowconfigure(len(params_basic), weight=1)
    citation_text = (
        "(The coregistration module is based on AROSICS 1.13.2: "
        "https://doi.org/10.5281/zenodo.17399222)"
    )
    tk.Label(
        basic_frame,
        text=citation_text,
        anchor='w',
        justify='left',
        wraplength=500,
        fg='gray'
    ).grid(row=len(params_basic) + 1, column=0, columnspan=3, sticky='sw', pady=(14, 0))

    # Tab 2: Advanced Parameters
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Advanced")

    adv_frame = tk.Frame(tab2, padx=10, pady=10)
    adv_frame.pack(fill='both', expand=True)

    params_advanced = [
        ("Residual threshold:", config_vars['residual_threshold'], "m"),
        ("Min tie points:", config_vars['min_tie_points'], ""),
        ("Max S2 candidates:", config_vars['max_s2_candidates'], ""),
        ("MAD factor:", config_vars['residual_mad_factor'], ""),
    ]

    for i, (label, var, unit) in enumerate(params_advanced):
        tk.Label(adv_frame, text=label, anchor='w').grid(row=i, column=0, sticky='w', pady=2)
        entry = tk.Entry(adv_frame, textvariable=var, width=10)
        entry.grid(row=i, column=1, padx=5, pady=2)
        tk.Label(adv_frame, text=unit).grid(row=i, column=2, sticky='w', pady=2)

    # S2 band selection
    row_idx = len(params_advanced)
    tk.Label(adv_frame, text="S2 reference band:", anchor='w').grid(
        row=row_idx, column=0, sticky='w', pady=2
    )

    s2_band_options = [f"B{k:02d} - {v['name']}" for k, v in S2_BANDS.items() if k in [2, 3, 4, 8]]
    s2_band_combo = ttk.Combobox(adv_frame, values=s2_band_options, state='readonly', width=20)
    s2_band_combo.set(f"B{DEFAULT_S2_BAND:02d} - {S2_BANDS[DEFAULT_S2_BAND]['name']}")
    s2_band_combo.grid(row=row_idx, column=1, columnspan=2, padx=5, pady=2, sticky='w')

    def update_s2_band(event):
        selected = s2_band_combo.get()
        band_num = int(selected.split(' ')[0][1:])
        config_vars['s2_ref_band'].set(band_num)

    s2_band_combo.bind('<<ComboboxSelected>>', update_s2_band)

    tk.Checkbutton(
        adv_frame,
        text="Use in-memory processing",
        variable=config_vars['use_inmemory'],
        anchor='w',
    ).grid(row=row_idx + 1, column=0, columnspan=3, sticky='w', pady=(8, 2))

    # Tab 3: Output Options
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text="Output")

    out_frame = tk.Frame(tab3, padx=10, pady=10)
    out_frame.pack(fill='both', expand=True)

    checkboxes = [
        ("Save pre-coregistration image", config_vars['save_pre']),
        ("Generate tie point PNGs", config_vars['gen_tiepoint_pngs']),
        ("Save displacement vectors (SHP + PNG)", config_vars['save_displacement_vectors']),
        ("Keep temporary files", config_vars['keep_temp_files']),
    ]
    if has_prisma_input:
        checkboxes.append(("Save PAN band (PRISMA)", config_vars['save_pan']))
    if has_prisma_input or has_enmap_input:
        checkboxes.append(
            ("Save quality auxiliaries (PRISMA masks / EnMAP QL)", config_vars['save_quality_mask'])
        )
    checkboxes.extend([
        ("Remove VNIR/SWIR overlap bands", config_vars['remove_detector_overlap_bands']),
        ("Build internal overviews (QGIS/ENVI)", config_vars['build_overviews']),
    ])

    for i, (label, var) in enumerate(checkboxes):
        cb = tk.Checkbutton(out_frame, text=label, variable=var, anchor='w')
        cb.grid(row=i, column=0, sticky='w', pady=2)

    next_output_row = len(checkboxes)
    if has_prisma_input:
        tk.Label(out_frame, text="PRISMA radiometry:", anchor='w').grid(
            row=next_output_row, column=0, sticky='w', pady=(10, 2)
        )
        prisma_radiometry_combo = ttk.Combobox(
            out_frame,
            state="readonly",
            values=["reflectance", "native-dn"],
            textvariable=config_vars['prisma_radiometric_mode'],
            width=12,
        )
        prisma_radiometry_combo.grid(
            row=next_output_row, column=1, sticky='w', padx=5, pady=(10, 2)
        )
        next_output_row += 1

    norm_row = next_output_row
    tk.Label(out_frame, text="Normalization mode:", anchor='w').grid(
        row=norm_row, column=0, sticky='w', pady=(10, 2)
    )
    norm_mode_combo = ttk.Combobox(
        out_frame,
        state="readonly",
        values=["none", "minmax", "percentile"],
        textvariable=config_vars['normalization_mode'],
        width=12,
    )
    norm_mode_combo.grid(row=norm_row, column=1, sticky='w', padx=5, pady=(10, 2))

    strict_row = norm_row + 1
    tk.Checkbutton(
        out_frame,
        text="Strict metadata writes",
        variable=config_vars['strict_metadata'],
        anchor='w',
    ).grid(row=strict_row, column=0, columnspan=3, sticky='w', pady=(6, 2))

    metadata_row = strict_row + 1
    tk.Label(out_frame, text="Metadata extension level:", anchor='w').grid(
        row=metadata_row, column=0, sticky='w', pady=(10, 2)
    )
    metadata_combo = ttk.Combobox(
        out_frame,
        state="readonly",
        values=list(GUI_METADATA_EXTENSION_CHOICES),
        textvariable=config_vars['metadata_extension_level'],
        width=12,
    )
    metadata_combo.grid(row=metadata_row, column=1, sticky='w', padx=5, pady=(10, 2))

    # Buttons
    button_frame = tk.Frame(param_root)
    button_frame.pack(fill='x', padx=10, pady=10)

    def on_run():
        try:
            for _name, var in config_vars.items():
                var.get()
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("Invalid parameter", f"Invalid parameter value: {exc}")
            return
        result_config['cancelled'] = False
        param_root.destroy()

    def on_cancel():
        result_config['cancelled'] = True
        param_root.destroy()

    tk.Button(
        button_frame,
        text="Run Coregistration",
        command=on_run,
        width=20,
        bg='#4CAF50',
        fg='white'
    ).pack(side='right', padx=5)

    tk.Button(
        button_frame,
        text="Cancel",
        command=on_cancel,
        width=10
    ).pack(side='right', padx=5)

    # Run dialog
    param_root.protocol("WM_DELETE_WINDOW", on_cancel)
    param_root.mainloop()

    if result_config['cancelled']:
        sys.exit("User cancelled")

    metadata_settings = _resolve_gui_metadata_runtime_settings(
        config_vars['metadata_extension_level'].get()
    )
    config_vars['metadata_stats_mode'].set(metadata_settings['metadata_stats_mode'])
    config_vars['metadata_stats_sample_windows'].set(metadata_settings['metadata_stats_sample_windows'])
    config_vars['metadata_stats_seed'].set(metadata_settings['metadata_stats_seed'])
    config_vars['metadata_histogram_buckets'].set(metadata_settings['metadata_histogram_buckets'])
    config_vars['metadata_label_precision'].set(metadata_settings['metadata_label_precision'])
    config_vars['validation_max_windows'].set(metadata_settings['validation_max_windows'])
    config_vars['enmap_metadata_stats_mode'].set(metadata_settings['enmap_metadata_stats_mode'])

    config = _build_gui_runtime_config(
        input_path=input_path,
        output_dir=output_dir,
        batch_mode=batch_mode,
        config_vars=config_vars,
        strict_metadata=bool(config_vars['strict_metadata'].get()),
        metadata_extension_level=metadata_settings['metadata_extension_level'],
        metadata_stats_mode=metadata_settings['metadata_stats_mode'],
        metadata_stats_sample_windows=metadata_settings['metadata_stats_sample_windows'],
        metadata_stats_seed=metadata_settings['metadata_stats_seed'],
        metadata_histogram_buckets=metadata_settings['metadata_histogram_buckets'],
        metadata_label_precision=metadata_settings['metadata_label_precision'],
        validation_max_windows=metadata_settings['validation_max_windows'],
    )

    return config


def run_gui_worker_pipeline(
    *,
    config: Dict[str, Any],
    progress_callback: Any,
    request_userpass_fn: Any,
    run_coregistration: Any,
    run_batch_coregistration: Any,
    detect_hyp_type: Any,
) -> None:
    """Execute the GUI worker pipeline without UI event-loop concerns."""
    worker_config = dict(config)
    worker_config["prompt_userpass_fn"] = request_userpass_fn
    if config["batch_mode"]:
        result = run_batch_coregistration(
            config["input_path"],
            config["output_dir"],
            worker_config,
            progress_callback=progress_callback,
        )
        if not isinstance(result, Mapping):
            raise RuntimeError("Batch runner returned no result payload.")
        failure_reason = describe_result_failure(result, "batch")
        if failure_reason:
            raise RuntimeError(failure_reason)
        return

    hyp_type = detect_hyp_type(config["input_path"])
    result = run_coregistration(
        config["input_path"],
        hyp_type,
        config["output_dir"],
        worker_config,
        progress_callback=progress_callback,
        scene_idx=1,
        scene_total=1,
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("Single-scene runner returned no result payload.")
    failure_reason = describe_result_failure(result, "single")
    if failure_reason:
        raise RuntimeError(failure_reason)


def main() -> int:
    """
    Main GUI entry point.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    log_section_header("HYPERCOREG GUI")

    progress_window = None
    try:
        config = gui_get_inputs()

        # Setup logging
        setup_logging(output_dir=config['output_dir'])

        logger.info(f"HyperCoreg version {__version__}")
        logger.info(f"Mode: {'Batch' if config['batch_mode'] else 'Single'}")
        logger.info(f"Input: {config['input_path']}")
        logger.info(f"Output: {config['output_dir']}")

        progress_window = _PipelineProgressWindow(is_batch=config.get('batch_mode', False))
        event_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        final_state = {
            "finished": False,
            "status": None,  # "success" | "interrupt" | "error"
            "error": None,
            "traceback": None,
        }
        prompt_lock = threading.Lock()
        pending_prompt_requests: List[Dict[str, Any]] = []

        def _unregister_prompt_request(req: Dict[str, Any]) -> None:
            with prompt_lock:
                if req in pending_prompt_requests:
                    pending_prompt_requests.remove(req)

        def _fail_pending_prompt_requests(message: str) -> None:
            with prompt_lock:
                pending = list(pending_prompt_requests)
                pending_prompt_requests.clear()
            for req in pending:
                if not req["done"].is_set():
                    req["error"] = RuntimeError(message)
                    req["done"].set()

        def progress_callback(event: Dict[str, Any]) -> None:
            event_queue.put(("progress", event))

        # Import and run
        from hypercoreg.coregistration import (
            run_coregistration,
            run_batch_coregistration,
            _prompt_cdse_userpass_gui,
        )
        from hypercoreg.utils import detect_hyp_type

        def _request_userpass_on_main_thread() -> tuple[str, str, Optional[str]]:
            req = {
                "done": threading.Event(),
                "result": None,
                "error": None,
            }
            with prompt_lock:
                pending_prompt_requests.append(req)
            event_queue.put(("prompt_userpass", req))
            req["done"].wait()
            _unregister_prompt_request(req)
            if req["error"] is not None:
                raise RuntimeError(str(req["error"]))
            result = req["result"]
            if result is None:
                raise RuntimeError("Credential prompt returned no credentials.")
            return result

        def worker() -> None:
            try:
                run_gui_worker_pipeline(
                    config=config,
                    progress_callback=progress_callback,
                    request_userpass_fn=_request_userpass_on_main_thread,
                    run_coregistration=run_coregistration,
                    run_batch_coregistration=run_batch_coregistration,
                    detect_hyp_type=detect_hyp_type,
                )
                event_queue.put(("done", None))
            except KeyboardInterrupt:
                event_queue.put(("interrupt", None))
            except Exception as exc:
                event_queue.put(("error", (exc, traceback.format_exc())))

        worker_thread = threading.Thread(target=worker, name="hypercoreg_worker", daemon=True)
        worker_thread.start()

        def _close_progress_window() -> None:
            nonlocal progress_window
            _fail_pending_prompt_requests("GUI closed while waiting for credential prompt.")
            if progress_window is not None:
                progress_window.close()
                progress_window = None

        def _poll_events() -> None:
            if progress_window is None or final_state["finished"]:
                return
            try:
                while True:
                    kind, payload = event_queue.get_nowait()
                    if kind == "progress":
                        progress_window.update_from_event(payload)
                    elif kind == "done":
                        final_state["finished"] = True
                        final_state["status"] = "success"
                        _close_progress_window()
                        return
                    elif kind == "interrupt":
                        final_state["finished"] = True
                        final_state["status"] = "interrupt"
                        _close_progress_window()
                        return
                    elif kind == "error":
                        final_state["finished"] = True
                        final_state["status"] = "error"
                        final_state["error"], final_state["traceback"] = payload
                        _close_progress_window()
                        return
                    elif kind == "prompt_userpass":
                        req = payload
                        try:
                            if progress_window is None or final_state["finished"]:
                                raise RuntimeError(
                                    "Credential prompt cancelled because processing is shutting down."
                                )
                            req["result"] = _prompt_cdse_userpass_gui()
                        except Exception as prompt_exc:
                            req["error"] = prompt_exc
                        finally:
                            req["done"].set()
                            _unregister_prompt_request(req)
            except queue.Empty:
                pass

            if progress_window is not None:
                progress_window.root.after(100, _poll_events)

        if progress_window is not None:
            progress_window.root.after(100, _poll_events)
            progress_window.root.mainloop()

        if final_state["status"] == "success":
            messagebox.showinfo("Complete", "Coregistration completed successfully!")
            return 0
        if final_state["status"] == "interrupt":
            logger.warning("Processing interrupted by user")
            return 130
        if final_state["status"] == "error":
            err = final_state.get("error")
            logger.error(f"Processing failed: {err}")
            tb_text = final_state.get("traceback")
            if tb_text:
                print(tb_text)
            try:
                messagebox.showerror("Error", f"Processing failed:\n{err}")
            except Exception:
                pass
            return 1

        # Fallback: worker ended unexpectedly without terminal event.
        if worker_thread.is_alive():
            logger.error("Processing did not finish correctly; worker still alive.")
        else:
            logger.error("Processing ended without a terminal state.")
        return 1

    except KeyboardInterrupt:
        if progress_window is not None:
            progress_window.update_from_event(
                {"stage": "Interrupted", "scene_idx": 1, "scene_total": 1, "status": "error"}
            )
            progress_window.close()
        logger.warning("Processing interrupted by user")
        return 130

    except Exception as e:
        if progress_window is not None:
            progress_window.update_from_event(
                {"stage": "Failed", "scene_idx": 1, "scene_total": 1, "status": "error"}
            )
            progress_window.close()
        logger.error(f"Processing failed: {e}")
        traceback.print_exc()

        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", f"Processing failed:\n{e}")
            root.destroy()
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    sys.exit(main())
