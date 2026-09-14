from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def find_panel_files(input_dir: Path, expected_panels: int = 8) -> list[Path]:
    """
    Locate the per-cancer KM panel PNG files created by
    plot_pathtokensurv_km_per_cancer.py.

    Expected naming:
        KM_PerCancer_Panel_01_of_08.png
        ...
        KM_PerCancer_Panel_08_of_08.png
    """
    files = [
        input_dir / f"KM_PerCancer_Panel_{i:02d}_of_{expected_panels:02d}.png"
        for i in range(1, expected_panels + 1)
    ]

    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "The following expected panel files were not found:\n  - "
            + "\n  - ".join(missing)
            + f"\n\nInput directory: {input_dir.resolve()}"
        )

    return files


def combine_panels(
    panel_files: list[Path],
    output_dir: Path,
    ncols: int = 4,
    nrows: int = 2,
    figure_width: float = 24.0,
    figure_height: float = 18.0,
    dpi: int = 600,
    title: str = (
        "Cancer-specific Kaplan–Meier survival analysis "
        "by PathTokenSurv risk groups"
    ),
    add_panel_labels: bool = True,
) -> None:
    """
    Combine the eight pre-rendered KM panel images into one master figure.

    Default layout:
        4 columns × 2 rows

        A  B  C  D
        E  F  G  H
    """
    if ncols * nrows < len(panel_files):
        raise ValueError(
            f"Layout {ncols}x{nrows} has only {ncols*nrows} slots, "
            f"but {len(panel_files)} panel images must be placed."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )
    axes = axes.ravel()

    for i, panel_path in enumerate(panel_files):
        img = mpimg.imread(panel_path)

        ax = axes[i]
        ax.imshow(img)
        ax.axis("off")

        if add_panel_labels:
            label = chr(ord("A") + i)
            ax.text(
                0.01,
                0.99,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=18,
                fontweight="bold",
                bbox={
                    "boxstyle": "square,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

    # Hide any unused axes.
    for ax in axes[len(panel_files):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=20, y=0.995)

    # Tight layout with room for the figure title.
    fig.tight_layout(rect=[0, 0, 1, 0.975], pad=0.4)

    output_png = output_dir / "KM_PerCancer_All8_Composite.png"
    output_pdf = output_dir / "KM_PerCancer_All8_Composite.pdf"
    output_svg = output_dir / "KM_PerCancer_All8_Composite.svg"

    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")

    plt.close(fig)

    print("Composite Kaplan–Meier figure created successfully.")
    print(f"PNG: {output_png.resolve()}")
    print(f"PDF: {output_pdf.resolve()}")
    print(f"SVG: {output_svg.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the 8 cancer-specific Kaplan–Meier panel images generated "
            "by plot_pathtokensurv_km_per_cancer.py into one publication-ready "
            "master figure."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing KM_PerCancer_Panel_01_of_08.png through "
            "KM_PerCancer_Panel_08_of_08.png."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for the composite figure. "
            "Default: same as --input-dir."
        ),
    )

    parser.add_argument(
        "--layout",
        choices=["4x2", "2x4", "8x1", "1x8"],
        default="4x2",
        help=(
            "Composite layout. Recommended for publication: 4x2. "
            "Use 8x1 only if you specifically want one horizontal row."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution. Default: 600 dpi.",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Optional figure width in inches.",
    )

    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Optional figure height in inches.",
    )

    parser.add_argument(
        "--title",
        default=(
            "Cancer-specific Kaplan–Meier survival analysis "
            "by PathTokenSurv risk groups"
        ),
        help="Overall composite figure title.",
    )

    parser.add_argument(
        "--no-panel-labels",
        action="store_true",
        help="Do not add A–H labels to the eight grouped panels.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else input_dir
    )

    # Parse layout as columns x rows.
    layout_map = {
        "4x2": (4, 2),
        "2x4": (2, 4),
        "8x1": (8, 1),
        "1x8": (1, 8),
    }
    ncols, nrows = layout_map[args.layout]

    # Sensible default physical sizes for each layout.
    default_sizes = {
        "4x2": (24.0, 18.0),
        "2x4": (16.0, 28.0),
        "8x1": (48.0, 8.0),
        "1x8": (8.0, 48.0),
    }

    default_width, default_height = default_sizes[args.layout]
    figure_width = args.width if args.width is not None else default_width
    figure_height = args.height if args.height is not None else default_height

    panel_files = find_panel_files(input_dir, expected_panels=8)

    combine_panels(
        panel_files=panel_files,
        output_dir=output_dir,
        ncols=ncols,
        nrows=nrows,
        figure_width=figure_width,
        figure_height=figure_height,
        dpi=args.dpi,
        title=args.title,
        add_panel_labels=not args.no_panel_labels,
    )


if __name__ == "__main__":
    main()
