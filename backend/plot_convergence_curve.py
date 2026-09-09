"""
plot_convergence_curve.py
---------------------------
Generates Figure 4.4 (Convergence Curve Comparison: Standard SMA vs.
ESMA) by calling YOUR RUNNING BACKEND directly -- no numbers are typed
in manually. Requires main.py (uvicorn) to already be running on
localhost:8000 (i.e. run this while your system is up, same as when
you use the frontend).

Usage:
    python plot_convergence_curve.py --image path/to/sample_opg.png --out figure_4.4_convergence.png

Requires: requests, matplotlib (pip install requests matplotlib --break-system-packages)
"""

import argparse

import matplotlib.pyplot as plt
import requests


def fetch_convergence(image_path, endpoint):
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"d": "4", "N": "30", "T": "100"}
        resp = requests.post(endpoint, files=files, data=data)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Backend returned an error: {payload.get('message')}")
    return payload["convergence_curve"]


def main(args):
    print("Fetching Standard SMA convergence curve...")
    standard_curve = fetch_convergence(args.image, "http://localhost:8000/analyze/standard/")
    print(f"  {len(standard_curve)} iterations, final entropy = {standard_curve[-1]:.4f}")

    print("Fetching ESMA convergence curve...")
    enhanced_curve = fetch_convergence(args.image, "http://localhost:8000/analyze/enhanced/")
    print(f"  {len(enhanced_curve)} iterations, final entropy = {enhanced_curve[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(standard_curve)), standard_curve, label="Standard SMA",
            color="#577E89", linewidth=2)
    ax.plot(range(len(enhanced_curve)), enhanced_curve, label="ESMA",
            color="#E1A36F", linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far Kapur's Entropy")
    ax.set_title("Convergence Curve Comparison: Standard SMA vs. ESMA")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Zoom into the first N iterations, since both curves typically
    # plateau early and the remaining iterations add no visible detail
    # -- this only changes the VIEW, not the underlying data (the full
    # curves are still used for both lines; iterations beyond the zoom
    # window are simply cropped out of frame).
    if args.zoom:
        ax.set_xlim(0, args.zoom)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a sample OPG image")
    parser.add_argument("--out", default="figure_4.4_convergence.png")
    parser.add_argument("--zoom", type=int, default=30,
                         help="Show only the first N iterations (0 to disable and show all)")
    main(parser.parse_args())