"""
esma_step_by_step.py
---------------------
Runs the REAL enhanced_sma() algorithm on an actual image, but prints out
every step of the computation in plain terms with real numbers -- for
walking through "how does ESMA actually compute this" during your defense.

Usage:
    python esma_step_by_step.py --image path/to/your_xray.png
    python esma_step_by_step.py                          (uses a built-in test image if no --image given)
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, '.')
import sma_algorithms as sma


def make_test_image():
    """A small synthetic image with a few distinct intensity bands, standing
    in for a real X-ray so this script runs even without one on hand."""
    rng = np.random.default_rng(7)
    img = np.zeros((150, 150), dtype=np.uint8)
    img[:, :40] = rng.integers(10, 30, size=(150, 40))     # dark band
    img[:, 40:90] = rng.integers(90, 120, size=(150, 50))   # mid band
    img[:, 90:130] = rng.integers(150, 180, size=(150, 40))  # bright band
    img[:, 130:] = rng.integers(210, 245, size=(150, 20))    # brightest band
    return img


def main(args):
    print("=" * 70)
    print("STEP 1 -- INPUT IMAGE")
    print("=" * 70)
    if args.image:
        import cv2
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {args.image}, using built-in test image instead.")
            img = make_test_image()
        else:
            print(f"Loaded: {args.image}")
    else:
        print("No --image given, using a small synthetic test image (4 intensity bands).")
        img = make_test_image()
    print(f"Image shape: {img.shape[0]} x {img.shape[1]} pixels")
    print(f"Pixel value range: {img.min()} to {img.max()} (0=black, 255=white)")

    print()
    print("=" * 70)
    print("STEP 2 -- BUILD THE GRAYSCALE HISTOGRAM")
    print("=" * 70)
    prob = sma.compute_histogram_prob(img)
    top5 = np.argsort(-prob)[:5]
    print("This turns the image into a probability distribution over the")
    print("256 possible pixel intensities (0-255). The 5 most common")
    print("intensity values in this image, with their share of all pixels:")
    for v in top5:
        print(f"  intensity {v:3d}: {prob[v]*100:5.2f}% of pixels")

    print()
    print("=" * 70)
    print("STEP 3 -- OBJECTIVE 2: QUASI-UNIFORM (LATIN HYPERCUBE) INITIALIZATION")
    print("=" * 70)
    rng = np.random.default_rng(args.seed)
    d, N = args.d, args.N
    X = sma._initial_population(rng, N, d, 0, 255, init="lhs")
    X.sort(axis=1)
    print(f"Placing N={N} agents, each proposing d={d} threshold values.")
    print("First 5 agents' starting threshold vectors (already sorted):")
    for i in range(5):
        vals = ", ".join(f"{v:.1f}" for v in X[i])
        print(f"  agent {i}: [{vals}]")
    print("Notice they're already spread out across the 0-255 range, not")
    print("clumped together -- that's the Latin Hypercube guarantee.")

    print()
    print("=" * 70)
    print("STEP 4 -- SCORE EVERY AGENT WITH KAPUR'S ENTROPY")
    print("=" * 70)
    table = sma.KapurEntropyTable(prob)
    fitness = table.evaluate(X, assume_sorted=True)
    print("Each agent's threshold vector is scored -- higher is better:")
    for i in range(5):
        print(f"  agent {i}: fitness = {fitness[i]:.4f}")
    best_idx = int(np.argmax(fitness))
    print(f"Best of the starting population: agent {best_idx}, "
          f"fitness = {fitness[best_idx]:.4f}, thresholds = "
          f"{[round(float(v),1) for v in X[best_idx]]}")

    print()
    print("=" * 70)
    print("STEP 5 -- OBJECTIVE 1: RANK AGENTS, PICK TOP-K LEADERS")
    print("=" * 70)
    k = args.k
    order = np.argsort(-fitness)
    leader_idx = order[:k]
    print(f"Ranking all {N} agents by fitness, taking the top k={k} as leaders:")
    for rank, li in enumerate(leader_idx):
        print(f"  leader #{rank+1}: agent {li}, fitness = {fitness[li]:.4f}")
    rank_w = np.arange(k, 0, -1, dtype=np.float64)
    rank_w = rank_w / rank_w.sum()
    print("Rank-based weights given to each leader (best leader gets the most influence):")
    for rank, w in enumerate(rank_w):
        print(f"  leader #{rank+1} weight: {w:.3f}")

    print()
    print("=" * 70)
    print("STEP 6 -- MOVE ONE AGENT (WORKED EXAMPLE)")
    print("=" * 70)
    example_agent = order[N // 2]  # a middling agent, not already a leader
    pick = int(np.searchsorted(np.cumsum(rank_w), rng.random()))
    pick = min(pick, k - 1)
    chosen_leader = leader_idx[pick]
    print(f"Agent {example_agent} (fitness {fitness[example_agent]:.4f}) samples a leader...")
    print(f"  -> drew leader #{pick+1} (agent {chosen_leader}, fitness {fitness[chosen_leader]:.4f})")
    a = 1.0
    ia, ib = rng.integers(0, N, size=2)
    vb = rng.uniform(-a, a, size=d)
    W_i = 1.2  # illustrative adaptive-weight value for this worked example
    inner = W_i * X[ia] - X[ib]
    new_pos = X[chosen_leader] + vb * inner
    new_pos = np.clip(new_pos, 0, 255)
    new_pos_sorted = np.sort(new_pos)
    new_fit = table.evaluate_one(new_pos_sorted)
    print(f"  old position: {[round(float(v),1) for v in X[example_agent]]}")
    print(f"  new proposed position: {[round(float(v),1) for v in new_pos_sorted]}")
    print(f"  new fitness: {new_fit:.4f}  (old was {fitness[example_agent]:.4f})")
    verdict = "IMPROVED -- keep the new position" if new_fit > fitness[example_agent] else "worse -- keep the old position (greedy selection)"
    print(f"  -> {verdict}")

    print()
    print("=" * 70)
    print("STEP 7 -- OBJECTIVE 3: MEASURE PROGRESS, ADJUST STEP SIZE a(t)")
    print("=" * 70)
    print("Run the full algorithm for a few iterations to show a(t) adapting:")
    result = sma.enhanced_sma(prob, d=d, N=N, T=15, seed=args.seed,
                               k=k, record_history=True)
    hist = result["history"]
    print(f"{'iter':>4} {'best fitness':>14} {'a(t)':>8} {'CR(t)':>8}")
    for t in range(len(hist["a"])):
        print(f"{t+1:4d} {result['convergence'][t+1]:14.4f} {hist['a'][t]:8.3f} {hist['CR'][t]:8.5f}")
    print()
    print("Notice: a(t) shrinks while fitness keeps improving (fine-tuning),")
    print("and would grow again if progress stalled for a while.")

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Best thresholds found: {result['thresholds']}")
    print(f"Best fitness: {result['fitness']:.4f}")
    print(f"Iterations actually used: {result.get('iterations_used', '?')}")
    print(f"Runtime: {result['runtime_sec']*1000:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=None, help="Path to a real image (optional)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    main(parser.parse_args())
