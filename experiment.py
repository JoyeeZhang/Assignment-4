from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def distances(x, centers):
    return ((x[:, None] - centers) ** 2).sum(axis=2)


def kmeans(x, k, seed, max_iter=300):
    rng = np.random.default_rng(seed)
    unique = np.unique(x, axis=0)
    centers = unique[rng.choice(len(unique), k, replace=False)]
    previous = None
    for _ in range(max_iter):
        squared = distances(x, centers)
        groups = squared.argmin(axis=1)
        counts = np.bincount(groups, minlength=k)
        for empty in np.flatnonzero(counts == 0):
            residual = squared[np.arange(len(x)), groups]
            donor = np.argmax(np.where(counts[groups] > 1, residual, -np.inf))
            counts[groups[donor]] -= 1
            groups[donor] = empty
            counts[empty] += 1
        centers = np.array([x[groups == j].mean(axis=0) for j in range(k)])
        if np.array_equal(previous, groups):
            break
        previous = groups
    inertia = np.sum((x - centers[groups]) ** 2)
    return inertia, groups, centers


def cluster(images):
    row, col = np.indices((8, 8))
    ink = images.sum(axis=(1, 2))
    raw = np.column_stack([ink / (64 * 16), (images * col).sum((1, 2)) / ink,
                           (images * row).sum((1, 2)) / ink])
    x = StandardScaler().fit_transform(raw)
    rows, solutions = [], {}
    for k in range(2, 8):
        runs = [kmeans(x, k, seed) for seed in range(12)]
        loss, labels, centers = min(runs, key=lambda r: r[0])
        agreement = [adjusted_rand_score(a[1], b[1]) for i, a in enumerate(runs)
                     for b in runs[i + 1:]]
        rows.append([k, loss, silhouette_score(x, labels), np.mean(agreement)])
        solutions[k] = labels, centers
    return raw, x, np.array(rows), solutions


def classify(pixels, target):
    train, hold = train_test_split(np.arange(len(target)), test_size=0.4,
                                   stratify=target, random_state=242)
    val, test = train_test_split(hold, test_size=0.5, stratify=target[hold], random_state=243)
    x, y = pixels[train] / 16, target[train]
    def fit(labels, c, gamma):
        return SVC(C=c, gamma=gamma, kernel="rbf").fit(x, labels)
    grid = []
    for c in (0.1, 1, 10):
        for gamma in (0.01, 0.05, 0.2):
            pred = fit(y, c, gamma).predict(pixels[val] / 16)
            grid.append((f1_score(target[val], pred, average="macro"), c, gamma))
    score, c, gamma = max(grid, key=lambda entry: entry[0])
    def evaluate(labels):
        pred = fit(labels, c, gamma).predict(pixels[test] / 16)
        return np.array([accuracy_score(target[test], pred),
                         f1_score(target[test], pred, average="macro")])
    # X=None queries fitted samples while excluding each sample itself.
    neighbors = NearestNeighbors(n_neighbors=9).fit(x).kneighbors(return_distance=False)
    runs = {}
    for setting in ("Uniform", "Directed"):
        records = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            observed = y.copy()
            draw = rng.random(len(y))
            if setting == "Uniform":
                mask = draw < 0.20
                observed[mask] = (y[mask] + rng.integers(1, 10, mask.sum())) % 10
            else:
                for source, destination in [(3, 8), (8, 3), (9, 4)]:
                    observed[(y == source) & (draw < 0.55)] = destination
            votes = np.array([np.bincount(labels, minlength=10)
                              for labels in observed[neighbors]])
            change = (votes.max(axis=1) >= 7) & (votes.argmax(axis=1) != observed)
            repaired = np.where(change, votes.argmax(axis=1), observed)
            records.append((*evaluate(observed), *evaluate(repaired),
                            np.sum(observed != y), np.sum(change & (repaired == y)),
                            np.sum(change & (observed == y)), np.sum(change)))
        runs[setting] = np.array(records)
    return (len(train), len(val), len(test)), (score, c, gamma), evaluate(y), runs


def generate_figures(images, clusters, runs, output_dir):
    """Save the four experiment figures as RGB PNGs beside the LaTeX report."""
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    raw, _, diagnostics, solutions = clusters
    groups, _ = solutions[4]
    means = np.array([raw[groups == j].mean(axis=0) for j in range(4)])
    order = [means[:,0].argmax(), means[:,2].argmax(), means[:,1].argmin(), means[:,1].argmax()]
    names = ["Dense", "Lower", "Upper left", "Upper right"]

    def save_rgb(fig, filename):
        fig.set_dpi(220)
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        Image.fromarray(pixels).save(output_dir / filename, dpi=(220, 220))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.25), layout="constrained")
    for ax, values, name in zip(axes, diagnostics[:, 1:].T, ["WCSS", "Silhouette", "Restart agreement (ARI)"]):
        ax.plot(diagnostics[:, 0], values, "o-", color="#386989", markersize=4)
        ax.scatter(4, values[diagnostics[:, 0] == 4], color="#c34f22", s=52, zorder=3)
        ax.set(xticks=diagnostics[:, 0], xlabel="Number of clusters", title=name)
    save_rgb(fig, "choosing_k.png")

    fig, axes = plt.subplots(2, 4, figsize=(8, 3.6), layout="constrained",
                             sharex="row", sharey="row")
    for column, (j, name, color) in enumerate(zip(order, names, ["#785498", "#007b73", "#ce7429", "#3474a4"])):
        top, bottom = axes[:, column]
        top.scatter(*raw[groups == j, 1:].T, s=5, color=color, alpha=.45, linewidths=0)
        top.plot(*means[j, 1:], "kx", markersize=8)
        top.set(title=name, xlabel="Column")
        bottom.imshow(images[groups == j].mean(axis=0), cmap="viridis",
                      vmin=0, vmax=16, interpolation="nearest")
        bottom.set_axis_off()
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylabel("Row")
    save_rgb(fig, "cluster_profiles.png")

    fig, ax = plt.subplots(figsize=(8, 3.1), layout="constrained")
    for setting, color, marker in [("Uniform", "#007b73", "o"), ("Directed", "#bc532b", "s")]:
        records = runs[setting]
        changes = 100 * (records[:, 3] - records[:, 1])
        ax.plot(np.arange(len(records)), changes, marker=marker, lw=0, color=color, label=setting)
        ax.axhline(changes.mean(), color=color, ls="--", lw=1)
    ax.axhline(0, color="#303030", lw=.9)
    ax.set(xticks=np.arange(10), xlabel="Corruption seed", ylabel="Change in test macro-F1 (points)")
    ax.legend(ncol=2, frameon=False)
    save_rgb(fig, "repair_by_seed.png")

    fig, ax = plt.subplots(figsize=(8, 2.85), layout="constrained")
    left = np.zeros(2)
    for caption, color, counts in [
        ("Restored correct digit", "#007b73", [a[:, 5].mean() for a in runs.values()]),
        ("Spoiled correct label", "#bc532b", [a[:, 6].mean() for a in runs.values()]),
        ("Wrong to different wrong", "#786597", [(a[:, 7]-a[:, 5]-a[:, 6]).mean() for a in runs.values()])]:
        ax.barh(list(runs), counts, left=left, color=color, label=caption, height=.55)
        left += counts
    ax.invert_yaxis()
    ax.set_xlabel("Mean number of training labels changed per run")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=3, frameon=False, fontsize=10)
    save_rgb(fig, "repair_outcomes.png")


if __name__ == "__main__":
    data = load_digits()
    clusters = cluster(data.images)
    _, _, _, runs = classify(data.data, data.target)
    generate_figures(data.images, clusters, runs, Path(__file__).resolve().parent)
