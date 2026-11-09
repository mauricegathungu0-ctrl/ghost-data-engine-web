"""
Optional 3D voxel visualizer for spatiotemporal (x, y, t) results.
Requires matplotlib. Saves a PNG; it does not alter engine data.
"""
def save_3d_visualization(result, output_path="ghost_data_3d.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib with: python -m pip install matplotlib") from exc

    known = result.get("known_voxels", {})
    inferred = result.get("reconstructed_voxels", {})
    unresolved = result.get("unresolved_voxels", [])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def scatter(points, marker, label, size):
        if points:
            xs, ys, ts = zip(*points)
            ax.scatter(xs, ys, ts, marker=marker, label=label, s=size)

    scatter(list(known), "s", "Recorded", 18)
    scatter(list(inferred), "o", "Reconstructed", 28)
    scatter(unresolved, "x", "Unresolved", 35)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time / frame")
    ax.set_title("Ghost Data Engine 3.3 — Space-Time Reconstruction")
    if known or inferred or unresolved:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
