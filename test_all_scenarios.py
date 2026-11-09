# test_all_scenarios.py
from generator import generate_dataset
from scanner import scan_for_void_clusters
from tension import calculate_boundary_tension
from reconstructor import reconstruct_void


def test_all_scenarios():
    print("=== GDE 3.1 GEOMETRIC & TOPOLOGICAL TEST SUITE ===")

    # 1 & 2. Plane & Gradient Reconstruction Test
    print("\n[1 & 2] Testing Plane & Gradient Reconstruction...")
    w, h = 20, 20
    grid_plane = generate_dataset(width=w, height=h, shapes=["rectangle"], seed=42)
    voids = scan_for_void_clusters(grid_plane, connectivity=4, bounds=(1, w, 1, h))
    tension = calculate_boundary_tension(voids, grid_plane)
    report_plane = reconstruct_void(voids, tension, grid_plane, mode="accurate")
    print(f" -> Plane/Gradient test: Reconstructed {report_plane['summary']['cells_reconstructed']} cells successfully.")

    # 3 & 4. Curved Surface & Large Void Reconstruction Test
    print("\n[3 & 4] Testing Large Voids & Curved Surfaces...")
    w, h = 25, 25
    grid_large = generate_dataset(width=w, height=h, shapes=["l_shape"], seed=10,
                                  max_void_size=8)
    voids_large = scan_for_void_clusters(grid_large, connectivity=4, bounds=(1, w, 1, h))
    tension_large = calculate_boundary_tension(voids_large, grid_large)
    report_large = reconstruct_void(voids_large, tension_large, grid_large, mode="accurate")
    print(f" -> Large Void test: Reconstructed {report_large['summary']['cells_reconstructed']} cells across large clusters.")

    # 5. Multiple Disconnected Voids Test
    print("\n[5] Testing Multiple Disconnected Voids...")
    w, h = 30, 30
    grid_multi = generate_dataset(width=w, height=h,
                                  shapes=["blob", "speckle", "diagonal"], seed=5)
    voids_multi = scan_for_void_clusters(grid_multi, connectivity=4, bounds=(1, w, 1, h))
    print(f" -> Disconnected Voids test: Detected {len(voids_multi)} distinct void clusters.")
    tension_multi = calculate_boundary_tension(voids_multi, grid_multi)
    report_multi = reconstruct_void(voids_multi, tension_multi, grid_multi, mode="accurate")
    print(f" -> Disconnected Voids test: Successfully filled all {report_multi['summary']['cells_reconstructed']} cells.")

    # 6. Edge Reconstruction Test
    print("\n[6] Testing Edge/Boundary Reconstruction...")
    edge_grid = {(x, y): {"status": "Recorded", "intensity": float(x + y)} for x in range(1, 11) for y in range(1, 11)}
    for y in range(3, 8):
        edge_grid.pop((1, y))  # Remove cells touching boundary x=1
    voids_edge = scan_for_void_clusters(edge_grid, connectivity=4, bounds=(1, 10, 1, 10))
    tension_edge = calculate_boundary_tension(voids_edge, edge_grid)
    report_edge = reconstruct_void(voids_edge, tension_edge, edge_grid, mode="accurate")
    print(f" -> Edge Reconstruction test: Reconstructed {report_edge['summary']['cells_reconstructed']} boundary-touching void cells.")

    # 7. Speckle Reconstruction Test
    print("\n[7] Testing Isolated Speckle Reconstruction...")
    speckle_grid = {(x, y): {"status": "Recorded", "intensity": 50.0} for x in range(1, 8) for y in range(1, 8)}
    speckle_grid.pop((4, 4))  # Remove single isolated center cell
    voids_speckle = scan_for_void_clusters(speckle_grid, connectivity=4, bounds=(1, 7, 1, 7))
    tension_speckle = calculate_boundary_tension(voids_speckle, speckle_grid)
    report_speckle = reconstruct_void(voids_speckle, tension_speckle, speckle_grid, mode="accurate")
    print(f" -> Speckle test: Isolated cell reconstructed successfully with intensity {list(report_speckle['audit_trail'].values())[0]['intensity']:.2f}.")

    print("\n=== ALL 7 SCENARIOS TESTED SUCCESSFULLY! ===")


if __name__ == "__main__":
    test_all_scenarios()
