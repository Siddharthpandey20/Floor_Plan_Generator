
# Corrected Demo Script - Fixing Boundary, Adjacency, and Semantic Issues

# Import the corrected classes
from improved_geometric_solver import ImprovedGeometricConstraintSolver
from hybrid_floor_plan_generator import ImprovedHybridFloorPlanGenerator

# You'll also need these base classes from the original implementation
from floor_plan_implementation import RoomSpec, BuildingBoundary, RoomGraph, ConsFormerEncoder

def run_corrected_floor_plan_demo():
    """Run the corrected floor plan generation demo"""

    print("=== CORRECTED Floor Plan Generation Demo ===\n")

    # Initialize the improved generator
    generator = ImprovedHybridFloorPlanGenerator(device='cpu')

    # Define boundary (same as user's example)
    boundary = [(0,0), (25,0), (25,16), (0,16)]  # 25m × 16m = 400 m²

    # Define rooms with realistic constraints for 400 sq meter building
    rooms = [
    {
        'room_id': 'living_room',
        'room_type': 'living',
        'min_area': 50.0,
        'max_area': 80.0
    },
    {
        'room_id': 'kitchen',
        'room_type': 'kitchen',
        'min_area': 20.0,
        'max_area': 40.0
    },
    {
        'room_id': 'bathroom',
        'room_type': 'bathroom',
        'min_area': 8.0,
        'max_area': 15.0
    },
    {
        'room_id': 'bedroom',
        'room_type': 'bedroom',
        'min_area': 30.0,
        'max_area': 60.0
    },
    {
        'room_id': 'study',
        'room_type': 'study',
        'min_area': 12.0,
        'max_area': 25.0
    }
    ]


    # Simplify adjacency constraints (remove duplicates and conflicts)
    adjacencies = [
    ('living_room', 'kitchen', 0.9),   # Should be close
    ('living_room', 'study', 0.6),     # Optional but useful
    ('living_room', 'bedroom', 0.4),   # Weak
    ('kitchen', 'bathroom', 0.5),      # Practical, medium
    ('bedroom', 'bathroom', 0.8),      # Strong – bathrooms near bedrooms
    ('study', 'bedroom', 0.3)          # Weak
    ]


    print(f"Building: {20}x{20} meters ({400} sq meters)")
    print(f"Total room area range: {sum(r['min_area'] for r in rooms):.0f} - {sum(r['max_area'] for r in rooms):.0f} sq meters")
    print(f"Area utilization: {sum(r['min_area'] for r in rooms)/400*100:.1f}% - {sum(r['max_area'] for r in rooms)/400*100:.1f}%")

    try:
        # Step 1: Load constraints with validation
        print("\nStep 1: Loading and validating constraints...")
        room_graph, building_boundary = generator.load_constraints(boundary, rooms, adjacencies)

        # Step 2: Use improved geometric constraint propagation
        print("\nStep 2: Solving geometric constraints with CP-SAT...")
        initial_layout = generator.geometric_constraint_propagation_improved(room_graph, building_boundary)

        if not initial_layout:
            print("ERROR: Failed to generate initial layout!")
            return None

        # Step 3: Load transformer model
        print("\nStep 3: Loading ConsFormer model...")
        model = generator.load_trained_consformer()

        # Step 4: Run improved iterative refinement
        print("\nStep 4: Running iterative refinement with constraint enforcement...")
        final_layout = generator.iterative_refinement_loop_improved(room_graph, initial_layout, model, max_iterations=10)

        # Step 5: Generate walls
        print("\nStep 5: Generating walls...")
        walls = generator.generate_walls(final_layout, building_boundary)  # Note: using corrected method name

        # Step 6: Enhanced visualization
        print("\nStep 6: Creating enhanced visualization...")
        generator.visualize_layout(final_layout, building_boundary, walls, save_path='corrected_floor_plan.png')

        # Step 7: Detailed validation and metrics
        print("\nStep 7: Validation and quality metrics...")

        # Check final layout quality
        quality_score = generator._evaluate_layout_quality(final_layout, room_graph)
        print(f"Final layout quality score: {quality_score:.3f}/1.0")

        # Detailed validation
        print("\nDetailed Layout Analysis:")
        total_area = 0
        for room_id, (x, y, w, h) in final_layout.items():
            area = w * h
            total_area += area
            room_data = room_graph.graph.nodes[room_id]
            within_constraints = room_data['min_area'] <= area <= room_data['max_area']

            print(f"  {room_id}:")
            print(f"    Position: ({x:.1f}, {y:.1f})")
            print(f"    Size: {w:.1f}x{h:.1f} = {area:.1f} sq meters")
            print(f"    Constraints: [{room_data['min_area']:.1f}, {room_data['max_area']:.1f}] - {'✓' if within_constraints else '✗'}")

            # Check boundary violations
            if x < 0 or y < 0 or x + w > 20 or y + h > 20:
                print(f"    ⚠️  Boundary violation detected!")
            else:
                print(f"    ✓ Within boundaries")

        print(f"\nTotal room area: {total_area:.1f} sq meters ({total_area/400*100:.1f}% utilization)")

        # Check adjacencies
        print("\nAdjacency Analysis:")
        satisfied_adjacencies = 0
        total_adjacencies = len(adjacencies)

        for room1, room2, weight in adjacencies:
            if room1 in final_layout and room2 in final_layout:
                is_adjacent = generator._rooms_adjacent(final_layout[room1], final_layout[room2], threshold=1.0)
                satisfied_adjacencies += is_adjacent
                status = "✓" if is_adjacent else "✗"
                print(f"  {room1} ↔ {room2} (weight: {weight}): {status}")

        print(f"\nAdjacency satisfaction: {satisfied_adjacencies}/{total_adjacencies} ({satisfied_adjacencies/total_adjacencies*100:.1f}%)")

        # Check overlaps
        print("\nOverlap Analysis:")
        room_ids = list(final_layout.keys())
        overlaps = 0
        for i, room1 in enumerate(room_ids):
            for room2 in room_ids[i+1:]:
                if generator._rooms_overlap(final_layout[room1], final_layout[room2]):
                    overlaps += 1
                    print(f"  ⚠️  {room1} overlaps with {room2}")

        if overlaps == 0:
            print("  ✓ No overlaps detected")
        else:
            print(f"  ✗ {overlaps} overlaps found")

        print("\n" + "="*50)
        print("SUMMARY:")
        print(f"✓ Boundary violations: {'None' if all(x >= 0 and y >= 0 and x+w <= 20 and y+h <= 20 for x,y,w,h in final_layout.values()) else 'Found'}")
        print(f"✓ Room overlaps: {'None' if overlaps == 0 else f'{overlaps} found'}")  
        print(f"✓ Adjacency satisfaction: {satisfied_adjacencies/total_adjacencies*100:.1f}%")
        print(f"✓ Area utilization: {total_area/400*100:.1f}%")
        print(f"✓ Overall quality score: {quality_score:.3f}/1.0")
        print("="*50)

        return final_layout

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Usage instructions
usage_instructions = """
USAGE INSTRUCTIONS FOR CORRECTED IMPLEMENTATION:

1. INSTALL DEPENDENCIES:
   pip install ortools torch networkx matplotlib scipy numpy

2. RUN THE CORRECTED DEMO:
   python corrected_floor_plan_demo.py

3. KEY IMPROVEMENTS MADE:

   A) BOUNDARY ENFORCEMENT:
   - CP-SAT constraints ensure rooms stay within building boundary
   - Boundary validation in constraint projection
   - Coordinate system properly aligned

   B) ADJACENCY HANDLING:
   - Strong adjacency constraints (weight > 0.7) enforced in CP-SAT
   - Adjacency enforcement in iterative refinement
   - Priority-based room positioning

   C) SEMANTIC IMPROVEMENTS:
   - Room type influences placement priority
   - Bathroom placement near edges
   - Living room gets central positioning preference
   - Area constraints strictly enforced

   D) CONSTRAINT SATISFACTION:
   - OR-Tools CP-SAT for optimal room placement
   - Multi-objective optimization (area, adjacency, compactness)
   - Fallback algorithms for infeasible cases

   E) QUALITY ASSESSMENT:
   - Comprehensive layout quality scoring
   - Detailed validation with specific issue reporting
   - Adjacency satisfaction tracking

4. EXPECTED IMPROVEMENTS:
   - Rooms will respect building boundaries
   - High-weight adjacencies will be satisfied
   - No overlapping rooms
   - Better area utilization
   - More architecturally sensible layouts

5. METHOD NAMING FIXED:
   - generate_walls() method now exists (was wall_generator)
   - All method calls match implementation
"""

if __name__ == "__main__":
    print(usage_instructions)
    print("\n" + "="*60)

    # Run the demo
    result = run_corrected_floor_plan_demo()

    if result:
        print("\n🎉 Corrected floor plan generation completed successfully!")
    else:
        print("\n❌ Floor plan generation failed. Check error messages above.")
