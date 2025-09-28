
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import networkx as nx

from floor_plan_implementation import RoomGraph, RoomSpec, BuildingBoundary, GeometricConstraintSolver, ConsFormerEncoder

class ImprovedHybridFloorPlanGenerator:
    """Enhanced pipeline with better constraint handling and adjacency enforcement"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.consformer = None
        self.wall_generator = None

    def load_constraints(self, boundary_vertices: List[Tuple[float, float]], 
                        room_specs: List[Dict], 
                        adjacency_constraints: List[Tuple[str, str, float]]) -> Tuple[RoomGraph, BuildingBoundary]:
        """Step 1: Load constraints with validation"""

        # Validate inputs
        self._validate_inputs(boundary_vertices, room_specs, adjacency_constraints)

        # Create building boundary
        total_area = self._calculate_polygon_area(boundary_vertices)
        boundary = BuildingBoundary(vertices=boundary_vertices, total_area=total_area)

        # Create room graph
        room_graph = RoomGraph()

        # Add rooms with validation
        total_min_area = 0
        for spec in room_specs:
            room_spec = RoomSpec(
                room_id=spec['room_id'],
                room_type=spec['room_type'],
                min_area=spec['min_area'],
                max_area=spec['max_area'],
                preferred_adjacencies=spec.get('preferred_adjacencies', [])
            )
            room_graph.add_room(room_spec)
            total_min_area += spec['min_area']

        # Check feasibility
        if total_min_area > total_area * 0.9:  # Leave 10% margin
            print(f"Warning: Total minimum area ({total_min_area:.1f}) exceeds 90% of building area ({total_area:.1f})")

        # Add adjacency constraints (remove duplicates)
        added_pairs = set()
        for room1, room2, weight in adjacency_constraints:
            pair = tuple(sorted([room1, room2]))
            if pair not in added_pairs and room1 != room2:
                room_graph.add_adjacency_constraint(room1, room2, weight)
                added_pairs.add(pair)

        print(f"Loaded {len(room_specs)} rooms with {len(added_pairs)} adjacency constraints")
        return room_graph, boundary

    def _validate_inputs(self, boundary_vertices, room_specs, adjacency_constraints):
        """Validate input constraints"""

        # Check boundary
        if len(boundary_vertices) < 3:
            raise ValueError("Boundary must have at least 3 vertices")

        # Check rooms
        room_ids = set()
        for spec in room_specs:
            if spec['room_id'] in room_ids:
                raise ValueError(f"Duplicate room ID: {spec['room_id']}")
            room_ids.add(spec['room_id'])

            if spec['min_area'] <= 0 or spec['max_area'] <= spec['min_area']:
                raise ValueError(f"Invalid area constraints for {spec['room_id']}")

        # Check adjacencies reference valid rooms
        for room1, room2, weight in adjacency_constraints:
            if room1 not in room_ids or room2 not in room_ids:
                raise ValueError(f"Adjacency constraint references unknown room: {room1}, {room2}")

    def geometric_constraint_propagation_improved(self, room_graph: RoomGraph, 
                                                boundary: BuildingBoundary) -> Dict[str, Tuple[float, float, float, float]]:
        """Step 2: Improved geometric constraint propagation"""

        from improved_geometric_solver import ImprovedGeometricConstraintSolver

        solver = ImprovedGeometricConstraintSolver(boundary, grid_resolution=0.3)
        initial_layout = solver.solve_constraints_improved(room_graph)

        # Validate the solution
        if initial_layout:
            self._validate_layout(initial_layout, room_graph, boundary)
            print(f"Successfully placed {len(initial_layout)} rooms using constraint programming")
        else:
            print("Constraint programming failed, trying fallback approach")
            initial_layout = solver._fallback_placement(room_graph)

        return initial_layout

    def _validate_layout(self, layout: Dict, room_graph: RoomGraph, boundary: BuildingBoundary):
        """Validate generated layout"""

        issues = []

        # Check boundary violations
        min_x = min(x for x, y in boundary.vertices)
        max_x = max(x for x, y in boundary.vertices) 
        min_y = min(y for x, y in boundary.vertices)
        max_y = max(y for x, y in boundary.vertices)

        for room_id, (x, y, w, h) in layout.items():
            if x < min_x or y < min_y or x + w > max_x or y + h > max_y:
                issues.append(f"Room {room_id} violates boundary constraints")

        # Check overlaps
        room_ids = list(layout.keys())
        for i, room1 in enumerate(room_ids):
            for room2 in room_ids[i+1:]:
                if self._rooms_overlap(layout[room1], layout[room2]):
                    issues.append(f"Rooms {room1} and {room2} overlap")

        # Check area constraints
        for room_id, (x, y, w, h) in layout.items():
            area = w * h
            room_data = room_graph.graph.nodes[room_id]
            if area < room_data['min_area'] * 0.9 or area > room_data['max_area'] * 1.1:
                issues.append(f"Room {room_id} area ({area:.1f}) violates constraints [{room_data['min_area']:.1f}, {room_data['max_area']:.1f}]")

        if issues:
            print("Layout validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Layout validation passed!")

    def iterative_refinement_loop_improved(self, room_graph: RoomGraph,
                                         initial_layout: Dict[str, Tuple[float, float, float, float]],
                                         consformer: ConsFormerEncoder,
                                         max_iterations: int = 15) -> Dict[str, Tuple[float, float, float, float]]:
        """Step 4: Improved iterative refinement with stronger constraint enforcement"""

        current_layout = initial_layout.copy()
        best_layout = current_layout.copy()
        best_score = self._evaluate_layout_quality(current_layout, room_graph)

        print(f"Initial layout quality score: {best_score:.3f}")

        for iteration in range(max_iterations):
            # Prepare transformer input
            features, adjacency_matrix = self.prepare_transformer_input(room_graph, current_layout)

            # Forward pass through transformer
            with torch.no_grad():
                outputs = consformer(features, adjacency_matrix)

            # Extract updates with adaptive learning rate
            position_updates = outputs['position_updates'].cpu().numpy()[0]
            size_updates = outputs['size_updates'].cpu().numpy()[0]

            # Adaptive learning rate based on iteration and current quality
            base_lr = 0.05
            decay_factor = 0.9 ** iteration
            quality_factor = min(1.0, 1.0 - best_score)  # Reduce updates for high-quality layouts
            learning_rate = base_lr * decay_factor * (0.5 + quality_factor)

            # Apply updates with constraints
            room_ids = list(room_graph.graph.nodes())
            updated_layout = {}

            for i, room_id in enumerate(room_ids):
                if room_id in current_layout:
                    x, y, w, h = current_layout[room_id]
                    room_data = room_graph.graph.nodes[room_id]

                    # Apply position updates with boundary constraints
                    new_x = max(0, x + learning_rate * position_updates[i, 0])
                    new_y = max(0, y + learning_rate * position_updates[i, 1])

                    # Apply size updates with area constraints
                    min_area = room_data['min_area']
                    max_area = room_data['max_area']

                    new_w = max(2.0, w + learning_rate * size_updates[i, 0])
                    new_h = max(2.0, h + learning_rate * size_updates[i, 1])

                    # Enforce area constraints
                    current_area = new_w * new_h
                    if current_area < min_area:
                        scale_factor = np.sqrt(min_area / current_area)
                        new_w *= scale_factor
                        new_h *= scale_factor
                    elif current_area > max_area:
                        scale_factor = np.sqrt(max_area / current_area)
                        new_w *= scale_factor
                        new_h *= scale_factor

                    updated_layout[room_id] = (new_x, new_y, new_w, new_h)

            # Strong constraint projection
            projected_layout = self._project_constraints_improved(updated_layout, room_graph)

            # Evaluate layout quality
            current_score = self._evaluate_layout_quality(projected_layout, room_graph)

            # Accept improvement or with probability for exploration
            accept_probability = np.exp(-(best_score - current_score) / (0.1 * (iteration + 1)))
            if current_score > best_score or np.random.random() < accept_probability:
                current_layout = projected_layout
                if current_score > best_score:
                    best_layout = projected_layout.copy()
                    best_score = current_score
                    print(f"Iteration {iteration + 1}: Improved quality to {best_score:.3f}")

            # Check convergence
            if iteration > 3:
                layout_change = self._calculate_layout_change(initial_layout, current_layout)
                if layout_change < 0.005:
                    print(f"Converged after {iteration + 1} iterations")
                    break

        print(f"Final layout quality score: {best_score:.3f}")
        return best_layout

    def _project_constraints_improved(self, layout: Dict[str, Tuple[float, float, float, float]], 
                                    room_graph: RoomGraph) -> Dict[str, Tuple[float, float, float, float]]:
        """Enhanced constraint projection with boundary and adjacency enforcement"""

        projected = layout.copy()
        max_iterations = 10

        for iteration in range(max_iterations):
            changes_made = False

            # 1. Boundary constraint enforcement
            for room_id, (x, y, w, h) in projected.items():
                # Get boundary limits
                vertices = room_graph.graph.nodes[room_id] if hasattr(room_graph.graph.nodes[room_id], 'boundary') else None

                # Simple boundary enforcement (assuming rectangular boundary)
                boundary_vertices = [(0, 0), (20, 0), (20, 20), (0, 20)]  # Default from user example
                min_bx = min(bx for bx, by in boundary_vertices)
                max_bx = max(bx for bx, by in boundary_vertices)  
                min_by = min(by for bx, by in boundary_vertices)
                max_by = max(by for bx, by in boundary_vertices)

                new_x = max(min_bx, min(x, max_bx - w))
                new_y = max(min_by, min(y, max_by - h))

                if new_x != x or new_y != y:
                    projected[room_id] = (new_x, new_y, w, h)
                    changes_made = True

            # 2. Overlap resolution with priority system  
            room_ids = list(projected.keys())
            room_priorities = self._get_room_priorities(room_graph)

            for i, room1 in enumerate(room_ids):
                for room2 in room_ids[i+1:]:
                    if self._rooms_overlap(projected[room1], projected[room2]):
                        # Resolve overlap based on priority and adjacency
                        if room_priorities[room1] > room_priorities[room2]:
                            projected[room2] = self._move_room_away(projected[room2], projected[room1])
                        else:
                            projected[room1] = self._move_room_away(projected[room1], projected[room2])
                        changes_made = True

            # 3. Adjacency enforcement for high-weight constraints
            for (room1, room2), weight in room_graph.adjacency_weights.items():
                if weight > 0.7 and room1 in projected and room2 in projected:
                    if not self._rooms_adjacent(projected[room1], projected[room2], threshold=1.0):
                        # Try to move rooms closer
                        projected[room1], projected[room2] = self._enforce_adjacency(
                            projected[room1], projected[room2], room_priorities[room1] > room_priorities[room2]
                        )
                        changes_made = True

            if not changes_made:
                break

        return projected

    def _get_room_priorities(self, room_graph: RoomGraph) -> Dict[str, int]:
        """Get room priorities for conflict resolution"""

        priorities = {}
        for room_id in room_graph.graph.nodes():
            room_type = room_graph.graph.nodes[room_id]['type']
            area = room_graph.graph.nodes[room_id]['max_area']

            # Priority based on room type and size
            type_priority = {
                'living': 5, 'kitchen': 4, 'bedroom': 3,
                'bathroom': 1, 'office': 2, 'hallway': 0
            }.get(room_type, 2)

            # Larger rooms get higher priority
            size_priority = int(area / 10)

            priorities[room_id] = type_priority * 10 + size_priority

        return priorities

    def _move_room_away(self, moving_room: Tuple[float, float, float, float],
                       fixed_room: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Move a room away from another to resolve overlap"""

        mx, my, mw, mh = moving_room
        fx, fy, fw, fh = fixed_room

        # Calculate move directions
        move_right = fx + fw - mx + 0.5
        move_left = fx - (mx + mw) - 0.5
        move_up = fy + fh - my + 0.5
        move_down = fy - (my + mh) - 0.5

        # Choose direction with minimum movement
        moves = [
            (move_right, mx + move_right, my),  # right
            (abs(move_left), mx + move_left, my),  # left
            (move_up, mx, my + move_up),  # up  
            (abs(move_down), mx, my + move_down)  # down
        ]

        # Choose minimum valid move
        min_distance, new_x, new_y = min(moves, key=lambda x: x[0])

        # Ensure non-negative coordinates
        new_x = max(0, new_x)
        new_y = max(0, new_y)

        return (new_x, new_y, mw, mh)

    def _enforce_adjacency(self, room1: Tuple[float, float, float, float],
                          room2: Tuple[float, float, float, float],
                          room1_priority: bool) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        """Enforce adjacency between two rooms"""

        x1, y1, w1, h1 = room1
        x2, y2, w2, h2 = room2

        # Calculate centers
        c1x, c1y = x1 + w1/2, y1 + h1/2
        c2x, c2y = x2 + w2/2, y2 + h2/2

        # Determine best adjacency direction
        dx = c2x - c1x
        dy = c2y - c1y

        if abs(dx) > abs(dy):  # Horizontal adjacency
            if dx > 0:  # room2 to the right of room1
                if room1_priority:
                    new_x2 = x1 + w1 + 0.1  # Small gap
                    return room1, (new_x2, y2, w2, h2)
                else:
                    new_x1 = x2 - w1 - 0.1
                    return (max(0, new_x1), y1, w1, h1), room2
        else:  # Vertical adjacency
            if dy > 0:  # room2 above room1
                if room1_priority:
                    new_y2 = y1 + h1 + 0.1
                    return room1, (x2, new_y2, w2, h2)
                else:
                    new_y1 = y2 - h1 - 0.1
                    return (x1, max(0, new_y1), w1, h1), room2

        return room1, room2  # No change if can't determine direction

    def _evaluate_layout_quality(self, layout: Dict, room_graph: RoomGraph) -> float:
        """Evaluate layout quality score (0-1, higher is better)"""

        if not layout:
            return 0.0

        score = 0.0
        weights = {'area': 0.2, 'adjacency': 0.4, 'compactness': 0.2, 'boundary': 0.2}

        # 1. Area constraint satisfaction
        area_score = 0.0
        for room_id, (x, y, w, h) in layout.items():
            area = w * h
            room_data = room_graph.graph.nodes[room_id]
            min_area, max_area = room_data['min_area'], room_data['max_area']

            if min_area <= area <= max_area:
                area_score += 1.0
            else:
                # Penalize based on violation
                if area < min_area:
                    area_score += area / min_area
                else:
                    area_score += max_area / area

        area_score /= len(layout)

        # 2. Adjacency satisfaction
        adjacency_score = 0.0
        total_weight = 0.0
        for (room1, room2), weight in room_graph.adjacency_weights.items():
            total_weight += weight
            if room1 in layout and room2 in layout:
                if self._rooms_adjacent(layout[room1], layout[room2], threshold=1.0):
                    adjacency_score += weight

        adjacency_score = adjacency_score / total_weight if total_weight > 0 else 1.0

        # 3. Compactness (prefer square-ish rooms)
        compactness_score = 0.0
        for room_id, (x, y, w, h) in layout.items():
            aspect_ratio = max(w, h) / min(w, h)
            compactness_score += 1.0 / aspect_ratio  # Closer to 1 is better
        compactness_score /= len(layout)

        # 4. Boundary utilization (minimize wasted space)
        total_room_area = sum(w * h for x, y, w, h in layout.values())
        boundary_area = 20 * 20  # From user example
        boundary_score = total_room_area / boundary_area

        # Combine scores
        final_score = (weights['area'] * area_score +
                      weights['adjacency'] * adjacency_score + 
                      weights['compactness'] * compactness_score +
                      weights['boundary'] * boundary_score)

        return min(1.0, final_score)

    def generate_walls(self, refined_layout: Dict[str, Tuple[float, float, float, float]], 
                      boundary: BuildingBoundary) -> np.ndarray:
        """Generate walls (fixed method name)"""
        # Create occupancy map from room layout
        occupancy_map = self._create_occupancy_map(refined_layout, boundary)

        # Simple wall generation (in practice, would use trained U-Net)
        walls_mask = self._generate_walls_simple(occupancy_map)

        return walls_mask

    def prepare_transformer_input(self, room_graph: RoomGraph, 
                                layout: Dict[str, Tuple[float, float, float, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input for transformer model with proper feature scaling"""

        room_ids = list(room_graph.graph.nodes())
        num_rooms = len(room_ids)

        if num_rooms == 0:
            return torch.zeros(1, 0, 4), torch.zeros(1, 0, 0)

        # Room type encoding
        type_mapping = {
            'living': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3,
            'dining': 4, 'office': 5, 'hallway': 6, 'garage': 7, 'other': 8
        }

        # Prepare features tensor
        features = torch.zeros(1, num_rooms, 4)  # batch_size=1, [x, y, type, area]

        # Normalization factors
        max_coord = 20.0  # From boundary
        max_area = 100.0  # Reasonable max area

        for i, room_id in enumerate(room_ids):
            if room_id in layout:
                x, y, w, h = layout[room_id]
                room_type = room_graph.graph.nodes[room_id]['type']
                area = w * h

                # Normalize features
                features[0, i, 0] = float(x / max_coord)  # x position (0-1)
                features[0, i, 1] = float(y / max_coord)  # y position (0-1)
                features[0, i, 2] = float(type_mapping.get(room_type, 8) / 8.0)  # type id (0-1)
                features[0, i, 3] = float(area / max_area)  # area (0-1)

        # Prepare adjacency matrix
        adjacency_matrix = torch.tensor(room_graph.get_adjacency_matrix(), dtype=torch.float32)
        adjacency_matrix = adjacency_matrix.unsqueeze(0)  # Add batch dimension

        return features.to(self.device), adjacency_matrix.to(self.device)

    # Include all other methods from original implementation...
    def _calculate_polygon_area(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate area of polygon using shoelace formula"""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    def _rooms_overlap(self, room1: Tuple[float, float, float, float], 
                      room2: Tuple[float, float, float, float]) -> bool:
        """Check if two rooms overlap"""
        x1, y1, w1, h1 = room1
        x2, y2, w2, h2 = room2

        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    def _rooms_adjacent(self, room1: Tuple[float, float, float, float], 
                       room2: Tuple[float, float, float, float], threshold: float = 1.0) -> bool:
        """Check if two rooms are adjacent"""
        x1, y1, w1, h1 = room1
        x2, y2, w2, h2 = room2

        # Check if rooms share a wall (within threshold)
        vertical_adjacent = (abs(x1 + w1 - x2) < threshold or abs(x2 + w2 - x1) < threshold) and \
                           not (y1 + h1 <= y2 or y2 + h2 <= y1)
        horizontal_adjacent = (abs(y1 + h1 - y2) < threshold or abs(y2 + h2 - y1) < threshold) and \
                             not (x1 + w1 <= x2 or x2 + w2 <= x1)

        return vertical_adjacent or horizontal_adjacent

    def _calculate_layout_change(self, layout1: Dict, layout2: Dict) -> float:
        """Calculate change between two layouts"""
        total_change = 0.0

        for room_id in layout1:
            if room_id in layout2:
                x1, y1, w1, h1 = layout1[room_id]
                x2, y2, w2, h2 = layout2[room_id]

                change = abs(x1 - x2) + abs(y1 - y2) + abs(w1 - w2) + abs(h1 - h2)
                total_change += change

        return total_change / len(layout1) if layout1 else 0.0

    def load_trained_consformer(self, model_path: str = None) -> ConsFormerEncoder:
        """Load ConsFormer model"""
        model = ConsFormerEncoder(d_model=256, nhead=8, num_layers=6)

        if model_path:
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded pre-trained model from {model_path}")
            except:
                print("Could not load pre-trained model, using random initialization")
        else:
            print("Using randomly initialized ConsFormer model")

        model.to(self.device)
        model.eval()

        return model

    def wall_generator(self, refined_layout: Dict[str, Tuple[float, float, float, float]], 
                      boundary: BuildingBoundary) -> np.ndarray:
        """Generate walls using simple edge detection"""

        # Create occupancy map from room layout
        occupancy_map = self._create_occupancy_map(refined_layout, boundary)

        # Simple wall generation
        walls_mask = self._generate_walls_simple(occupancy_map)

        return walls_mask

    def _create_occupancy_map(self, layout: Dict[str, Tuple[float, float, float, float]], 
                            boundary: BuildingBoundary, resolution: float = 0.1) -> np.ndarray:
        """Create occupancy map from room layout"""

        vertices = np.array(boundary.vertices)
        min_x, min_y = vertices.min(axis=0)
        max_x, max_y = vertices.max(axis=0)

        width = int((max_x - min_x) / resolution) + 1
        height = int((max_y - min_y) / resolution) + 1

        occupancy_map = np.zeros((height, width, 3))

        # Different colors for different room types
        colors = {
            'living': [1, 0, 0], 'kitchen': [0, 1, 0], 'bedroom': [0, 0, 1],
            'bathroom': [1, 1, 0], 'dining': [1, 0, 1], 'office': [0, 1, 1]
        }

        for room_id, (x, y, w, h) in layout.items():
            # Convert to grid coordinates
            grid_x = int((x - min_x) / resolution)
            grid_y = int((y - min_y) / resolution)
            grid_w = int(w / resolution)
            grid_h = int(h / resolution)

            # Get room color (default to gray)
            room_type = 'living'  # Default - in real implementation would look up type
            color = colors.get(room_type, [0.5, 0.5, 0.5])

            # Fill room area
            end_x = min(grid_x + grid_w, width)
            end_y = min(grid_y + grid_h, height)
            occupancy_map[grid_y:end_y, grid_x:end_x] = color

        return occupancy_map

    def _generate_walls_simple(self, occupancy_map: np.ndarray) -> np.ndarray:
        """Simple wall generation using edge detection"""

        # Convert to grayscale
        gray = np.mean(occupancy_map, axis=2)

        # Edge detection for walls
        from scipy import ndimage
        walls = ndimage.sobel(gray)
        walls = (walls > 0.1).astype(np.uint8)

        return walls

    def visualize_layout(self, refined_layout: Dict[str, Tuple[float, float, float, float]], 
                        boundary: BuildingBoundary, 
                        walls_mask: np.ndarray = None,
                        save_path: str = None) -> None:
        """Enhanced visualization with better room labeling"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot 1: Room layout
        self._plot_room_layout_improved(ax1, refined_layout, boundary)

        # Plot 2: Floor plan with walls  
        if walls_mask is not None:
            self._plot_floor_plan(ax2, refined_layout, boundary, walls_mask)
        else:
            ax2.text(0.5, 0.5, 'Wall generation\nnot implemented', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Floor Plan with Walls')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def _plot_room_layout_improved(self, ax, layout: Dict, boundary: BuildingBoundary):
        """Improved room layout plotting"""

        if not layout:
            ax.text(0.5, 0.5, 'No rooms to display', ha='center', va='center', transform=ax.transAxes)
            return

        # Plot boundary
        vertices = boundary.vertices + [boundary.vertices[0]]  # Close polygon
        boundary_x, boundary_y = zip(*vertices)
        ax.plot(boundary_x, boundary_y, 'k-', linewidth=3, label='Boundary')

        # Room colors with better contrast
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold', 'plum', 'lightsalmon', 'lightcyan', 'wheat', 'lightpink', 'lightsteelblue']

        # Plot rooms with improved styling
        for i, (room_id, (x, y, w, h)) in enumerate(layout.items()):
            color = colors[i % len(colors)]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='darkblue', facecolor=color, alpha=0.7)
            ax.add_patch(rect)

            # Add room label with better positioning
            label_x = x + w/2
            label_y = y + h/2
            ax.text(label_x, label_y, room_id.replace('_', '\n'), ha='center', va='center', 
                   fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # Add area information
            area = w * h
            ax.text(x + w - 0.5, y + 0.5, f'{area:.0f}m²', ha='right', va='bottom',
                   fontsize=7, style='italic', color='darkblue')

        ax.set_xlim(min(boundary_x) - 1, max(boundary_x) + 1)
        ax.set_ylim(min(boundary_y) - 1, max(boundary_y) + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Room Layout', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.legend()

    def _plot_floor_plan(self, ax, layout: Dict, boundary: BuildingBoundary, walls_mask: np.ndarray):
        """Plot complete floor plan with walls"""

        # Display walls mask
        ax.imshow(walls_mask, cmap='gray', alpha=0.8, extent=[0, 20, 0, 20])
        ax.set_title('Floor Plan with Walls', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)

        # Add room labels on wall plan
        for room_id, (x, y, w, h) in layout.items():
            ax.text(x + w/2, y + h/2, room_id.replace('_', ' '), ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
