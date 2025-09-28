
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from ortools.sat.python import cp_model
import networkx as nx
from floor_plan_implementation import BuildingBoundary, RoomGraph

class ImprovedGeometricConstraintSolver:
    """Enhanced geometric constraint solver with proper boundary and adjacency handling"""

    def __init__(self, boundary: BuildingBoundary, grid_resolution: float = 0.5):
        self.boundary = boundary
        self.grid_resolution = grid_resolution
        self.min_x, self.min_y, self.max_x, self.max_y = self._get_boundary_limits()
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y

    def _get_boundary_limits(self):
        """Get boundary bounding box"""
        vertices = np.array(self.boundary.vertices)
        min_x, min_y = vertices.min(axis=0)
        max_x, max_y = vertices.max(axis=0)
        return min_x, min_y, max_x, max_y

    def solve_constraints_improved(self, room_graph: RoomGraph) -> Dict[str, Tuple[float, float, float, float]]:
        """Solve constraints using CP-SAT optimizer for better results"""

        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        # Get room information
        rooms = list(room_graph.graph.nodes())
        num_rooms = len(rooms)

        if num_rooms == 0:
            return {}

        # Convert to grid units for integer programming
        grid_width = int(self.width / self.grid_resolution)
        grid_height = int(self.height / self.grid_resolution)

        # Decision variables for each room: x, y, width, height (in grid units)
        room_vars = {}
        for room_id in rooms:
            room_data = room_graph.graph.nodes[room_id]

            # Calculate grid-based min/max dimensions
            min_area_grid = int(room_data['min_area'] / (self.grid_resolution ** 2))
            max_area_grid = int(room_data['max_area'] / (self.grid_resolution ** 2))
            min_dim_grid = max(2, int(np.sqrt(min_area_grid)))
            max_dim_grid = min(grid_width, grid_height, int(np.sqrt(max_area_grid)) + 5)

            room_vars[room_id] = {
                'x': model.NewIntVar(0, grid_width - min_dim_grid, f'{room_id}_x'),
                'y': model.NewIntVar(0, grid_height - min_dim_grid, f'{room_id}_y'),
                'width': model.NewIntVar(min_dim_grid, max_dim_grid, f'{room_id}_width'),
                'height': model.NewIntVar(min_dim_grid, max_dim_grid, f'{room_id}_height')
            }

            # Area constraints
            area_var = model.NewIntVar(min_area_grid, max_area_grid, f'{room_id}_area')
            model.AddMultiplicationEquality(area_var, [room_vars[room_id]['width'], room_vars[room_id]['height']])

            # Boundary constraints - rooms must be fully inside
            model.Add(room_vars[room_id]['x'] + room_vars[room_id]['width'] <= grid_width)
            model.Add(room_vars[room_id]['y'] + room_vars[room_id]['height'] <= grid_height)
            model.Add(room_vars[room_id]['x'] >= 0)
            model.Add(room_vars[room_id]['y'] >= 0)

        # Non-overlap constraints
        self._add_non_overlap_constraints_improved(model, room_vars, rooms, grid_width, grid_height)

        # Adjacency constraints
        self._add_adjacency_constraints_improved(model, room_vars, room_graph)

        # Semantic placement constraints
        self._add_semantic_constraints(model, room_vars, room_graph)

        # Objective: minimize total area waste and maximize adjacency satisfaction
        self._add_optimization_objective(model, room_vars, room_graph)

        # Solve
        solver.parameters.max_time_in_seconds = 30.0  # Time limit
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return self._extract_solution(solver, room_vars)
        else:
            print(f"Constraint solver failed with status: {solver.StatusName(status)}")
            # Fallback to simple placement
            return self._fallback_placement(room_graph)

    def _add_non_overlap_constraints_improved(self, model, room_vars, rooms, grid_width, grid_height):
        """Add improved non-overlapping constraints"""

        for i, room1 in enumerate(rooms):
            for room2 in rooms[i+1:]:
                r1_vars = room_vars[room1]
                r2_vars = room_vars[room2]

                # Use no_overlap constraint which is more efficient
                # Create end variables for intervals first
                x_end1 = model.NewIntVar(0, grid_width, f'{room1}_x_end')
                y_end1 = model.NewIntVar(0, grid_height, f'{room1}_y_end')
                x_end2 = model.NewIntVar(0, grid_width, f'{room2}_x_end')
                y_end2 = model.NewIntVar(0, grid_height, f'{room2}_y_end')
                
                # Add constraints to define end variables
                model.Add(x_end1 == r1_vars['x'] + r1_vars['width'])
                model.Add(y_end1 == r1_vars['y'] + r1_vars['height'])
                model.Add(x_end2 == r2_vars['x'] + r2_vars['width'])
                model.Add(y_end2 == r2_vars['y'] + r2_vars['height'])
                
                # Create interval variables using the end variables
                x_interval1 = model.NewIntervalVar(r1_vars['x'], r1_vars['width'], x_end1, f'{room1}_x_interval')
                y_interval1 = model.NewIntervalVar(r1_vars['y'], r1_vars['height'], y_end1, f'{room1}_y_interval')
                x_interval2 = model.NewIntervalVar(r2_vars['x'], r2_vars['width'], x_end2, f'{room2}_x_interval')
                y_interval2 = model.NewIntervalVar(r2_vars['y'], r2_vars['height'], y_end2, f'{room2}_y_interval')
                
                # Add no-overlap constraint
                model.AddNoOverlap2D([x_interval1, x_interval2], [y_interval1, y_interval2])

    def _add_adjacency_constraints_improved(self, model, room_vars, room_graph):
        """Add adjacency constraints to encourage room connections"""

        adjacency_bonuses = []

        for (room1, room2), weight in room_graph.adjacency_weights.items():
            if weight > 0.3:  # Only enforce strong adjacencies
                r1_vars = room_vars[room1]
                r2_vars = room_vars[room2]

                # Create boolean variables for adjacency conditions
                adjacent_vertical = model.NewBoolVar(f'{room1}_{room2}_adj_vert')
                adjacent_horizontal = model.NewBoolVar(f'{room1}_{room2}_adj_horiz')

                # Vertical adjacency: rooms share vertical edge
                model.Add(r1_vars['x'] + r1_vars['width'] == r2_vars['x']).OnlyEnforceIf(adjacent_vertical)
                model.Add(r1_vars['y'] < r2_vars['y'] + r2_vars['height']).OnlyEnforceIf(adjacent_vertical)
                model.Add(r2_vars['y'] < r1_vars['y'] + r1_vars['height']).OnlyEnforceIf(adjacent_vertical)

                # Alternative: r2 adjacent to r1
                adjacent_vertical_alt = model.NewBoolVar(f'{room2}_{room1}_adj_vert')
                model.Add(r2_vars['x'] + r2_vars['width'] == r1_vars['x']).OnlyEnforceIf(adjacent_vertical_alt)
                model.Add(r1_vars['y'] < r2_vars['y'] + r2_vars['height']).OnlyEnforceIf(adjacent_vertical_alt)
                model.Add(r2_vars['y'] < r1_vars['y'] + r1_vars['height']).OnlyEnforceIf(adjacent_vertical_alt)

                # Horizontal adjacency: rooms share horizontal edge  
                model.Add(r1_vars['y'] + r1_vars['height'] == r2_vars['y']).OnlyEnforceIf(adjacent_horizontal)
                model.Add(r1_vars['x'] < r2_vars['x'] + r2_vars['width']).OnlyEnforceIf(adjacent_horizontal)
                model.Add(r2_vars['x'] < r1_vars['x'] + r1_vars['width']).OnlyEnforceIf(adjacent_horizontal)

                # Alternative: r2 adjacent to r1
                adjacent_horizontal_alt = model.NewBoolVar(f'{room2}_{room1}_adj_horiz')
                model.Add(r2_vars['y'] + r2_vars['height'] == r1_vars['y']).OnlyEnforceIf(adjacent_horizontal_alt)
                model.Add(r1_vars['x'] < r2_vars['x'] + r2_vars['width']).OnlyEnforceIf(adjacent_horizontal_alt)
                model.Add(r2_vars['x'] < r1_vars['x'] + r1_vars['width']).OnlyEnforceIf(adjacent_horizontal_alt)

                # At least one adjacency type should be true for strongly connected rooms
                if weight > 0.7:
                    model.AddBoolOr([adjacent_vertical, adjacent_vertical_alt, 
                                   adjacent_horizontal, adjacent_horizontal_alt])

                # Add bonus for adjacency (for objective function)
                adjacency_bonus = model.NewIntVar(0, int(weight * 100), f'adj_bonus_{room1}_{room2}')
                adjacency_bonuses.append(adjacency_bonus)

                # Set bonus value based on adjacency
                is_adjacent = model.NewBoolVar(f'{room1}_{room2}_is_adjacent')
                model.AddBoolOr([adjacent_vertical, adjacent_vertical_alt, 
                               adjacent_horizontal, adjacent_horizontal_alt]).OnlyEnforceIf(is_adjacent)
                model.Add(adjacency_bonus == int(weight * 100)).OnlyEnforceIf(is_adjacent)
                model.Add(adjacency_bonus == 0).OnlyEnforceIf(is_adjacent.Not())

        return adjacency_bonuses

    def _add_semantic_constraints(self, model, room_vars, room_graph):
        """Add semantic placement constraints based on room types"""

        # Get room types
        room_types = {}
        for room_id in room_vars.keys():
            room_types[room_id] = room_graph.graph.nodes[room_id]['type']

        # Bathroom should be smaller and in corners/edges
        for room_id, room_type in room_types.items():
            if room_type == 'bathroom':
                # Encourage bathroom to be near edges
                edge_bonus = model.NewIntVar(0, 50, f'{room_id}_edge_bonus')

                # Bonus for being near boundary edges
                near_left = model.NewBoolVar(f'{room_id}_near_left')
                near_right = model.NewBoolVar(f'{room_id}_near_right')
                near_top = model.NewBoolVar(f'{room_id}_near_top')
                near_bottom = model.NewBoolVar(f'{room_id}_near_bottom')

                model.Add(room_vars[room_id]['x'] <= 2).OnlyEnforceIf(near_left)
                model.Add(room_vars[room_id]['x'] + room_vars[room_id]['width'] >= int(self.width/self.grid_resolution) - 2).OnlyEnforceIf(near_right)
                model.Add(room_vars[room_id]['y'] <= 2).OnlyEnforceIf(near_bottom)
                model.Add(room_vars[room_id]['y'] + room_vars[room_id]['height'] >= int(self.height/self.grid_resolution) - 2).OnlyEnforceIf(near_top)

                # Set bonus for edge placement
                near_edge = model.NewBoolVar(f'{room_id}_near_edge')
                model.AddBoolOr([near_left, near_right, near_top, near_bottom]).OnlyEnforceIf(near_edge)
                model.Add(edge_bonus == 25).OnlyEnforceIf(near_edge)
                model.Add(edge_bonus == 0).OnlyEnforceIf(near_edge.Not())

        # Living room should be larger and central
        for room_id, room_type in room_types.items():
            if room_type == 'living':
                # Encourage larger size within constraints
                # This is handled by the area constraints already
                pass

        # Kitchen should be near living room (handled by adjacency constraints)

    def _add_optimization_objective(self, model, room_vars, room_graph):
        """Add optimization objective to maximize layout quality"""

        # Get adjacency bonuses
        adjacency_bonuses = self._add_adjacency_constraints_improved(model, room_vars, room_graph)

        # Area utilization bonus (minimize wasted space)
        total_area = model.NewIntVar(0, int(self.boundary.total_area / (self.grid_resolution ** 2)), 'total_area')
        area_terms = []
        for room_id in room_vars.keys():
            area_var = model.NewIntVar(0, 10000, f'{room_id}_area_calc')
            model.AddMultiplicationEquality(area_var, [room_vars[room_id]['width'], room_vars[room_id]['height']])
            area_terms.append(area_var)
        model.Add(total_area == sum(area_terms))

        # Compactness bonus (prefer square-ish rooms)
        compactness_bonuses = []
        for room_id in room_vars.keys():
            aspect_ratio_bonus = model.NewIntVar(0, 50, f'{room_id}_aspect_bonus')

            # Prefer width and height to be similar (aspect ratio close to 1)
            width_height_diff = model.NewIntVar(-100, 100, f'{room_id}_wh_diff')
            model.Add(width_height_diff == room_vars[room_id]['width'] - room_vars[room_id]['height'])

            # Bonus for small differences
            small_diff = model.NewBoolVar(f'{room_id}_small_diff')
            model.Add(width_height_diff <= 2).OnlyEnforceIf(small_diff)
            model.Add(width_height_diff >= -2).OnlyEnforceIf(small_diff)
            model.Add(aspect_ratio_bonus == 30).OnlyEnforceIf(small_diff)
            model.Add(aspect_ratio_bonus == 0).OnlyEnforceIf(small_diff.Not())

            compactness_bonuses.append(aspect_ratio_bonus)

        # Maximize total objective
        objective_terms = adjacency_bonuses + compactness_bonuses + [total_area]
        total_objective = model.NewIntVar(0, 100000, 'total_objective')
        model.Add(total_objective == sum(objective_terms))

        model.Maximize(total_objective)

    def _extract_solution(self, solver, room_vars) -> Dict[str, Tuple[float, float, float, float]]:
        """Extract solution and convert back to real coordinates"""

        solution = {}
        for room_id, vars_dict in room_vars.items():
            # Get grid coordinates
            grid_x = solver.Value(vars_dict['x'])
            grid_y = solver.Value(vars_dict['y']) 
            grid_width = solver.Value(vars_dict['width'])
            grid_height = solver.Value(vars_dict['height'])

            # Convert to real coordinates
            real_x = self.min_x + grid_x * self.grid_resolution
            real_y = self.min_y + grid_y * self.grid_resolution
            real_width = grid_width * self.grid_resolution
            real_height = grid_height * self.grid_resolution

            solution[room_id] = (real_x, real_y, real_width, real_height)

        return solution

    def _fallback_placement(self, room_graph: RoomGraph) -> Dict[str, Tuple[float, float, float, float]]:
        """Fallback simple placement when constraint solving fails"""

        print("Using fallback placement algorithm")

        rooms = list(room_graph.graph.nodes())
        placement = {}

        # Simple grid-based placement
        cols = int(np.sqrt(len(rooms))) + 1
        cell_width = self.width / cols
        cell_height = self.height / cols

        for i, room_id in enumerate(rooms):
            row = i // cols
            col = i % cols

            room_data = room_graph.graph.nodes[room_id]

            # Calculate room size to fit in cell
            max_cell_area = cell_width * cell_height * 0.8  # Leave some margin
            target_area = min(room_data['max_area'], max_cell_area)
            target_area = max(room_data['min_area'], target_area)

            # Make room roughly square
            side_length = np.sqrt(target_area)
            room_width = min(side_length, cell_width * 0.9)
            room_height = target_area / room_width

            # Position in cell
            x = self.min_x + col * cell_width + (cell_width - room_width) / 2
            y = self.min_y + row * cell_height + (cell_height - room_height) / 2

            placement[room_id] = (x, y, room_width, room_height)

        return placement
