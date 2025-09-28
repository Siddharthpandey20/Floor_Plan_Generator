
# Training and Optimization Guide for Hybrid Floor Plan Generation

## Training the ConsFormer Transformer

### 1. Dataset Preparation

def prepare_training_data():
    """
    Prepare training dataset for ConsFormer model

    Dataset should include:
    - Floor plan layouts with room positions and sizes
    - Room adjacency graphs  
    - Building boundaries
    - Quality scores for layouts
    """

    # Example data structure
    training_sample = {
        'room_features': torch.tensor([  # [num_rooms, 4] - x, y, type, area
            [2.0, 3.0, 0, 25.0],  # living room
            [8.0, 3.0, 1, 15.0],  # kitchen  
            [2.0, 7.0, 2, 20.0],  # bedroom
        ]),
        'adjacency_matrix': torch.tensor([  # [num_rooms, num_rooms]
            [0, 1, 0],  # living adjacent to kitchen
            [1, 0, 1],  # kitchen adjacent to living and bedroom
            [0, 1, 0]   # bedroom adjacent to kitchen
        ]),
        'target_positions': torch.tensor([  # Ground truth positions
            [2.5, 3.5], [8.5, 3.5], [2.5, 7.5]
        ]),
        'target_sizes': torch.tensor([  # Ground truth sizes
            [4.0, 5.0], [3.0, 4.0], [4.0, 4.0]
        ]),
        'quality_score': 0.85  # Layout quality (0-1)
    }

    return training_sample

### 2. Loss Functions

import torch
import torch.nn as nn

class FloorPlanLoss(nn.Module):
    """Multi-objective loss for floor plan generation"""

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, delta=0.3):
        super().__init__()
        self.alpha = alpha    # Position loss weight
        self.beta = beta      # Size loss weight  
        self.gamma = gamma    # Overlap penalty weight
        self.delta = delta    # Adjacency loss weight

    def forward(self, predictions, targets, adjacency_matrix):
        """
        Args:
            predictions: Dict with 'position_updates', 'size_updates', 'adjacency_updates'
            targets: Dict with 'target_positions', 'target_sizes'
            adjacency_matrix: Ground truth adjacency
        """

        # Position loss (L2)
        position_loss = nn.MSELoss()(
            predictions['position_updates'], 
            targets['target_positions']
        )

        # Size loss (L2) 
        size_loss = nn.MSELoss()(
            predictions['size_updates'],
            targets['target_sizes']
        )

        # Overlap penalty (differentiable)
        overlap_penalty = self.compute_overlap_penalty(
            predictions['position_updates'], 
            predictions['size_updates']
        )

        # Adjacency preservation loss
        adjacency_loss = self.compute_adjacency_loss(
            predictions['position_updates'],
            predictions['size_updates'], 
            adjacency_matrix
        )

        total_loss = (self.alpha * position_loss + 
                     self.beta * size_loss +
                     self.gamma * overlap_penalty +
                     self.delta * adjacency_loss)

        return {
            'total_loss': total_loss,
            'position_loss': position_loss,
            'size_loss': size_loss, 
            'overlap_penalty': overlap_penalty,
            'adjacency_loss': adjacency_loss
        }

    def compute_overlap_penalty(self, positions, sizes):
        """Differentiable overlap penalty"""
        batch_size, num_rooms, _ = positions.shape
        penalty = 0.0

        for i in range(num_rooms):
            for j in range(i+1, num_rooms):
                # Room i bounds
                x1, y1 = positions[:, i, 0], positions[:, i, 1]  
                w1, h1 = sizes[:, i, 0], sizes[:, i, 1]

                # Room j bounds
                x2, y2 = positions[:, j, 0], positions[:, j, 1]
                w2, h2 = sizes[:, j, 0], sizes[:, j, 1]

                # Compute overlap area (differentiable)
                overlap_w = torch.clamp(torch.min(x1 + w1, x2 + w2) - torch.max(x1, x2), min=0)
                overlap_h = torch.clamp(torch.min(y1 + h1, y2 + h2) - torch.max(y1, y2), min=0)
                overlap_area = overlap_w * overlap_h

                penalty += overlap_area.mean()

        return penalty

    def compute_adjacency_loss(self, positions, sizes, adjacency_matrix):
        """Encourage adjacent rooms to be close"""
        batch_size, num_rooms, _ = positions.shape
        adjacency_loss = 0.0

        # Get room centers
        centers = positions + sizes / 2

        for i in range(num_rooms):
            for j in range(i+1, num_rooms):
                if adjacency_matrix[0, i, j] > 0:  # Should be adjacent
                    # Distance between room centers
                    dist = torch.norm(centers[:, i] - centers[:, j], dim=1)
                    adjacency_loss += dist.mean()

        return adjacency_loss

### 3. Training Loop

def train_consformer(model, train_loader, val_loader, num_epochs=100):
    """Training loop for ConsFormer model"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = FloorPlanLoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()

            room_features = batch['room_features']
            adjacency_matrix = batch['adjacency_matrix']
            targets = {
                'target_positions': batch['target_positions'],
                'target_sizes': batch['target_sizes']
            }

            # Forward pass
            predictions = model(room_features, adjacency_matrix)

            # Compute loss
            losses = criterion(predictions, targets, adjacency_matrix)
            loss = losses['total_loss']

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                predictions = model(batch['room_features'], batch['adjacency_matrix'])
                targets = {
                    'target_positions': batch['target_positions'],
                    'target_sizes': batch['target_sizes']
                }
                losses = criterion(predictions, targets, batch['adjacency_matrix'])
                val_losses.append(losses['total_loss'].item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_consformer_model.pth')

        scheduler.step()

## Training the U-Net Wall Generator

def train_unet_wall_generator():
    """Training setup for U-Net wall generation model"""

    # Dataset should include:
    # - Room occupancy maps as input
    # - Wall/door masks as targets

    class WallGenerationDataset(torch.utils.data.Dataset):
        def __init__(self, occupancy_maps, wall_masks):
            self.occupancy_maps = occupancy_maps  # [N, 3, H, W] RGB room layouts
            self.wall_masks = wall_masks          # [N, 1, H, W] Binary wall masks

        def __len__(self):
            return len(self.occupancy_maps)

        def __getitem__(self, idx):
            return {
                'occupancy_map': torch.tensor(self.occupancy_maps[idx], dtype=torch.float32),
                'wall_mask': torch.tensor(self.wall_masks[idx], dtype=torch.float32)
            }

    model = UNetWallGenerator(in_channels=3, out_channels=1)
    criterion = nn.BCELoss()  # Binary cross-entropy for wall/no-wall
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop similar to ConsFormer but with image data

## Data Augmentation

def augment_floor_plan_data(room_features, adjacency_matrix):
    """Data augmentation for floor plan training"""

    augmentations = []

    # 1. Random rotation (90, 180, 270 degrees)
    for angle in [0, 90, 180, 270]:
        rotated_features = rotate_room_features(room_features, angle)
        augmentations.append((rotated_features, adjacency_matrix))

    # 2. Random scaling (0.8x to 1.2x)
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        scaled_features = scale_room_features(room_features, scale)
        augmentations.append((scaled_features, adjacency_matrix))

    # 3. Random room type permutation (while preserving constraints)
    permuted_features, permuted_adj = permute_room_types(room_features, adjacency_matrix)
    augmentations.append((permuted_features, permuted_adj))

    return augmentations

## Optimization Techniques

class ConstraintOptimizer:
    """Advanced constraint optimization using OR-Tools"""

    def __init__(self):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

    def optimize_room_placement(self, room_specs, boundary, adjacency_constraints):
        """Use CP-SAT for optimal room placement"""

        # Decision variables for room positions and sizes
        room_vars = {}

        for room in room_specs:
            room_id = room['room_id']
            room_vars[room_id] = {
                'x': self.model.NewIntVar(0, 100, f'{room_id}_x'),
                'y': self.model.NewIntVar(0, 100, f'{room_id}_y'), 
                'width': self.model.NewIntVar(2, 20, f'{room_id}_width'),
                'height': self.model.NewIntVar(2, 20, f'{room_id}_height')
            }

            # Area constraints
            area_var = self.model.NewIntVar(
                int(room['min_area']), int(room['max_area']), f'{room_id}_area'
            )
            self.model.AddMultiplicationEquality(
                area_var, [room_vars[room_id]['width'], room_vars[room_id]['height']]
            )

        # Non-overlap constraints
        self.add_non_overlap_constraints(room_vars)

        # Boundary constraints  
        self.add_boundary_constraints(room_vars, boundary)

        # Adjacency constraints
        self.add_adjacency_constraints(room_vars, adjacency_constraints)

        # Solve
        status = self.solver.Solve(self.model)

        if status == cp_model.OPTIMAL:
            solution = {}
            for room_id, vars_dict in room_vars.items():
                solution[room_id] = {
                    'x': self.solver.Value(vars_dict['x']),
                    'y': self.solver.Value(vars_dict['y']),
                    'width': self.solver.Value(vars_dict['width']),
                    'height': self.solver.Value(vars_dict['height'])
                }
            return solution
        else:
            return None

    def add_non_overlap_constraints(self, room_vars):
        """Add non-overlapping constraints between rooms"""

        rooms = list(room_vars.keys())

        for i, room1 in enumerate(rooms):
            for room2 in rooms[i+1:]:
                # Room1 is to the left of room2 OR room2 is to the left of room1 OR
                # Room1 is above room2 OR room2 is above room1

                r1_vars = room_vars[room1]
                r2_vars = room_vars[room2]

                # Create boolean variables for relative positions
                left_of = self.model.NewBoolVar(f'{room1}_left_of_{room2}')
                right_of = self.model.NewBoolVar(f'{room1}_right_of_{room2}')
                above = self.model.NewBoolVar(f'{room1}_above_{room2}')
                below = self.model.NewBoolVar(f'{room1}_below_{room2}')

                # Exactly one must be true
                self.model.AddExactlyOne([left_of, right_of, above, below])

                # Add corresponding constraints
                self.model.Add(r1_vars['x'] + r1_vars['width'] <= r2_vars['x']).OnlyEnforceIf(left_of)
                self.model.Add(r2_vars['x'] + r2_vars['width'] <= r1_vars['x']).OnlyEnforceIf(right_of) 
                self.model.Add(r1_vars['y'] + r1_vars['height'] <= r2_vars['y']).OnlyEnforceIf(above)
                self.model.Add(r2_vars['y'] + r2_vars['height'] <= r1_vars['y']).OnlyEnforceIf(below)

print("Training and optimization guide created successfully!")
