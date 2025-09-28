
# Hybrid Floor Plan Generation: Complete Implementation Framework

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from ortools.sat.python import cp_model
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class RoomSpec:
    """Room specification with constraints"""
    room_id: str
    room_type: str  # 'living', 'kitchen', 'bedroom', etc.
    min_area: float
    max_area: float
    min_width: float = 2.0
    min_height: float = 2.0
    preferred_adjacencies: List[str] = None

    def __post_init__(self):
        if self.preferred_adjacencies is None:
            self.preferred_adjacencies = []

@dataclass 
class BuildingBoundary:
    """Building boundary specification"""
    vertices: List[Tuple[float, float]]
    total_area: float

class RoomGraph:
    """Spatial room graph with adjacency relationships"""

    def __init__(self):
        self.graph = nx.Graph()
        self.room_positions = {}
        self.room_sizes = {}
        self.adjacency_weights = {}

    def add_room(self, room_spec: RoomSpec, position: Tuple[float, float] = None):
        """Add room node to graph"""
        self.graph.add_node(room_spec.room_id, 
                          type=room_spec.room_type,
                          min_area=room_spec.min_area,
                          max_area=room_spec.max_area)
        if position:
            self.room_positions[room_spec.room_id] = position

    def add_adjacency_constraint(self, room1: str, room2: str, weight: float = 1.0):
        """Add adjacency constraint between rooms"""
        self.graph.add_edge(room1, room2, weight=weight)
        self.adjacency_weights[(room1, room2)] = weight

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation"""
        return nx.adjacency_matrix(self.graph).toarray()

class GeometricConstraintSolver:
    """Geometric constraint propagation using corner-filling algorithm"""

    def __init__(self, boundary: BuildingBoundary):
        self.boundary = boundary
        self.grid_resolution = 0.5  # meters
        self.available_corners = []

    def create_occupancy_grid(self) -> np.ndarray:
        """Create occupancy grid from boundary"""
        # Convert boundary to grid coordinates
        vertices = np.array(self.boundary.vertices)
        min_x, min_y = vertices.min(axis=0)
        max_x, max_y = vertices.max(axis=0)

        grid_width = int((max_x - min_x) / self.grid_resolution) + 1
        grid_height = int((max_y - min_y) / self.grid_resolution) + 1

        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        # Fill boundary polygon
        boundary_points = []
        for x, y in vertices:
            grid_x = int((x - min_x) / self.grid_resolution)
            grid_y = int((y - min_y) / self.grid_resolution)
            boundary_points.append([grid_x, grid_y])

        boundary_points = np.array(boundary_points, dtype=np.int32)
        cv2.fillPoly(grid, [boundary_points], 1)

        return grid

    def find_corners(self, occupancy_grid: np.ndarray) -> List[Tuple[int, int]]:
        """Find available corners in occupancy grid"""
        corners = []
        h, w = occupancy_grid.shape

        # Detect corners using convolution
        corner_kernel = np.array([
            [1, 1, 0],
            [1, 1, 0], 
            [0, 0, 0]
        ])

        for i in range(h-2):
            for j in range(w-2):
                if occupancy_grid[i:i+3, j:j+3].sum() > 0:
                    region = occupancy_grid[i:i+3, j:j+3]
                    if np.array_equal(region * corner_kernel, corner_kernel):
                        corners.append((i, j))

        return corners

    def place_room_at_corner(self, room_spec: RoomSpec, corner: Tuple[int, int], 
                           occupancy_grid: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        """Try to place room at corner position"""
        y, x = corner
        h, w = occupancy_grid.shape

        # Calculate room dimensions in grid units
        min_room_width = max(2, int(room_spec.min_width / self.grid_resolution))
        min_room_height = max(2, int(np.sqrt(room_spec.min_area) / self.grid_resolution))

        # Try different room sizes
        for room_h in range(min_room_height, min(h-y, 20)):
            for room_w in range(min_room_width, min(w-x, 20)):
                # Check if area constraints satisfied
                room_area = room_h * room_w * (self.grid_resolution ** 2)
                if not (room_spec.min_area <= room_area <= room_spec.max_area):
                    continue

                # Check if placement is valid (no overlap)
                room_region = occupancy_grid[y:y+room_h, x:x+room_w]
                if room_region.shape == (room_h, room_w) and room_region.sum() == room_h * room_w:
                    return True, (x, y, room_w, room_h)

        return False, None

    def solve_constraints(self, room_graph: RoomGraph) -> Dict[str, Tuple[float, float, float, float]]:
        """Solve geometric constraints using corner-filling"""
        occupancy_grid = self.create_occupancy_grid()
        room_placements = {}

        # Sort rooms by area (largest first)
        rooms = list(room_graph.graph.nodes())
        rooms.sort(key=lambda r: room_graph.graph.nodes[r]['max_area'], reverse=True)

        for room_id in rooms:
            room_data = room_graph.graph.nodes[room_id]
            room_spec = RoomSpec(
                room_id=room_id,
                room_type=room_data['type'],
                min_area=room_data['min_area'],
                max_area=room_data['max_area']
            )

            # Find available corners
            corners = self.find_corners(occupancy_grid)
            placed = False

            for corner in corners:
                success, placement = self.place_room_at_corner(room_spec, corner, occupancy_grid)
                if success:
                    x, y, w, h = placement
                    # Mark area as occupied
                    occupancy_grid[y:y+h, x:x+w] = 0

                    # Convert back to real coordinates
                    vertices = np.array(self.boundary.vertices)
                    min_x, min_y = vertices.min(axis=0)

                    real_x = min_x + x * self.grid_resolution
                    real_y = min_y + y * self.grid_resolution
                    real_w = w * self.grid_resolution
                    real_h = h * self.grid_resolution

                    room_placements[room_id] = (real_x, real_y, real_w, real_h)
                    placed = True
                    break

            if not placed:
                print(f"Warning: Could not place room {room_id}")

        return room_placements

# ConsFormer Transformer Architecture
class ConsFormerEncoder(nn.Module):
    """ConsFormer encoder with graph attention for spatial layouts"""

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model

        # Positional and categorical embeddings
        self.position_embedding = nn.Linear(2, d_model)  # x, y coordinates
        self.type_embedding = nn.Embedding(10, d_model)  # room types
        self.area_embedding = nn.Linear(1, d_model)      # room area

        # Transformer layers with cross attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Graph attention mechanism
        self.graph_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Output projections
        self.position_head = nn.Linear(d_model, 2)  # x, y updates
        self.size_head = nn.Linear(d_model, 2)      # width, height updates
        self.adjacency_head = nn.Linear(d_model, 1) # adjacency weights

    def create_adjacency_mask(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Create attention mask from adjacency matrix"""
        # Convert adjacency matrix to attention mask
        batch_size, num_rooms, _ = adjacency_matrix.shape
        mask = adjacency_matrix.clone()
        mask[mask == 0] = float('-inf')
        mask[mask == 1] = 0.0
        # Expand mask for multi-head attention: (batch_size * num_heads, seq_len, seq_len)
        mask = mask.repeat(self.graph_attention.num_heads, 1, 1)
        return mask

    def forward(self, room_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            room_features: [batch_size, num_rooms, feature_dim] 
                         where features = [x, y, type_id, area]
            adjacency_matrix: [batch_size, num_rooms, num_rooms]
        """
        batch_size, num_rooms, _ = room_features.shape

        # Extract features
        positions = room_features[:, :, :2]  # x, y
        types = room_features[:, :, 2].long()  # room type ids
        areas = room_features[:, :, 3:4]     # areas

        # Create embeddings
        pos_emb = self.position_embedding(positions)
        type_emb = self.type_embedding(types)
        area_emb = self.area_embedding(areas)

        # Combine embeddings
        embeddings = pos_emb + type_emb + area_emb

        # Apply transformer
        transformed = self.transformer(embeddings)

        # Apply graph attention with adjacency mask
        attn_mask = self.create_adjacency_mask(adjacency_matrix)
        graph_attended, _ = self.graph_attention(
            transformed, transformed, transformed, attn_mask=attn_mask
        )

        # Final feature combination
        final_features = transformed + graph_attended

        # Generate outputs
        position_updates = self.position_head(final_features)
        size_updates = self.size_head(final_features)
        adjacency_updates = self.adjacency_head(final_features).squeeze(-1)

        return {
            'position_updates': position_updates,
            'size_updates': size_updates, 
            'adjacency_updates': adjacency_updates,
            'features': final_features
        }

# U-Net for Wall Generation
class UNetWallGenerator(nn.Module):
    """U-Net architecture for generating floor plan walls"""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)

        # Decoder
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Convolutional block with batch norm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Upconvolutional block for decoder"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input occupancy map [batch_size, in_channels, height, width]
        Returns:
            Wall mask [batch_size, out_channels, height, width]
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))

        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))

        # Decoder with skip connections
        d4 = self.dec4(b) + e4
        d3 = self.dec3(d4) + e3  
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1

        # Output
        output = self.sigmoid(self.output(d1))

        return output

print("Core implementation classes defined successfully")
