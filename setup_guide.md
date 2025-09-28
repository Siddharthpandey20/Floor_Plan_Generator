
# Requirements and Installation Guide

## requirements.txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
networkx>=2.6.0
matplotlib>=3.5.0
opencv-python>=4.5.0
scipy>=1.7.0
ortools>=9.0.0
Pillow>=8.3.0
tqdm>=4.62.0
scikit-learn>=1.0.0
pandas>=1.3.0
seaborn>=0.11.0

## Development dependencies (optional)
jupyter>=1.0.0
pytest>=6.2.0
black>=21.0.0
flake8>=3.9.0

## Installation Instructions

### 1. Basic Installation
```bash
# Create virtual environment
python -m venv floor_plan_env
source floor_plan_env/bin/activate  # On Windows: floor_plan_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. GPU Support (Optional)
```bash
# For CUDA support (if you have compatible GPU)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. Development Setup
```bash
# Install development dependencies
pip install jupyter pytest black flake8

# Install in development mode
pip install -e .
```

## Project Structure

```
hybrid_floor_plan_generator/
├── floor_plan_implementation.py    # Core classes and algorithms
├── hybrid_floor_plan_generator.py  # Main pipeline
├── floor_plan_demo.py             # Usage examples
├── training_optimization_guide.py # Training procedures
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── data/                          # Training data
│   ├── floor_plans/              # Floor plan datasets
│   ├── boundaries/               # Building boundaries
│   └── adjacency_matrices/       # Room relationships
├── models/                        # Trained models
│   ├── consformer_model.pth      # ConsFormer weights
│   └── unet_wall_model.pth       # U-Net weights
├── outputs/                       # Generated floor plans
├── tests/                         # Unit tests
└── notebooks/                     # Jupyter notebooks
    ├── training_consformer.ipynb
    ├── data_analysis.ipynb
    └── visualization_examples.ipynb
```

## Quick Start Guide

### 1. Basic Usage
```python
from hybrid_floor_plan_generator import HybridFloorPlanGenerator

# Initialize generator
generator = HybridFloorPlanGenerator(device='cpu')

# Define simple rectangular building
boundary = [(0,0), (10,0), (10,8), (0,8)]

# Define rooms
rooms = [
    {
        'room_id': 'living_room',
        'room_type': 'living', 
        'min_area': 20.0,
        'max_area': 35.0
    },
    {
        'room_id': 'kitchen',
        'room_type': 'kitchen',
        'min_area': 12.0, 
        'max_area': 20.0
    }
]

# Define adjacencies
adjacencies = [('living_room', 'kitchen', 1.0)]

# Generate floor plan
room_graph, building_boundary = generator.load_constraints(boundary, rooms, adjacencies)
initial_layout = generator.geometric_constraint_propagation(room_graph, building_boundary)
model = generator.load_trained_consformer()
final_layout = generator.iterative_refinement_loop(room_graph, initial_layout, model)
walls = generator.wall_generator(final_layout, building_boundary)
generator.visualize_layout(final_layout, building_boundary, walls)
```

### 2. Advanced Configuration
```python
# Custom ConsFormer configuration
consformer = ConsFormerEncoder(
    d_model=512,      # Larger model
    nhead=16,         # More attention heads
    num_layers=12     # Deeper network
)

# Custom geometric solver
class CustomConstraintSolver(GeometricConstraintSolver):
    def __init__(self, boundary):
        super().__init__(boundary)
        self.grid_resolution = 0.1  # Higher resolution
        self.max_iterations = 1000   # More iterations

# Custom loss function
class CustomFloorPlanLoss(FloorPlanLoss):
    def __init__(self):
        super().__init__(
            alpha=2.0,    # Higher position weight
            beta=1.5,     # Higher size weight
            gamma=1.0,    # Overlap penalty
            delta=0.8     # Adjacency weight
        )
```

## Performance Optimization

### 1. GPU Acceleration
```python
# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = HybridFloorPlanGenerator(device=device)
```

### 2. Batch Processing
```python
# Process multiple floor plans in batch
batch_boundaries = [boundary1, boundary2, boundary3]
batch_room_specs = [rooms1, rooms2, rooms3]

results = []
for boundary, rooms in zip(batch_boundaries, batch_room_specs):
    layout = generator.generate_floor_plan(boundary, rooms)
    results.append(layout)
```

### 3. Memory Optimization
```python
# For large-scale generation
import gc
import torch

def generate_with_memory_cleanup(generator, boundary, rooms):
    layout = generator.generate_floor_plan(boundary, rooms)

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return layout
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size or model size
   - Use gradient checkpointing
   - Switch to CPU processing

2. **Constraint satisfaction fails** 
   - Check room area constraints are feasible
   - Verify building boundary is large enough
   - Adjust grid resolution in geometric solver

3. **Poor layout quality**
   - Increase number of refinement iterations
   - Adjust loss function weights
   - Use pre-trained models

4. **Slow generation**
   - Use GPU acceleration
   - Reduce model complexity
   - Optimize constraint solver parameters

### Performance Tips

1. **Pre-compute adjacency matrices** for repeated room configurations
2. **Cache building boundaries** for similar layouts
3. **Use mixed precision training** to reduce memory usage
4. **Implement early stopping** in refinement loop
5. **Parallelize constraint solving** for multiple room placements

## Model Training

### 1. Data Preparation
```python
# Prepare training dataset
from torch.utils.data import DataLoader

dataset = FloorPlanDataset('data/floor_plans/')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train ConsFormer
model = ConsFormerEncoder()
train_consformer(model, train_loader, val_loader, num_epochs=100)
```

### 2. Model Evaluation
```python
# Evaluate model performance
metrics = evaluate_model(model, test_loader)
print(f"Position Error: {metrics['position_error']:.3f}")
print(f"Size Error: {metrics['size_error']:.3f}") 
print(f"Adjacency Score: {metrics['adjacency_score']:.3f}")
```

## Extensions and Customization

### 1. Custom Room Types
```python
# Add new room types
ROOM_TYPES = {
    'living': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3,
    'office': 4, 'garage': 5, 'balcony': 6, 'storage': 7,
    'laundry': 8, 'pantry': 9  # New room types
}
```

### 2. Additional Constraints
```python
# Add custom constraints
class AdvancedConstraints:
    def __init__(self):
        self.privacy_constraints = {}  # Bedroom away from living areas
        self.lighting_constraints = {}  # Windows for natural light
        self.accessibility_constraints = {}  # ADA compliance
```

### 3. Multi-story Buildings
```python
# Extend for multi-story generation
class MultiStoryGenerator(HybridFloorPlanGenerator):
    def generate_building(self, floors, vertical_constraints):
        layouts = []
        for floor_spec in floors:
            layout = self.generate_floor_plan(floor_spec)
            layouts.append(layout)
        return self.align_vertical_elements(layouts, vertical_constraints)
```
