import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from pyvista import examples
import time

class MeshQualityAnalyzerFixed:
    """Fixed mesh quality analyzer with proper cell type detection."""
    
    def __init__(self, mesh_file: str):
        """Initialize the analyzer with a mesh file."""
        self.mesh = pv.read(mesh_file)
        self.results = {}
        
        # First, properly detect cell types
        self.cell_types = self._get_cell_types_properly()
        print(f"Detected cell types: {self.cell_types}")
        
        # Define metric configurations for TRIANGULAR mesh
        self.metric_configs = {
            'element_quality': ('shape', (1.0, 0.0), ['TRIANGLE']),
            'aspect_ratio_tri': ('aspect_ratio', (1.0, 20.0), ['TRIANGLE']),
            'parallel_deviation': ('skew', (1.0, 170.0), ['TRIANGLE']),
            'max_angle_tri': ('max_angle', (60.0, 179.0), ['TRIANGLE']),
            'skewness_tri': ('skew', (0.0, 1.0), ['TRIANGLE']),
            'orthogonal_quality': ('shear', (1.0, 0.0), ['TRIANGLE'])
        }
        
    def _get_cell_types_properly(self):
        """Properly identify cell types present in the mesh."""
        cell_types = set()
        
        # Method 1: Check cell types array if it exists
        if hasattr(self.mesh, 'cell_types'):
            if self.mesh.cell_types is not None:
                unique_types = np.unique(self.mesh.cell_types)
                for cell_type in unique_types:
                    try:
                        type_name = pv.CellType(cell_type).name
                        cell_types.add(type_name)
                    except:
                        cell_types.add(f'UNKNOWN_{cell_type}')
        
        # Method 2: Check individual cells if no cell_types array
        if not cell_types and self.mesh.n_cells > 0:
            # Sample some cells to determine type
            sample_cells = min(10, self.mesh.n_cells)
            for i in range(sample_cells):
                cell = self.mesh.get_cell(i)
                if cell.type == 5:  # VTK_TRIANGLE
                    cell_types.add('TRIANGLE')
                elif cell.type == 9:  # VTK_QUAD
                    cell_types.add('QUAD')
                elif cell.type == 12:  # VTK_HEXAHEDRON
                    cell_types.add('HEXAHEDRON')
        
        # Method 3: If still empty, assume triangles based on your diagnostic output
        if not cell_types:
            print("Warning: Could not detect cell types, assuming TRIANGLE based on diagnostic")
            cell_types.add('TRIANGLE')
            
        return list(cell_types)
    
    def calculate_all_metrics(self):
        """Calculate all quality metrics for the mesh."""
        print(f"Mesh contains: {self.cell_types}")
        print(f"Total cells: {self.mesh.n_cells}")
        
        for metric_name, (pv_metric, acceptable_range, applicable_types) in self.metric_configs.items():
            # Check if mesh contains applicable cell types
            applicable = any(ct in self.cell_types for ct in applicable_types)
            if not applicable:
                print(f"Skipping {metric_name}: No applicable cell types found")
                continue
                
            print(f"\nCalculating: {metric_name} ({pv_metric})...")
            
            try:
                # Calculate the metric
                mesh_with_quality = self.mesh.cell_quality(pv_metric)
                
                # Check if the metric was actually calculated
                if pv_metric not in mesh_with_quality.cell_data:
                    print(f"  Warning: Metric '{pv_metric}' not found in cell data")
                    print(f"  Available cell data: {list(mesh_with_quality.cell_data.keys())}")
                    continue
                    
                quality_values = mesh_with_quality.cell_data[pv_metric]
                
                # Calculate statistics
                stats = {
                    'values': quality_values,
                    'mean': float(np.mean(quality_values)),
                    'min': float(np.min(quality_values)),
                    'max': float(np.max(quality_values)),
                    'std': float(np.std(quality_values)),
                    'acceptable_range': acceptable_range,
                    'applicable_cells': int(len(quality_values))
                }
                
                # Identify problematic cells
                good_val, bad_val = acceptable_range
                if good_val < bad_val:
                    # Higher values are worse (e.g., aspect ratio)
                    bad_cells_mask = quality_values > bad_val
                    poor_cells_mask = (quality_values > good_val) & (quality_values <= bad_val)
                else:
                    # Lower values are worse (e.g., element quality)
                    bad_cells_mask = quality_values < bad_val
                    poor_cells_mask = (quality_values >= bad_val) & (quality_values < good_val)
                
                stats['bad_cells_count'] = int(np.sum(bad_cells_mask))
                stats['poor_cells_count'] = int(np.sum(poor_cells_mask))
                stats['bad_cells_indices'] = np.where(bad_cells_mask)[0].tolist()
                stats['poor_cells_indices'] = np.where(poor_cells_mask)[0].tolist()
                
                self.results[metric_name] = stats
                
                print(f"  Mean: {stats['mean']:.3f}, Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")
                print(f"  Acceptable range: {acceptable_range}")
                print(f"  Bad cells: {stats['bad_cells_count']} ({stats['bad_cells_count']/stats['applicable_cells']*100:.1f}%), "
                      f"Poor cells: {stats['poor_cells_count']} ({stats['poor_cells_count']/stats['applicable_cells']*100:.1f}%)")
                
            except Exception as e:
                print(f"  Error calculating {metric_name}: {str(e)[:200]}")
                import traceback
                traceback.print_exc()
                self.results[metric_name] = {'error': str(e)}
        
        return self.results
    
    def visualize_metric(self, metric_name: str, save_plot: bool = False):
        """Visualize a specific metric on the mesh."""
        if metric_name not in self.results:
            print(f"Metric '{metric_name}' not calculated yet.")
            return
        
        if 'error' in self.results[metric_name]:
            print(f"Cannot visualize {metric_name}: {self.results[metric_name]['error']}")
            return
        
        stats = self.results[metric_name]
        pv_metric = self.metric_configs[metric_name][0]
        
        # Create mesh with quality data
        mesh_with_quality = self.mesh.cell_quality(pv_metric)
        
        # Get acceptable range
        good_val, bad_val = stats['acceptable_range']
        
        # Create a custom colormap
        if good_val < bad_val:
            # For metrics where higher is worse
            clim = (good_val, bad_val)
            cmap = 'RdYlGn_r'  # Red (bad) to Green (good), reversed
        else:
            # For metrics where lower is worse
            clim = (bad_val, good_val)
            cmap = 'RdYlGn'  # Red (bad) to Green (good)
        
        # Create plot
        plotter = pv.Plotter()
        plotter.add_mesh(
            mesh_with_quality,
            scalars=pv_metric,
            clim=clim,
            cmap=cmap,
            show_edges=True,
            scalar_bar_args={'title': f'{metric_name.replace("_", " ").title()}'}
        )
        
        plotter.add_text(f"Acceptable range: {stats['acceptable_range']}", 
                        position='upper_left', font_size=10)
        plotter.add_text(f"Bad cells: {stats['bad_cells_count']} ({stats['bad_cells_count']/stats['applicable_cells']*100:.1f}%)", 
                        position='upper_left', font_size=10, vertical_offset=40)
        
        if save_plot:
            plotter.screenshot(f'{metric_name}_visualization.png', window_size=[800, 600])
        
        plotter.show()
    
    def create_summary_report(self):
        """Create a comprehensive summary report of all metrics."""
        if not self.results:
            print("No metrics calculated yet. Run calculate_all_metrics() first.")
            return
        
        print("\n" + "="*80)
        print("MESH QUALITY ANALYSIS SUMMARY")
        print("="*80)
        
        report_data = []
        
        for metric_name in self.metric_configs.keys():
            if metric_name not in self.results:
                report_data.append([metric_name, "NOT CALCULATED", "-", "-", "-"])
                continue
                
            if 'error' in self.results[metric_name]:
                report_data.append([metric_name, "ERROR", "-", "-", "-"])
                continue
            
            stats = self.results[metric_name]
            bad_percentage = (stats['bad_cells_count'] / stats['applicable_cells']) * 100 if stats['applicable_cells'] > 0 else 0
            
            quality_status = "PASS"
            if bad_percentage > 10:
                quality_status = "FAIL"
            elif bad_percentage > 5:
                quality_status = "WARNING"
            
            report_data.append([
                metric_name,
                f"{stats['mean']:.3f}",
                f"{stats['min']:.3f}-{stats['max']:.3f}",
                f"{stats['bad_cells_count']} ({bad_percentage:.1f}%)",
                quality_status
            ])
        
        # Print table
        headers = ["Metric", "Mean", "Range", "Bad Cells", "Status"]
        col_widths = [25, 10, 15, 20, 10]
        
        print("\n" + "-"*80)
        print(f"{headers[0]:<{col_widths[0]}} {headers[1]:<{col_widths[1]}} {headers[2]:<{col_widths[2]}} {headers[3]:<{col_widths[3]}} {headers[4]:<{col_widths[4]}}")
        print("-"*80)
        
        for row in report_data:
            print(f"{row[0]:<{col_widths[0]}} {row[1]:<{col_widths[1]}} {row[2]:<{col_widths[2]}} {row[3]:<{col_widths[3]}} {row[4]:<{col_widths[4]}}")
        
        # Create visual summary
        self._create_visual_summary()
    
    def _create_visual_summary(self):
        """Create a visual summary chart of all metrics."""
        if not self.results:
            return
            
        metrics_with_data = [m for m in self.metric_configs.keys() 
                            if m in self.results and 'error' not in self.results[m]]
        
        if not metrics_with_data:
            return
            
        n_metrics = len(metrics_with_data)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_metrics > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for idx, metric_name in enumerate(metrics_with_data):
            if idx >= len(axes):
                break
                
            stats = self.results[metric_name]
            ax = axes[idx]
            
            # Create histogram
            ax.hist(stats['values'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add acceptable range lines
            good_val, bad_val = stats['acceptable_range']
            ax.axvline(x=good_val, color='green', linestyle='--', label='Good threshold', linewidth=2)
            ax.axvline(x=bad_val, color='red', linestyle='--', label='Bad threshold', linewidth=2)
            
            # Format plot
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add text with statistics
            text_str = f"Mean: {stats['mean']:.3f}\nBad: {stats['bad_cells_count']}"
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide empty subplots
        for j in range(len(metrics_with_data), len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Mesh Quality Metrics Distribution', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('mesh_quality_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def export_results(self, filename: str = 'mesh_quality_results.json'):
        """Export all results to a JSON file."""
        import json
        
        if not self.results:
            print("No results to export. Run calculate_all_metrics() first.")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults exported to {filename}")

class InteractiveQualityViewer:
    def __init__(self, mesh_file):
        # 1. Load mesh and compute all metrics
        self.analyzer = MeshQualityAnalyzerFixed(mesh_file)
        self.mesh = self.analyzer.mesh
        self.results = self.analyzer.calculate_all_metrics()
        
        # 2. Store the PyVista metric name for each of your 6 metrics
        self.metric_list = ['element_quality', 'aspect_ratio_tri', 
                           'parallel_deviation', 'max_angle_tri', 
                           'skewness_tri', 'orthogonal_quality']
        self.current_metric_index = 0
        
        # 3. Prepare the mesh for visualization
        # Create a working copy and attach all quality data arrays
        self.viz_mesh = self.mesh.copy()
        for metric_name in self.metric_list:
            if metric_name in self.results and 'values' in self.results[metric_name]:
                pv_metric = self.analyzer.metric_configs[metric_name][0]
                self.viz_mesh.cell_data[f'{metric_name}_data'] = self.results[metric_name]['values']
        
        # 4. Create the plotter
        self.plotter = pv.Plotter()
        
        # 5. Add the mesh actor (initially colored by the first metric)
        self.mesh_actor = self.plotter.add_mesh(
            self.viz_mesh,
            scalars=self.get_current_scalar_name(),
            cmap='bwr',  # Blue-White-Red colormap (bad-average-good)
            show_edges=True,
            scalar_bar_args={'title': self.get_current_metric_display_name()}
        )
        
        # 6. Set up keyboard callbacks
        self.plotter.add_key_event('Right', self.next_metric)
        self.plotter.add_key_event('Left', self.previous_metric)
        self.plotter.add_key_event('c', self.toggle_colorbar)
        self.plotter.add_text(
            "Press 'Right' or 'Left' to cycle metrics | 'c' to toggle colorbar", 
            position='lower_left', font_size=8
        )
    
    def get_current_metric_display_name(self):
        """Get formatted name for current metric."""
        metric = self.metric_list[self.current_metric_index]
        return metric.replace('_', ' ').title()
    
    def get_current_scalar_name(self):
        """Get the cell data array name for current metric."""
        metric = self.metric_list[self.current_metric_index]
        return f'{metric}_data'
    
    def update_display(self):
        #Update the mesh colors and display info for current metric.
        # Get current metric data
        current_metric = self.metric_list[self.current_metric_index]
    
        if current_metric not in self.results or 'values' not in self.results[current_metric]:
            print(f"No data for {current_metric}")
            return
    
        # Update scalar data and color limits
        scalars = self.get_current_scalar_name()
    
        # Get your custom acceptable range for this metric
        config = self.analyzer.metric_configs[current_metric]
        acceptable_range = config[1]  # (good_value, bad_value) tuple
    
        # Determine color limits based on whether higher or lower is worse
        good_val, bad_val = acceptable_range
        if good_val < bad_val:
            # Higher values are worse (e.g., aspect ratio)
            clim = (good_val, bad_val)
        else:
            # Lower values are worse (e.g., element quality)
            clim = (bad_val, good_val)
    
        # --- CRITICAL FIXES START ---
        # 1. UPDATE THE MESH SCALARS DIRECTLY (instead of using deprecated update_scalars)
        # Assign the data array to the mesh's active scalars
        self.viz_mesh.set_active_scalars(scalars)
    
        # 2. UPDATE THE SCALAR BAR RANGE WITH THE CORRECT NAME
        # Get the actual scalar bar title/name that was created with add_mesh
        # The name is typically the title you set, not a generic 'scalar_bar'
        scalar_bar_title = self.get_current_metric_display_name()
        try:
            # Update the scalar bar range using the actual title
            self.plotter.update_scalar_bar_range(clim, name=scalar_bar_title)
        except (KeyError, ValueError) as e:
            # If that fails, update without a name (affects the active scalar bar)
            print(f"Note: Updating active scalar bar. {e}")
            self.plotter.update_scalar_bar_range(clim)
        # --- CRITICAL FIXES END ---
    
        # Update the window title to show current metric
        self.plotter.add_title(f"Mesh Quality: {self.get_current_metric_display_name()}")
    
        # Force the plotter to render the changes
        self.plotter.render()
    
        # Print info to console
        stats = self.results[current_metric]
        print(f"\nCurrent view: {self.get_current_metric_display_name()}")
        print(f"  Acceptable range: {acceptable_range}")
        print(f"  Mean: {stats['mean']:.3f}, Bad cells: {stats['bad_cells_count']} ({stats['bad_cells_count']/stats['applicable_cells']*100:.1f}%)")
    
    def next_metric(self):
        """Cycle to the next metric."""
        self.current_metric_index = (self.current_metric_index + 1) % len(self.metric_list)
        self.update_display()
    
    def previous_metric(self):
        """Cycle to the previous metric."""
        self.current_metric_index = (self.current_metric_index - 1) % len(self.metric_list)
        self.update_display()
    
    def toggle_colorbar(self):
        """Toggle colorbar visibility."""
        self.plotter.scalar_bars.toggle_visibility()
        self.plotter.render()
    
    def show(self):
        """Display the interactive viewer."""
        self.update_display()  # Initialize with first metric
        self.plotter.show()


import pyvista as pv
import numpy as np

class MeshCleaner:
    """Proper mesh cleaning with hole filling and remeshing."""
    
    def __init__(self, mesh_file: str):
        """Initialize with mesh file."""
        self.original_mesh = pv.read(mesh_file)
        self.cleaned_mesh = None
        print(f"Original mesh: {self.original_mesh.n_points} vertices, {self.original_mesh.n_cells} faces")
    
    def clean_with_hole_filling(self, hole_size=1000):
        """Clean mesh by filling holes and smoothing."""
        print(f"\nCleaning mesh with hole filling...")
        
        # Make a copy to work with
        mesh = self.original_mesh.copy()
        
        # 1. First, fill small holes
        print("Step 1: Filling holes...")
        filled_mesh = mesh.fill_holes(hole_size)
        
        # 2. Smooth the mesh to reduce artifacts
        print("Step 2: Smoothing...")
        smoothed_mesh = filled_mesh.smooth(n_iter=50, relaxation_factor=0.01)
        
        # 3. Optional: Simplify while preserving shape
        print("Step 3: Remeshing...")
        
        # Try to reconstruct surface
        try:
            # Create point cloud and reconstruct
            points = smoothed_mesh.points
            point_cloud = pv.PolyData(points)
            self.cleaned_mesh = point_cloud.reconstruct_surface()
        except:
            # Fallback: Use original smoothed mesh
            self.cleaned_mesh = smoothed_mesh
        
        print(f"Cleaned mesh: {self.cleaned_mesh.n_points} vertices, {self.cleaned_mesh.n_cells} faces")
        
        return self.cleaned_mesh
    
    def clean_with_remeshing(self, target_edge_length=None):
        """Clean mesh with proper remeshing (edge length based)."""
        print(f"\nCleaning mesh with remeshing...")
        
        mesh = self.original_mesh.copy()
        
        if target_edge_length is None:
            # Calculate average edge length from original mesh
            edges = []
            for i in range(mesh.n_cells):
                cell = mesh.get_cell(i)
                if cell.n_points >= 3:
                    # Calculate edges for triangles
                    p1, p2, p3 = cell.points
                    edges.extend([
                        np.linalg.norm(p1 - p2),
                        np.linalg.norm(p2 - p3),
                        np.linalg.norm(p3 - p1)
                    ])
            target_edge_length = np.mean(edges) * 1.2  # Slightly larger edges
        
        print(f"Target edge length: {target_edge_length:.4f}")
        
        # This is a simplified remeshing approach
        # In practice, you might want to use external libraries like Open3D or CGAL
        
        # 1. Subdivide if needed
        if target_edge_length < np.mean(edges) * 0.8:
            print("Subdividing mesh...")
            mesh = mesh.subdivide(1, 'loop')
        
        # 2. Decimate to target resolution
        target_faces = int(mesh.n_cells * (np.mean(edges) / target_edge_length))
        target_faces = max(1000, min(target_faces, mesh.n_cells))
        
        print(f"Target faces: {target_faces}")
        self.cleaned_mesh = mesh.decimate_pro(target_faces)
        
        # 3. Fill holes from decimation
        self.cleaned_mesh = self.cleaned_mesh.fill_holes(1000)
        
        # 4. Smooth
        self.cleaned_mesh = self.cleaned_mesh.smooth(n_iter=30)
        
        print(f"Remeshed: {self.cleaned_mesh.n_points} vertices, {self.cleaned_mesh.n_cells} faces")
        
        return self.cleaned_mesh
    
    def clean_with_voxel_reconstruction(self, voxel_size=None):
        """Clean mesh by converting to voxels and back (very effective for watertight meshes)."""
        print(f"\nCleaning with voxel reconstruction...")
        
        if voxel_size is None:
            # Calculate voxel size from mesh bounds
            bounds = self.original_mesh.bounds
            avg_dim = np.mean([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
            voxel_size = avg_dim / 50
        
        print(f"Voxel size: {voxel_size:.4f}")
        
        # 1. Voxelize the mesh
        print("Step 1: Voxelizing...")
        try:
            voxel_grid = self.original_mesh.voxelize(voxel_size)
        except:
            # Alternative voxelization
            voxel_grid = self._manual_voxelize(voxel_size)
        
        # 2. Extract surface from voxels (this creates a watertight mesh)
        print("Step 2: Extracting surface...")
        self.cleaned_mesh = voxel_grid.extract_surface()
        
        # 3. Optional: Smooth the result
        print("Step 3: Smoothing...")
        self.cleaned_mesh = self.cleaned_mesh.smooth(n_iter=20)
        
        print(f"Voxel reconstructed: {self.cleaned_mesh.n_points} vertices, {self.cleaned_mesh.n_cells} faces")
        
        return self.cleaned_mesh
    
    def clean_with_poisson_reconstruction(self, depth=8):
        """Clean mesh using Poisson surface reconstruction."""
        print(f"\nCleaning with Poisson reconstruction (depth={depth})...")
    
        # SIMPLEST: Use the original mesh directly since it already has faces
        print("Step 1: Using original mesh (already has faces for normal computation)...")
    
        # Create a copy of the original mesh
        mesh_copy = self.original_mesh.copy()
    
        # Ensure the mesh has normals
        print("Step 2: Ensuring mesh has normals...")
        if 'Normals' not in mesh_copy.array_names:
            mesh_copy = mesh_copy.compute_normals(cell_normals=False, point_normals=True)
    
        # For reconstruction, we need a dense point cloud with normals
        # Let's sample points from the mesh surface
        print("Step 3: Creating point cloud from mesh surface...")
        try:
            # Sample points from the mesh (preserving normals)
            # First, let's get the points and their normals
            points = mesh_copy.points
            if 'Normals' in mesh_copy.array_names:
                normals = mesh_copy['Normals']
                # Create point cloud with normals
                point_cloud = pv.PolyData(points)
                point_cloud['Normals'] = normals
            else:
                # Fallback: just use points
                point_cloud = pv.PolyData(points)
        except:
            # Ultimate fallback: use vertices directly
            point_cloud = pv.PolyData(self.original_mesh.points)
    
        # Reconstruct surface
        print("Step 4: Reconstructing surface...")
        try:
            if 'Normals' in point_cloud.array_names:
                # If we have normals, use them for reconstruction
                self.cleaned_mesh = point_cloud.reconstruct_surface()
            else:
                # Try Delaunay triangulation
                self.cleaned_mesh = point_cloud.delaunay_3d()
        except Exception as e:
            print(f"Surface reconstruction failed: {e}")
            print("Using original mesh with hole filling...")
            self.cleaned_mesh = self.original_mesh.copy()
    
        # Fill any remaining holes
        if self.cleaned_mesh.n_cells > 0:
            self.cleaned_mesh = self.cleaned_mesh.fill_holes(1000)
            # Smooth the result
            self.cleaned_mesh = self.cleaned_mesh.smooth(n_iter=30)
    
        print(f"Poisson reconstructed: {self.cleaned_mesh.n_points} vertices, {self.cleaned_mesh.n_cells} faces")
    
        return self.cleaned_mesh
    
    def detect_and_visualize_holes(self):
        """Detect and visualize holes in the mesh."""
        print("\nDetecting holes...")
        
        mesh = self.original_mesh.copy()
        
        # Extract edges and find boundary edges
        edges = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False
        )
        
        if edges.n_points == 0:
            print("No holes detected - mesh appears watertight!")
            return None
        
        print(f"Found {edges.n_cells} hole edges")
        
        # Visualize holes
        plotter = pv.Plotter(shape=(1, 2))
        
        # Left: Original mesh with holes highlighted
        plotter.subplot(0, 0)
        plotter.add_text("Original Mesh with Holes", font_size=12)
        plotter.add_mesh(
            mesh,
            color='lightblue',
            opacity=1,
            show_edges=True,
            edge_color='gray'
        )
        plotter.add_mesh(
            edges,
            color='red',
            line_width=5,
            render_lines_as_tubes=True
        )
        
        # Right: Fixed mesh
        if self.cleaned_mesh is not None:
            plotter.subplot(0, 1)
            plotter.add_text("Cleaned Mesh", font_size=12)
            plotter.add_mesh(
                self.cleaned_mesh,
                color='lightgreen',
                opacity=1,
                show_edges=True,
                edge_color='gray'
            )
            
            # Check if holes are fixed
            fixed_edges = self.cleaned_mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False
            )
            if fixed_edges.n_points > 0:
                plotter.add_mesh(
                    fixed_edges,
                    color='orange',
                    line_width=3,
                    render_lines_as_tubes=True
                )
                plotter.add_text(
                    f"Still has {fixed_edges.n_cells} holes",
                    position='lower_left',
                    color='orange'
                )
            else:
                plotter.add_text(
                    "All holes filled!",
                    position='lower_left',
                    color='green'
                )
        
        plotter.link_views()
        plotter.show()
        
        return edges
    
    def calculate_watertightness(self):
        """Calculate how watertight the mesh is."""
        mesh = self.cleaned_mesh if self.cleaned_mesh else self.original_mesh
        
        edges = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False
        )
        
        if edges.n_points == 0:
            print("? Mesh is watertight (no boundary edges)")
            return True
        else:
            # Calculate hole size metrics
            total_edge_length = 0
            for i in range(edges.n_cells):
                cell = edges.get_cell(i)
                if cell.n_points == 2:
                    p1, p2 = cell.points
                    total_edge_length += np.linalg.norm(p1 - p2)
            
            bounds = mesh.bounds
            mesh_size = np.mean([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
            
            print(f"? Mesh has {edges.n_cells} holes")
            print(f"  Total hole perimeter: {total_edge_length:.3f}")
            print(f"  Relative to mesh size: {(total_edge_length/mesh_size)*100:.1f}%")
            
            return False
    
    def visualize_side_by_side_with_progress(self):
        """Visualize cleaning progress with multiple views."""
        if self.cleaned_mesh is None:
            print("No cleaned mesh to visualize.")
            return
        
        plotter = pv.Plotter(shape=(1, 3))
        
        # View 1: Original mesh
        plotter.subplot(0, 0)
        plotter.add_text("Original Mesh", font_size=12)
        plotter.add_mesh(
            self.original_mesh,
            color='lightblue',
            opacity=1,
            show_edges=True,
            edge_color='black',
            line_width=0.5
        )
        plotter.add_text(
            f"Vertices: {self.original_mesh.n_points:,}\nFaces: {self.original_mesh.n_cells:,}",
            position='lower_left',
            font_size=9
        )
        
        # View 2: Cleaned mesh
        plotter.subplot(0, 1)
        plotter.add_text("Cleaned Mesh", font_size=12)
        plotter.add_mesh(
            self.cleaned_mesh,
            color='lightgreen',
            opacity=1,
            show_edges=True,
            edge_color='black',
            line_width=0.5
        )
        plotter.add_text(
            f"Vertices: {self.cleaned_mesh.n_points:,}\nFaces: {self.cleaned_mesh.n_cells:,}",
            position='lower_left',
            font_size=9
        )
        
        # View 3: Wireframe comparison
        plotter.subplot(0, 2)
        plotter.add_text("Wireframe Comparison", font_size=12)
        
        # Show both meshes in wireframe
        plotter.add_mesh(
            self.original_mesh,
            color='blue',
            opacity=0.5,
            style='wireframe',
            line_width=1
        )
        plotter.add_mesh(
            self.cleaned_mesh,
            color='green',
            opacity=0.5,
            style='wireframe',
            line_width=2
        )
        
        # Calculate and display statistics
        v_reduction = (1 - self.cleaned_mesh.n_points/self.original_mesh.n_points)*100
        f_reduction = (1 - self.cleaned_mesh.n_cells/self.original_mesh.n_cells)*100
        
        plotter.add_text(
            f"Vertex reduction: {v_reduction:.1f}%\nFace reduction: {f_reduction:.1f}%",
            position='upper_right',
            font_size=10,
            color='red'
        )
        
        plotter.link_views()
        
        # Check watertightness
        is_watertight = self.calculate_watertightness()
        status_color = 'green' if is_watertight else 'orange'
        status_text = "Watertight ?" if is_watertight else "Has holes ?"
        
        plotter.add_text(
            f"Status: {status_text}",
            position='lower_edge',
            font_size=11,
            color=status_color
        )
        
        print("\n" + "="*60)
        print("VISUALIZATION CONTROLS")
        print("="*60)
        print("? Click and drag to rotate any view")
        print("? All views are linked (rotate together)")
        print("? Press 'r' to reset camera")
        print("? Press 'w' to toggle wireframe")
        print("? Close window to continue")
        
        plotter.show()

# Update main function
def main():
    print("=" * 60)
    print("STEP 1: Mesh Quality Analysis")
    print("=" * 60)
    
    analyzer = MeshQualityAnalyzerFixed('3DModel.obj')
    results = analyzer.calculate_all_metrics()
    
    print("\n" + "=" * 60)
    print("STEP 2: Interactive Quality Inspection")
    print("=" * 60)
    
    viewer = InteractiveQualityViewer('3DModel.obj')
    print("\nClose window to continue to mesh cleaning...")
    viewer.show()
    
    print("\n" + "=" * 60)
    print("STEP 3: Advanced Mesh Cleaning with Hole Filling")
    print("=" * 60)
    
    cleaner = MeshCleaner('3DModel.obj')
    
    # First detect holes
    print("\n1. Detecting existing holes...")
    cleaner.detect_and_visualize_holes()
    
    # Try different cleaning methods
    methods = [
        ("Poisson", lambda: cleaner.clean_with_poisson_reconstruction()),
        ("Hole Filling", lambda: cleaner.clean_with_hole_filling()),
        ("Remeshing", lambda: cleaner.clean_with_remeshing()),
    ]
    
    for i, (name, method_func) in enumerate(methods, 1):
        print(f"\n{i}. Trying {name}...")
        method_func()
        cleaner.visualize_side_by_side_with_progress()
        
        if i < len(methods):
            response = input(f"\nTry next method ({i+1}/{len(methods)})? (y/n): ")
            if response.lower() != 'y':
                break
    
    print("\n" + "=" * 60)
    print("Cleaning complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()