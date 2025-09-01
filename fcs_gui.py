import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import sys
from collections import deque
import time
import os
from datetime import datetime
import cv2  # For video encoding

class FCSSimulation:
    def __init__(self):
        # Physical constants
        self.k_B = 1.38e-23  # Boltzmann constant (J/K)
        self.temperature = 298.15  # Temperature (K)
        self.viscosity = 8.9e-4  # Viscosity of water (Pa*s)
        self.avogadro = 6.022e23  # Avogadro's number
        
        # Confocal volume parameters
        self.w_xy = 0.4e-6  # Radial radius (meters)
        self.w_z = 1.5e-6   # Axial radius (meters)
        
        # Two particle types configuration
        self.two_particle_mode = False  # Enable/disable second particle type
        self.separate_traces = False  # If True, each particle type has its own trace
        
        # Particle type 1 parameters
        self.hydrodynamic_radius_1 = 1.0e-9  # 1 nm
        self.concentration_nM_1 = 1.0  # Concentration in nanomolar
        self.brightness_1 = 1.0  # Relative brightness

        # Particle type 2 parameters
        self.hydrodynamic_radius_2 = 5.0e-9  # 5 nm (larger, slower)
        self.concentration_nM_2 = 1.0  # Concentration in nanomolar
        self.brightness_2 = 1.5  # Relative brightness
        
        # Legacy single particle parameters (map to type 1)
        self.hydrodynamic_radius = self.hydrodynamic_radius_1
        self.concentration_nM = self.concentration_nM_1
        
        self.time_step = 1e-5  # 10 microseconds
        
        # Simulation mode
        self.is_2d_mode = False  # Default to 3D mode
        
        # Simulation box
        self.box_size_factor = 3.0
        self.box_size = self.box_size_factor * self.w_z
        
        # Calculate number of particles from concentration
        self.update_particle_count_from_concentration()
        
        # Calculate diffusion coefficients
        self.update_diffusion_coefficient()
        
        # Initialize particles
        self.init_particles()
        
        # Intensity trace buffers
        self.intensity_buffer_size = 2000
        self.intensity_trace = deque(maxlen=self.intensity_buffer_size)
        self.intensity_trace_1 = deque(maxlen=self.intensity_buffer_size)  # Type 1 only
        self.intensity_trace_2 = deque(maxlen=self.intensity_buffer_size)  # Type 2 only
        self.time_trace = deque(maxlen=self.intensity_buffer_size)
        self.current_time = 0
        
        # Autocorrelation
        self.G_tau = None
        self.tau_axis = None
        
    def update_particle_count_from_concentration(self):
        """Calculate number of particles from concentration and box volume"""
        box_volume_m3 = (2 * self.box_size) ** 3
        box_volume_L = box_volume_m3 * 1000
        
        # Type 1 particles
        num_molecules_1 = self.concentration_nM_1 * 1e-9 * box_volume_L * self.avogadro
        self.num_particles_1 = max(1, int(round(num_molecules_1)))
        
        # Type 2 particles (if enabled)
        if self.two_particle_mode:
            num_molecules_2 = self.concentration_nM_2 * 1e-9 * box_volume_L * self.avogadro
            self.num_particles_2 = max(1, int(round(num_molecules_2)))
        else:
            self.num_particles_2 = 0
            
        # Total particles
        self.num_particles = self.num_particles_1 + self.num_particles_2
        
        # Store particle type indices
        self.particle_types = np.zeros(self.num_particles, dtype=int)
        if self.two_particle_mode:
            self.particle_types[self.num_particles_1:] = 1  # Mark type 2 particles
        
    def update_diffusion_coefficient(self):
        """Calculate diffusion coefficients using Stokes-Einstein equation"""
        # Type 1 diffusion
        self.D_1 = (self.k_B * self.temperature) / (6 * np.pi * self.viscosity * self.hydrodynamic_radius_1)
        self.step_size_1 = np.sqrt(2 * self.D_1 * self.time_step)
        
        # Type 2 diffusion
        self.D_2 = (self.k_B * self.temperature) / (6 * np.pi * self.viscosity * self.hydrodynamic_radius_2)
        self.step_size_2 = np.sqrt(2 * self.D_2 * self.time_step)
        
        # Legacy single particle (use type 1)
        self.D = self.D_1
        self.step_size = self.step_size_1
        
    def init_particles(self):
        """Initialize particle positions randomly within the box"""
        if self.is_2d_mode:
            # 2D mode: particles only in XY plane at z=0
            self.positions = np.random.uniform(-self.box_size, self.box_size, (self.num_particles, 3))
            self.positions[:, 2] = 0  # Constrain all particles to z=0
        else:
            # 3D mode: particles distributed throughout volume
            self.positions = np.random.uniform(-self.box_size, self.box_size, (self.num_particles, 3))
        self.velocities = np.zeros((self.num_particles, 3))  # For smooth visualization
        
    def update_positions(self):
        """Update particle positions with Brownian motion"""
        # Store previous positions to detect boundary crossings
        old_positions = self.positions.copy()
        
        # Create steps array
        steps = np.zeros((self.num_particles, 3))
        
        # Generate steps based on particle type
        if self.two_particle_mode:
            # Type 1 particles
            type1_mask = self.particle_types == 0
            steps[type1_mask] = np.random.normal(0, self.step_size_1, (np.sum(type1_mask), 3))
            
            # Type 2 particles (slower diffusion)
            type2_mask = self.particle_types == 1
            steps[type2_mask] = np.random.normal(0, self.step_size_2, (np.sum(type2_mask), 3))
        else:
            # Single particle type
            steps = np.random.normal(0, self.step_size, (self.num_particles, 3))
        
        if self.is_2d_mode:
            # 2D mode: only XY diffusion, no Z movement
            steps[:, 2] = 0  # No Z-direction movement
            self.positions += steps
            
            # Ensure particles stay at z=0
            self.positions[:, 2] = 0
        else:
            # 3D mode: full 3D diffusion
            self.positions += steps
        
        # Smooth velocity for visualization (not physically accurate, just for trails)
        self.velocities = self.velocities * 0.9 + steps * 0.1
        
        # Apply periodic boundary conditions and detect wrapping
        boundary_wrapped = np.zeros(self.num_particles, dtype=bool)
        
        # Check for boundary crossings in each dimension
        for axis in range(3):
            # Particles that crossed the upper boundary
            upper_crossed = self.positions[:, axis] > self.box_size
            # Particles that crossed the lower boundary  
            lower_crossed = self.positions[:, axis] < -self.box_size
            
            # Mark particles that wrapped around
            boundary_wrapped |= upper_crossed | lower_crossed
            
            # Apply wrapping
            self.positions[upper_crossed, axis] -= 2*self.box_size
            self.positions[lower_crossed, axis] += 2*self.box_size
        
        # In 2D mode, ensure particles remain at z=0 after boundary conditions
        if self.is_2d_mode:
            self.positions[:, 2] = 0
            
        return boundary_wrapped
        
    def calculate_intensity(self):
        """Calculate total fluorescence intensity - vectorized"""
        if self.is_2d_mode:
            # 2D mode: only radial component (z=0 always)
            r_xy_squared = (self.positions[:, 0]**2 + self.positions[:, 1]**2) / self.w_xy**2
            individual_intensities = np.exp(-2 * r_xy_squared)
        else:
            # 3D mode: full 3D Gaussian
            r_squared = (self.positions[:, 0]**2 + self.positions[:, 1]**2) / self.w_xy**2 + \
                       self.positions[:, 2]**2 / self.w_z**2
            individual_intensities = np.exp(-2 * r_squared)
        
        # Apply brightness factors for different particle types
        if self.two_particle_mode:
            # Apply different brightness to each particle type
            brightness = np.ones(self.num_particles)
            brightness[self.particle_types == 0] = self.brightness_1
            brightness[self.particle_types == 1] = self.brightness_2
            individual_intensities *= brightness
            
            # Calculate separate intensities if needed
            if self.separate_traces:
                intensity_1 = np.sum(individual_intensities[self.particle_types == 0])
                intensity_2 = np.sum(individual_intensities[self.particle_types == 1])
                return np.sum(individual_intensities), intensity_1, intensity_2
        
        return np.sum(individual_intensities)
    
    def get_particle_colors_and_sizes(self):
        """Get colors and sizes based on position in confocal volume and particle type"""
        if self.is_2d_mode:
            # 2D mode: only consider radial distance (z=0)
            r_xy_squared = (self.positions[:, 0]**2 + self.positions[:, 1]**2) / self.w_xy**2
            inside = r_xy_squared < 2  # Within 2 sigma radially
        else:
            # 3D mode: consider full 3D distance
            r_squared = (self.positions[:, 0]**2 + self.positions[:, 1]**2) / self.w_xy**2 + \
                       self.positions[:, 2]**2 / self.w_z**2
            inside = r_squared < 2  # Within 2 sigma
        
        colors = np.zeros((self.num_particles, 4))
        sizes = np.ones(self.num_particles) * 5  # Default size
        
        if self.two_particle_mode:
            # Type 1: Green when inside, light gray when outside
            type1_mask = self.particle_types == 0
            colors[type1_mask & inside] = [0.0, 1.0, 0.0, 1.0]  # Bright green
            colors[type1_mask & ~inside] = [0.7, 0.7, 0.7, 0.6]  # Gray
            
            # Type 2: Blue when inside, light gray when outside  
            type2_mask = self.particle_types == 1
            colors[type2_mask & inside] = [0.0, 0.5, 1.0, 1.0]  # Bright blue
            colors[type2_mask & ~inside] = [0.7, 0.7, 0.7, 0.6]  # Gray
            
            # Type 2 particles are larger (slower diffusion)
            sizes[type2_mask] = 8  # Larger base size for type 2
            sizes[type2_mask & inside] = 15  # Even larger when inside
            sizes[type1_mask & inside] = 10  # Type 1 inside
        else:
            # Single particle type: green for inside, gray for outside
            colors[inside] = [0.0, 1.0, 0.0, 1.0]  # Bright green for inside
            colors[~inside] = [0.7, 0.7, 0.7, 0.6]  # Gray for outside
            sizes[inside] = 10  # Larger when inside
        
        return colors, sizes
    
    def calculate_autocorrelation_fft(self, intensity_trace):
        """Fast autocorrelation using FFT"""
        if len(intensity_trace) < 50:
            return None, None
            
        intensity = np.array(intensity_trace)
        mean_I = np.mean(intensity)
        
        if mean_I == 0:
            return None, None
            
        # Normalize
        intensity_norm = intensity - mean_I
        
        # FFT-based autocorrelation
        n = len(intensity_norm)
        # Use power of 2 for faster FFT
        n_padded = 2**int(np.ceil(np.log2(2*n)))
        padded = np.zeros(n_padded)
        padded[:n] = intensity_norm
        
        fft = np.fft.fft(padded)
        acf = np.fft.ifft(fft * np.conj(fft))[:n].real
        acf = acf / (np.arange(n, 0, -1) * mean_I**2)
        
        # Create lag time axis
        max_lag = min(n // 2, 500)
        tau = np.arange(1, max_lag) * self.time_step
        G = acf[1:max_lag]
        
        return tau, G

class FCSVisualizationPyQtGraph(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.sim = FCSSimulation()
        self.is_paused = False
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Performance optimization: Variable update rates
        self.physics_rate = 1      # Physics every frame (for accuracy)
        self.visual_rate = 2       # 3D visualization every 2 frames
        self.trail_rate = 5        # Trails every 5 frames
        self.plot_rate = 3         # Plots every 3 frames
        self.acf_rate = 200        # Autocorrelation every 200 frames
        
        # Frame counters
        self.physics_counter = 0
        self.visual_counter = 0
        self.trail_counter = 0
        self.plot_counter = 0
        self.acf_counter = 0
        
        # Level-of-detail system
        self.max_trail_particles = 20  # Limit trail rendering to N particles
        self.max_visual_particles = 200  # Simplify visualization above N particles
        
        # Video recording
        self.is_recording = False
        self.video_writer = None
        self.video_filename = None
        self.video_frame_count = 0
        self.video_fps = 30  # Video output FPS
        self.video_quality = 'Medium'  # Low, Medium, High
        
        # Trail management
        self.trail_length = 30  # Shorter trails for performance
        self.trail_update_rate = 1  # Legacy support
        self.trail_update_counter = 0
        self.trails = [deque(maxlen=self.trail_length) for _ in range(min(self.sim.num_particles, self.max_trail_particles))]
        
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """Setup the PyQtGraph UI"""
        self.setWindowTitle('FCS Simulation - PyQtGraph (Optimized)')
        self.resize(1400, 900)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)
        
        # Top section: 3D view and plots
        top_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_layout)
        
        # Left: 3D visualization
        self.setup_3d_view()
        top_layout.addWidget(self.gl_widget, 2)
        
        # Right: Plots
        plots_layout = QtWidgets.QVBoxLayout()
        top_layout.addLayout(plots_layout, 1)
        
        # Intensity plot
        self.setup_intensity_plot()
        plots_layout.addWidget(self.intensity_widget)
        
        # Autocorrelation plot
        self.setup_autocorrelation_plot()
        plots_layout.addWidget(self.autocorr_widget)
        
        # Bottom: Controls
        self.setup_controls()
        main_layout.addWidget(self.controls_widget)
        
        # Status bar
        self.status_label = QtWidgets.QLabel('Starting simulation...')
        main_layout.addWidget(self.status_label)
        
    def setup_3d_view(self):
        """Setup 3D visualization with PyQtGraph OpenGL"""
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=self.sim.box_size*4e6)
        self.gl_widget.setWindowTitle('3D Particle Visualization')
        
        # Add reference grids centered at origin
        # XY grid at z=0 (horizontal surface)
        gxy = gl.GLGridItem()
        gxy.scale(self.sim.w_z*1e6/5, self.sim.w_z*1e6/5, 1)  # Scale to confocal volume size
        gxy.translate(0, 0, 0)
        self.gl_widget.addItem(gxy)
        
        # XZ grid at y=0 
        gxz = gl.GLGridItem()
        gxz.rotate(90, 1, 0, 0)
        gxz.scale(self.sim.w_z*1e6/5, self.sim.w_z*1e6/5, 1)
        gxz.translate(0, 0, 0)
        self.gl_widget.addItem(gxz)
        
        # YZ grid at x=0
        gyz = gl.GLGridItem()
        gyz.rotate(90, 0, 1, 0)
        gyz.scale(self.sim.w_z*1e6/5, self.sim.w_z*1e6/5, 1)
        gyz.translate(0, 0, 0)
        self.gl_widget.addItem(gyz)
        
        # Add confocal volume as mesh
        self.add_confocal_volume()
        
        # Add 2D plane indicator (will be shown/hidden based on mode)
        self.add_2d_plane()
        
        # Update visibility based on initial mode
        self.update_grid_visibility()
        
        # Add simulation box boundaries
        self.add_box_boundaries()
        
        # Initialize particle scatter plot
        pos = self.sim.positions * 1e6  # Convert to micrometers
        colors, sizes = self.sim.get_particle_colors_and_sizes()
        self.particles = gl.GLScatterPlotItem(
            pos=pos,
            color=colors,
            size=sizes,
            pxMode=True
        )
        self.gl_widget.addItem(self.particles)
        
        # Initialize trail lines (limited number for performance)
        self.trail_lines = []
        self.max_trail_lines = 20  # Limit number of trail lines
        
    def add_confocal_volume(self):
        """Add confocal volume visualization using 3D Gaussian scalar field and isosurfaces"""
        print("Generating 3D Gaussian field for confocal volume...")
        
        # Define 3D grid size (keep reasonable for performance)
        grid_size = 40
        
        # Create coordinate arrays centered at origin
        # Make grid extend to ~2 sigma in each direction
        x_range = 2 * self.sim.w_xy * 1e6
        z_range = 2 * self.sim.w_z * 1e6
        
        def gaussian_intensity(i, j, k, offset=(grid_size//2, grid_size//2, grid_size//2)):
            """3D Gaussian intensity function for confocal volume"""
            # Convert indices to physical coordinates (centered at origin)
            x = (i - offset[0]) * (2 * x_range) / grid_size
            y = (j - offset[1]) * (2 * x_range) / grid_size  
            z = (k - offset[2]) * (2 * z_range) / grid_size
            
            # Calculate normalized distances
            r_xy_squared = (x**2 + y**2) / (self.sim.w_xy * 1e6)**2
            r_z_squared = z**2 / (self.sim.w_z * 1e6)**2
            
            # 3D Gaussian intensity profile
            intensity = np.exp(-2 * (r_xy_squared + r_z_squared))
            return intensity
        
        # Generate the 3D scalar field
        data = np.fromfunction(gaussian_intensity, (grid_size, grid_size, grid_size))
        max_intensity = data.max()
        
        # Create multiple isosurfaces at different intensity levels
        intensity_levels = [0.7, 0.5, 0.3, 0.15, 0.05]  # From bright center to dim edge
        colors = [
            [1.0, 1.0, 1.0, 0.8],  # White (70%)
            [1.0, 1.0, 0.0, 0.6],  # Yellow (50%)
            [1.0, 0.5, 0.0, 0.5],  # Orange (30%)
            [1.0, 0.0, 0.0, 0.4],  # Red (15%)
            [0.5, 0.0, 0.5, 0.3]   # Purple (5%)
        ]
        
        for i, (level, color) in enumerate(zip(intensity_levels, colors)):
            threshold = level * max_intensity
            
            print(f"Generating isosurface at {level*100:.0f}% intensity...")
            try:
                verts, faces = pg.isosurface(data, threshold)
                
                if len(verts) > 0:
                    # Scale vertices to proper physical coordinates (centered at origin)
                    verts_scaled = verts.copy()
                    verts_scaled[:, 0] = (verts[:, 0] - grid_size//2) * (2 * x_range) / grid_size
                    verts_scaled[:, 1] = (verts[:, 1] - grid_size//2) * (2 * x_range) / grid_size
                    verts_scaled[:, 2] = (verts[:, 2] - grid_size//2) * (2 * z_range) / grid_size
                    
                    # Create vertex colors
                    vertex_colors = np.ones((len(verts_scaled), 4), dtype=np.float32)
                    vertex_colors[:] = color
                    
                    # Create mesh data
                    mesh_data = gl.MeshData(vertexes=verts_scaled, faces=faces, vertexColors=vertex_colors)
                    
                    # Create mesh item with balloon shader for better 3D appearance
                    mesh = gl.GLMeshItem(meshdata=mesh_data, smooth=True, shader='balloon')
                    mesh.setGLOptions('translucent')
                    
                    self.gl_widget.addItem(mesh)
                    
            except Exception as e:
                print(f"Could not create isosurface at {level*100:.0f}% level: {e}")
                continue
        
        print("Confocal volume visualization complete.")
        
    def add_box_boundaries(self):
        """Add simulation box boundaries"""
        box = self.sim.box_size * 1e6
        
        # Create box edges
        lines = []
        colors_list = []
        
        # Define the 12 edges of a box
        edges = [
            # Bottom square
            [[-box, -box, -box], [box, -box, -box]],
            [[box, -box, -box], [box, box, -box]],
            [[box, box, -box], [-box, box, -box]],
            [[-box, box, -box], [-box, -box, -box]],
            # Top square
            [[-box, -box, box], [box, -box, box]],
            [[box, -box, box], [box, box, box]],
            [[box, box, box], [-box, box, box]],
            [[-box, box, box], [-box, -box, box]],
            # Vertical edges
            [[-box, -box, -box], [-box, -box, box]],
            [[box, -box, -box], [box, -box, box]],
            [[box, box, -box], [box, box, box]],
            [[-box, box, -box], [-box, box, box]]
        ]
        
        for edge in edges:
            lines.append(edge)
            colors_list.append([0.5, 0.5, 0.5, 0.3])
            
        for line, color in zip(lines, colors_list):
            line_item = gl.GLLinePlotItem(pos=np.array(line), color=color, width=1)
            self.gl_widget.addItem(line_item)
            
    def setup_intensity_plot(self):
        """Setup intensity trace plot"""
        self.intensity_widget = pg.PlotWidget(title="Fluorescence Intensity")
        self.intensity_widget.setLabel('left', 'Intensity', units='a.u.')
        self.intensity_widget.setLabel('bottom', 'Time', units='ms')
        self.intensity_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add legend right away
        self.intensity_legend = self.intensity_widget.addLegend(offset=(10, 10))
        self.intensity_legend.setBrush(pg.mkBrush(50, 50, 50, 180))  # Dark semi-transparent
        self.intensity_legend.setPen(pg.mkPen(200, 200, 200, 100))  # Light border
        
        # Create plot curves with thicker lines for better visibility
        self.intensity_curve = self.intensity_widget.plot(pen=pg.mkPen('y', width=2), name='Total')
        # Additional curves for separate particle types (initially hidden)
        self.intensity_curve_1 = self.intensity_widget.plot(pen=pg.mkPen('g', width=2), name='Type 1')
        self.intensity_curve_2 = self.intensity_widget.plot(pen=pg.mkPen('b', width=2), name='Type 2')
        self.intensity_curve_1.hide()
        self.intensity_curve_2.hide()
        
        # Hide legend initially
        self.intensity_legend.setVisible(False)
        
    def setup_autocorrelation_plot(self):
        """Setup autocorrelation plot"""
        self.autocorr_widget = pg.PlotWidget(title="Autocorrelation G(τ)")
        self.autocorr_widget.setLabel('left', 'G(τ)')
        self.autocorr_widget.setLabel('bottom', 'τ', units='ms')
        self.autocorr_widget.setLogMode(x=True, y=False)
        self.autocorr_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add legend right away
        self.autocorr_legend = self.autocorr_widget.addLegend(offset=(10, 10))
        self.autocorr_legend.setBrush(pg.mkBrush(50, 50, 50, 180))  # Dark semi-transparent
        self.autocorr_legend.setPen(pg.mkPen(200, 200, 200, 100))  # Light border
        
        # Create plot curves with thicker lines
        self.autocorr_curve = self.autocorr_widget.plot(pen=pg.mkPen('r', width=2), name='Total')
        # Additional curves for separate particle types (initially hidden)
        self.autocorr_curve_1 = self.autocorr_widget.plot(pen=pg.mkPen('g', width=2), name='Type 1')
        self.autocorr_curve_2 = self.autocorr_widget.plot(pen=pg.mkPen('b', width=2), name='Type 2')
        self.autocorr_curve_1.hide()
        self.autocorr_curve_2.hide()
        
        # Hide legend initially
        self.autocorr_legend.setVisible(False)
        
    def setup_controls(self):
        """Setup control widgets"""
        # Create tab widget for organized controls
        self.control_tabs = QtWidgets.QTabWidget()
        
        # Tab 1: Main Controls
        main_tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QGridLayout()
        main_tab.setLayout(main_layout)
        
        row = 0
        
        # Concentration control
        main_layout.addWidget(QtWidgets.QLabel('Concentration (nM):'), row, 0)
        self.conc_spin = QtWidgets.QDoubleSpinBox()
        self.conc_spin.setRange(0.1, 100)
        self.conc_spin.setValue(self.sim.concentration_nM)
        self.conc_spin.setSingleStep(0.5)
        self.conc_spin.valueChanged.connect(self.update_concentration)
        main_layout.addWidget(self.conc_spin, row, 1)
        
        # Box size control
        main_layout.addWidget(QtWidgets.QLabel('Box Size Factor:'), row, 2)
        self.box_spin = QtWidgets.QDoubleSpinBox()
        self.box_spin.setRange(2, 10)
        self.box_spin.setValue(self.sim.box_size_factor)
        self.box_spin.setSingleStep(0.5)
        self.box_spin.valueChanged.connect(self.update_box_size)
        main_layout.addWidget(self.box_spin, row, 3)
        
        # Particle radius control
        main_layout.addWidget(QtWidgets.QLabel('Radius (nm):'), row, 4)
        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(1, 20)
        self.radius_spin.setValue(self.sim.hydrodynamic_radius * 1e9)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.valueChanged.connect(self.update_radius)
        main_layout.addWidget(self.radius_spin, row, 5)
        
        row += 1
        
        # Time step control
        main_layout.addWidget(QtWidgets.QLabel('Time Step (μs):'), row, 0)
        self.timestep_spin = QtWidgets.QSpinBox()
        self.timestep_spin.setRange(1, 500)
        self.timestep_spin.setValue(int(self.sim.time_step * 1e6))
        self.timestep_spin.setSingleStep(5)
        self.timestep_spin.valueChanged.connect(self.update_timestep)
        main_layout.addWidget(self.timestep_spin, row, 1)
        
        # Pause button
        self.pause_btn = QtWidgets.QPushButton('Pause')
        self.pause_btn.clicked.connect(self.toggle_pause)
        main_layout.addWidget(self.pause_btn, row, 2)
        
        # Reset button
        self.reset_btn = QtWidgets.QPushButton('Reset')
        self.reset_btn.clicked.connect(self.reset_simulation)
        main_layout.addWidget(self.reset_btn, row, 3)
        
        # 2D/3D mode toggle
        self.mode_2d_check = QtWidgets.QCheckBox('2D Mode')
        self.mode_2d_check.setChecked(self.sim.is_2d_mode)
        self.mode_2d_check.stateChanged.connect(self.toggle_2d_mode)
        main_layout.addWidget(self.mode_2d_check, row, 4)
        
        # Trail toggle
        self.trail_check = QtWidgets.QCheckBox('Show Trails')
        self.trail_check.setChecked(True)
        main_layout.addWidget(self.trail_check, row, 5)
        
        row += 1  # Add a new row for performance controls
        
        # Performance Level Control
        main_layout.addWidget(QtWidgets.QLabel('Performance:'), row, 0)
        self.performance_combo = QtWidgets.QComboBox()
        self.performance_combo.addItems(['Balanced', 'High Quality', 'Max Speed'])
        self.performance_combo.setCurrentText('Balanced')
        self.performance_combo.currentTextChanged.connect(self.update_performance_level)
        main_layout.addWidget(self.performance_combo, row, 1)
        
        # Max Particles Display Control
        main_layout.addWidget(QtWidgets.QLabel('Max Trails:'), row, 2)
        self.max_trails_spin = QtWidgets.QSpinBox()
        self.max_trails_spin.setRange(1, 100)
        self.max_trails_spin.setValue(self.max_trail_particles)
        self.max_trails_spin.valueChanged.connect(self.update_max_trails)
        main_layout.addWidget(self.max_trails_spin, row, 3)
        
        # FPS display
        self.fps_label = QtWidgets.QLabel('FPS: 0')
        main_layout.addWidget(self.fps_label, row, 4)
        
        # Add main tab to tab widget
        self.control_tabs.addTab(main_tab, "Main Controls")
        
        # Tab 2: Two-Particle Mode
        particle_tab = QtWidgets.QWidget()
        particle_layout = QtWidgets.QGridLayout()
        particle_tab.setLayout(particle_layout)
        
        # Enable two-particle mode checkbox
        self.two_particle_check = QtWidgets.QCheckBox('Enable Second Particle Type')
        self.two_particle_check.setChecked(self.sim.two_particle_mode)
        self.two_particle_check.stateChanged.connect(self.toggle_two_particle_mode)
        particle_layout.addWidget(self.two_particle_check, 0, 0, 1, 3)
        
        # Separate traces checkbox
        self.separate_traces_check = QtWidgets.QCheckBox('Separate Intensity Traces')
        self.separate_traces_check.setChecked(self.sim.separate_traces)
        self.separate_traces_check.stateChanged.connect(self.toggle_separate_traces)
        particle_layout.addWidget(self.separate_traces_check, 0, 3, 1, 3)
        
        # Type 2 concentration
        particle_layout.addWidget(QtWidgets.QLabel('Type 2 Conc (nM):'), 1, 0)
        self.conc2_spin = QtWidgets.QDoubleSpinBox()
        self.conc2_spin.setRange(0.1, 100)
        self.conc2_spin.setValue(self.sim.concentration_nM_2)
        self.conc2_spin.setSingleStep(0.5)
        self.conc2_spin.valueChanged.connect(self.update_concentration_2)
        particle_layout.addWidget(self.conc2_spin, 1, 1)
        
        # Type 2 radius
        particle_layout.addWidget(QtWidgets.QLabel('Type 2 Radius (nm):'), 1, 2)
        self.radius2_spin = QtWidgets.QDoubleSpinBox()
        self.radius2_spin.setRange(1, 50)
        self.radius2_spin.setValue(self.sim.hydrodynamic_radius_2 * 1e9)
        self.radius2_spin.setSingleStep(0.5)
        self.radius2_spin.valueChanged.connect(self.update_radius_2)
        particle_layout.addWidget(self.radius2_spin, 1, 3)
        
        # Type 2 brightness
        particle_layout.addWidget(QtWidgets.QLabel('Type 2 Brightness:'), 1, 4)
        self.brightness2_spin = QtWidgets.QDoubleSpinBox()
        self.brightness2_spin.setRange(0.1, 5.0)
        self.brightness2_spin.setValue(self.sim.brightness_2)
        self.brightness2_spin.setSingleStep(0.1)
        self.brightness2_spin.valueChanged.connect(self.update_brightness_2)
        particle_layout.addWidget(self.brightness2_spin, 1, 5)
        
        # Add spacer to push everything to top
        particle_layout.setRowStretch(2, 1)
        
        # Add particle tab to tab widget
        self.control_tabs.addTab(particle_tab, "Two-Particle Mode")
        
        # Tab 3: Video Recording
        video_tab = QtWidgets.QWidget()
        video_layout = QtWidgets.QHBoxLayout()
        video_tab.setLayout(video_layout)
        
        # Record button
        self.record_btn = QtWidgets.QPushButton('● Start Recording')
        self.record_btn.setStyleSheet(
            "QPushButton { background-color: #ff4444; color: white; font-weight: bold; }"
        )
        self.record_btn.clicked.connect(self.toggle_recording)
        video_layout.addWidget(self.record_btn)
        
        # Video quality selector
        video_layout.addWidget(QtWidgets.QLabel('Quality:'))
        self.video_quality_combo = QtWidgets.QComboBox()
        self.video_quality_combo.addItems(['Low', 'Medium', 'High'])
        self.video_quality_combo.setCurrentText('Medium')
        self.video_quality_combo.currentTextChanged.connect(self.update_video_quality)
        video_layout.addWidget(self.video_quality_combo)
        
        # Video FPS selector
        video_layout.addWidget(QtWidgets.QLabel('FPS:'))
        self.video_fps_spin = QtWidgets.QSpinBox()
        self.video_fps_spin.setRange(15, 60)
        self.video_fps_spin.setValue(30)
        self.video_fps_spin.valueChanged.connect(self.update_video_fps)
        video_layout.addWidget(self.video_fps_spin)
        
        # Recording status
        self.recording_status = QtWidgets.QLabel('Ready to record')
        self.recording_status.setStyleSheet("color: green; font-weight: bold;")
        video_layout.addWidget(self.recording_status)
        
        # Add spacer to left-align controls
        video_layout.addStretch()
        
        # Add video tab to tab widget
        self.control_tabs.addTab(video_tab, "Video Recording")
        
        # Create a wrapper widget for the controls
        self.controls_widget = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout()
        self.controls_widget.setLayout(self.controls_layout)
        self.controls_layout.addWidget(self.control_tabs)
        
    def setup_timer(self):
        """Setup animation timer"""
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(8)  # ~120 FPS target (since we do less work per frame)
        
        # Autocorrelation update timer (less frequent)
        self.acf_timer = QtCore.QTimer()
        self.acf_timer.timeout.connect(self.update_autocorrelation)
        self.acf_timer.start(1000)  # Update every 1 second (slower)
        
    def update_animation(self):
        """Optimized animation update with variable frame rates"""
        if not self.is_paused:
            # PHYSICS: Update every frame for accuracy
            self.physics_counter += 1
            if self.physics_counter >= self.physics_rate:
                boundary_wrapped = self.sim.update_positions()
                
                # Clear trails for wrapped particles
                for i, wrapped in enumerate(boundary_wrapped):
                    if wrapped and i < len(self.trails):
                        self.trails[i].clear()
                
                # Calculate intensity (needed every frame for data accuracy)
                result = self.sim.calculate_intensity()
                if self.sim.two_particle_mode and self.sim.separate_traces:
                    # Handle separate traces for each particle type
                    intensity_total, intensity_1, intensity_2 = result
                    self.sim.intensity_trace.append(intensity_total)
                    self.sim.intensity_trace_1.append(intensity_1)
                    self.sim.intensity_trace_2.append(intensity_2)
                else:
                    # Single combined intensity
                    intensity = result
                    self.sim.intensity_trace.append(intensity)
                self.sim.time_trace.append(self.sim.current_time)
                self.sim.current_time += self.sim.time_step
                
                self.physics_counter = 0
            
            # 3D VISUALIZATION: Update less frequently
            self.visual_counter += 1
            if self.visual_counter >= self.visual_rate:
                pos = self.sim.positions * 1e6
                colors, sizes = self.sim.get_particle_colors_and_sizes()
                
                # Level-of-detail: Simplify for high particle counts
                if self.sim.num_particles > self.max_visual_particles:
                    # Show only subset of particles
                    subset_indices = np.linspace(0, self.sim.num_particles-1, self.max_visual_particles, dtype=int)
                    pos_subset = pos[subset_indices]
                    colors_subset = colors[subset_indices] 
                    sizes_subset = sizes[subset_indices]
                    self.particles.setData(pos=pos_subset, color=colors_subset, size=sizes_subset)
                else:
                    self.particles.setData(pos=pos, color=colors, size=sizes)
                
                self.visual_counter = 0
            
            # TRAILS: Update even less frequently
            self.trail_counter += 1
            if self.trail_counter >= self.trail_rate and self.trail_check.isChecked():
                pos = self.sim.positions * 1e6
                colors, sizes = self.sim.get_particle_colors_and_sizes()
                self.update_trails_optimized(pos, colors)
                self.trail_counter = 0
            
            # PLOTS: Update moderately
            self.plot_counter += 1
            if self.plot_counter >= self.plot_rate:
                if len(self.sim.time_trace) > 0:
                    times_ms = np.array(self.sim.time_trace) * 1000
                    
                    if self.sim.two_particle_mode and self.sim.separate_traces:
                        # Show separate traces for each particle type
                        intensities = np.array(self.sim.intensity_trace)
                        intensities_1 = np.array(self.sim.intensity_trace_1) if len(self.sim.intensity_trace_1) > 0 else []
                        intensities_2 = np.array(self.sim.intensity_trace_2) if len(self.sim.intensity_trace_2) > 0 else []
                        
                        self.intensity_curve.setData(times_ms, intensities)
                        if len(intensities_1) > 0:
                            self.intensity_curve_1.setData(times_ms[:len(intensities_1)], intensities_1)
                        if len(intensities_2) > 0:
                            self.intensity_curve_2.setData(times_ms[:len(intensities_2)], intensities_2)
                    else:
                        # Show only combined trace
                        intensities = np.array(self.sim.intensity_trace)
                        self.intensity_curve.setData(times_ms, intensities)
                
                self.plot_counter = 0
            
            # Update FPS every frame, status less frequently
            self.update_fps()
            
            # Update status display less frequently (every 30 frames ≈ 4x/sec)
            if hasattr(self, 'status_counter'):
                self.status_counter += 1
            else:
                self.status_counter = 0
                
            if self.status_counter >= 30:
                self.update_status()
                self.status_counter = 0
                
            # Capture video frame if recording
            if self.is_recording:
                self.capture_video_frame()
            
    def update_trails(self, positions, colors):
        """Legacy trail update function"""
        self.update_trails_optimized(positions, colors)
        
    def update_trails_optimized(self, positions, colors):
        """Optimized trail update with level-of-detail"""
        # Clear old trail lines
        for line in self.trail_lines:
            self.gl_widget.removeItem(line)
        self.trail_lines.clear()
        
        # Level-of-detail: Limit number of particles with trails
        max_trails = min(self.max_trail_particles, self.sim.num_particles, len(self.trails))
        
        # Add new positions to trails (only for first N particles)
        for i in range(max_trails):
            if i < len(self.trails):
                self.trails[i].append(positions[i].copy())
                
                if len(self.trails[i]) > 1:
                    trail_array = np.array(self.trails[i])
                    color = colors[i].copy()
                    color[3] = 0.3  # Make trails more transparent
                    
                    line = gl.GLLinePlotItem(pos=trail_array, color=color, width=1)
                    self.gl_widget.addItem(line)
                    self.trail_lines.append(line)
                    
    def update_autocorrelation(self):
        """Update autocorrelation plot (called less frequently)"""
        if len(self.sim.intensity_trace) > 50:
            tau, G = self.sim.calculate_autocorrelation_fft(self.sim.intensity_trace)
            
            if tau is not None and G is not None:
                tau_ms = tau * 1000
                # Filter out invalid values
                valid = np.isfinite(G) & (G > -1) & (G < 10)
                if np.any(valid):
                    self.autocorr_curve.setData(tau_ms[valid], G[valid])
                    
            # Handle separate traces if enabled
            if self.sim.two_particle_mode and self.sim.separate_traces:
                # Type 1 autocorrelation
                if len(self.sim.intensity_trace_1) > 50:
                    tau_1, G_1 = self.sim.calculate_autocorrelation_fft(self.sim.intensity_trace_1)
                    if tau_1 is not None and G_1 is not None:
                        tau_ms_1 = tau_1 * 1000
                        valid_1 = np.isfinite(G_1) & (G_1 > -1) & (G_1 < 10)
                        if np.any(valid_1):
                            self.autocorr_curve_1.setData(tau_ms_1[valid_1], G_1[valid_1])
                
                # Type 2 autocorrelation
                if len(self.sim.intensity_trace_2) > 50:
                    tau_2, G_2 = self.sim.calculate_autocorrelation_fft(self.sim.intensity_trace_2)
                    if tau_2 is not None and G_2 is not None:
                        tau_ms_2 = tau_2 * 1000
                        valid_2 = np.isfinite(G_2) & (G_2 > -1) & (G_2 < 10)
                        if np.any(valid_2):
                            self.autocorr_curve_2.setData(tau_ms_2[valid_2], G_2[valid_2])
                    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_time > 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_time)
            self.fps_label.setText(f'FPS: {self.current_fps:.1f}')
            self.fps_counter = 0
            self.fps_time = current_time
            
    def update_status(self):
        """Update status bar"""
        status = f"Particles: {self.sim.num_particles} | "
        status += f"D = {self.sim.D*1e12:.2f} μm²/s | "
        status += f"τD = {(self.sim.w_xy**2 / (4 * self.sim.D))*1000:.3f} ms | "
        status += f"Time: {self.sim.current_time*1000:.1f} ms"
        self.status_label.setText(status)
        
    def update_concentration(self, value):
        """Update concentration"""
        self.sim.concentration_nM = value
        self.sim.concentration_nM_1 = value
        old_num = self.sim.num_particles
        self.sim.update_particle_count_from_concentration()
        
        if self.sim.num_particles != old_num:
            self.sim.init_particles()
            self.trails = [deque(maxlen=self.trail_length) for _ in range(self.sim.num_particles)]
            
    def update_box_size(self, value):
        """Update box size"""
        self.sim.box_size_factor = value
        self.sim.box_size = self.sim.box_size_factor * self.sim.w_z
        self.sim.update_particle_count_from_concentration()
        self.sim.init_particles()
        self.trails = [deque(maxlen=self.trail_length) for _ in range(self.sim.num_particles)]
        
        # Update camera distance
        self.gl_widget.setCameraPosition(distance=self.sim.box_size*4e6)
        
    def update_radius(self, value):
        """Update particle radius"""
        self.sim.hydrodynamic_radius = value * 1e-9
        self.sim.hydrodynamic_radius_1 = value * 1e-9
        self.sim.update_diffusion_coefficient()
        
    def update_timestep(self, value):
        """Update time step"""
        self.sim.time_step = value * 1e-6
        self.sim.update_diffusion_coefficient()
        
    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        self.pause_btn.setText('Resume' if self.is_paused else 'Pause')
        
    def update_trail_skip(self, value):
        """Update trail update rate (how often trails are rendered)"""
        self.trail_update_rate = value
        self.trail_update_counter = 0  # Reset counter
        
    def update_performance_level(self, level):
        """Update performance optimization level"""
        if level == "High Quality":
            # Higher quality, lower performance
            self.visual_rate = 1      # Update every frame
            self.trail_rate = 2       # Update every 2 frames
            self.plot_rate = 2        # Update every 2 frames
            self.acf_rate = 100       # Update every 100 frames
            self.max_visual_particles = 500  # Show more particles
        elif level == "Max Speed":
            # Maximum performance, lower quality
            self.visual_rate = 5      # Update every 5 frames
            self.trail_rate = 10      # Update every 10 frames
            self.plot_rate = 5        # Update every 5 frames
            self.acf_rate = 500       # Update every 500 frames
            self.max_visual_particles = 100  # Show fewer particles
        else:  # Balanced
            # Default balanced settings
            self.visual_rate = 2
            self.trail_rate = 5
            self.plot_rate = 3
            self.acf_rate = 200
            self.max_visual_particles = 200
        
        # Reset counters
        self.visual_counter = 0
        self.trail_counter = 0
        self.plot_counter = 0
        self.acf_counter = 0
        
        print(f"Performance level set to: {level}")
        
    def update_max_trails(self, value):
        """Update maximum number of particles with trails"""
        self.max_trail_particles = value
        # Resize trails list if needed
        if len(self.trails) > value:
            self.trails = self.trails[:value]
        elif len(self.trails) < value and len(self.trails) < self.sim.num_particles:
            while len(self.trails) < min(value, self.sim.num_particles):
                self.trails.append(deque(maxlen=self.trail_length))
                
    def update_video_quality(self, quality):
        """Update video quality setting"""
        self.video_quality = quality
        
    def update_video_fps(self, fps):
        """Update video FPS setting"""
        self.video_fps = fps
        
    def toggle_recording(self):
        """Start or stop video recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start video recording"""
        try:
            # Create videos directory if it doesn't exist
            if not os.path.exists('fcs_videos'):
                os.makedirs('fcs_videos')
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = f'fcs_videos/fcs_simulation_{timestamp}.avi'
            
            # Get widget size for video dimensions
            widget_size = self.size()
            width = widget_size.width()
            height = widget_size.height()
            
            # Ensure dimensions are even (required by some codecs)
            if width % 2 != 0:
                width -= 1
            if height % 2 != 0:
                height -= 1
            
            # Set up video writer with quality settings
            if self.video_quality == 'Low':
                bitrate = 1000000  # 1 Mbps
            elif self.video_quality == 'Medium':
                bitrate = 3000000  # 3 Mbps
            else:  # High
                bitrate = 8000000  # 8 Mbps
            
            # Use XVID codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            self.video_writer = cv2.VideoWriter(
                self.video_filename,
                fourcc,
                float(self.video_fps),
                (width, height)
            )
            
            if not self.video_writer.isOpened():
                raise Exception("Failed to open video writer")
            
            self.is_recording = True
            self.video_frame_count = 0
            
            # Update UI
            self.record_btn.setText('⏹ Stop Recording')
            self.record_btn.setStyleSheet(
                "QPushButton { background-color: #44ff44; color: black; font-weight: bold; }"
            )
            self.recording_status.setText('Recording...')
            self.recording_status.setStyleSheet("color: red; font-weight: bold;")
            
            print(f"Started recording to: {self.video_filename}")
            print(f"Video settings: {width}x{height} @ {self.video_fps} FPS, Quality: {self.video_quality}")
            
        except Exception as e:
            print(f"Error starting video recording: {e}")
            self.recording_status.setText(f'Error: {str(e)}')
            self.recording_status.setStyleSheet("color: red; font-weight: bold;")
    
    def stop_recording(self):
        """Stop video recording"""
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
            self.is_recording = False
            
            # Update UI
            self.record_btn.setText('● Start Recording')
            self.record_btn.setStyleSheet(
                "QPushButton { background-color: #ff4444; color: white; font-weight: bold; }"
            )
            self.recording_status.setText(f'Saved: {self.video_frame_count} frames')
            self.recording_status.setStyleSheet("color: green; font-weight: bold;")
            
            print(f"Recording stopped. Saved {self.video_frame_count} frames to: {self.video_filename}")
            
            # Show completion message
            QtWidgets.QMessageBox.information(
                self, 
                "Recording Complete",
                f"Video saved successfully!\n\nFile: {self.video_filename}\nFrames: {self.video_frame_count}\nDuration: {self.video_frame_count/self.video_fps:.1f} seconds"
            )
    
    def capture_video_frame(self):
        """Capture current frame for video recording"""
        if not self.is_recording or not self.video_writer:
            return
            
        try:
            # Capture the entire widget as a pixmap
            pixmap = self.grab()
            
            # Convert QPixmap to QImage
            qimg = pixmap.toImage()
            
            # Convert QImage to numpy array
            width = qimg.width()
            height = qimg.height()
            
            # Convert to RGB format (compatible with different PyQt versions)
            try:
                qimg = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
            except AttributeError:
                # Fallback for older PyQt versions
                qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB888)
            
            # Create numpy array from QImage data
            ptr = qimg.bits()
            ptr.setsize(height * width * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            
            # OpenCV uses BGR format, so convert RGB to BGR
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
            # Write frame to video
            self.video_writer.write(frame)
            self.video_frame_count += 1
            
            # Update status periodically
            if self.video_frame_count % 30 == 0:  # Every 30 frames
                self.recording_status.setText(f'Recording... {self.video_frame_count} frames')
                
        except Exception as e:
            print(f"Error capturing video frame: {e}")
            self.stop_recording()
        
    def add_2d_plane(self):
        """Add a semi-transparent plane at z=0 to highlight 2D simulation surface"""
        # Create a flat plane at z=0
        box = self.sim.box_size * 1e6  # Convert to micrometers
        vertices = np.array([
            [-box, -box, 0],
            [box, -box, 0],
            [box, box, 0],
            [-box, box, 0],
        ])
        
        # Define faces (2 triangles to make a square)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        
        # Create mesh item with semi-transparent blue color
        meshdata = gl.MeshData(vertexes=vertices, faces=faces)
        self.plane_2d = gl.GLMeshItem(
            meshdata=meshdata,
            color=(0.3, 0.5, 0.8, 0.2),  # Light blue, semi-transparent
            smooth=True,
            glOptions='translucent'
        )
        self.gl_widget.addItem(self.plane_2d)
        self.plane_2d.setVisible(False)  # Initially hidden
        
    def update_grid_visibility(self):
        """Update grid visibility and prominence based on simulation mode"""
        if hasattr(self, 'plane_2d'):
            # Show/hide 2D plane based on mode
            self.plane_2d.setVisible(self.sim.is_2d_mode)

    def toggle_2d_mode(self, state):
        """Toggle between 2D and 3D simulation modes"""
        self.sim.is_2d_mode = state == 2  # Qt.Checked = 2
        
        # Update grid visibility for new mode
        self.update_grid_visibility()
        
        # Reset simulation with new mode
        self.reset_simulation()
        
        # Update status
        mode_text = "2D" if self.sim.is_2d_mode else "3D"
        print(f"Switched to {mode_text} simulation mode")
        
    def reset_simulation(self):
        """Reset the simulation"""
        self.sim.init_particles()
        self.sim.intensity_trace.clear()
        self.sim.time_trace.clear()
        self.sim.intensity_trace_1.clear()
        self.sim.intensity_trace_2.clear()
        self.sim.current_time = 0
        
        # Update trails for new particle count
        self.trails = [deque(maxlen=self.trail_length) for _ in range(self.sim.num_particles)]
        
        # Clear old trail lines
        for line in self.trail_lines:
            self.gl_widget.removeItem(line)
        self.trail_lines.clear()
        
        # Clear plots
        self.intensity_curve.setData([], [])
        self.intensity_curve_1.setData([], [])
        self.intensity_curve_2.setData([], [])
        self.autocorr_curve.setData([], [])
        self.autocorr_curve_1.setData([], [])
        self.autocorr_curve_2.setData([], [])
        
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        event.accept()
        
    def toggle_two_particle_mode(self, state):
        """Toggle two particle mode"""
        self.sim.two_particle_mode = state == 2  # Qt.Checked = 2
        self.sim.update_particle_count_from_concentration()
        self.reset_simulation()
        
        # Enable/disable type 2 controls
        enabled = self.sim.two_particle_mode
        self.conc2_spin.setEnabled(enabled)
        self.radius2_spin.setEnabled(enabled)
        self.brightness2_spin.setEnabled(enabled)
        self.separate_traces_check.setEnabled(enabled)
        
        # Update curve visibility based on mode and separate traces setting
        if enabled and self.sim.separate_traces:
            self.intensity_curve_1.show()
            self.intensity_curve_2.show()
            self.autocorr_curve_1.show()
            self.autocorr_curve_2.show()
            self.show_legends()
        else:
            self.intensity_curve_1.hide()
            self.intensity_curve_2.hide()
            self.autocorr_curve_1.hide()
            self.autocorr_curve_2.hide()
            self.hide_legends()
        
        print(f"Two-particle mode: {'Enabled' if enabled else 'Disabled'}")
        if enabled:
            print(f"Type 1: {self.sim.num_particles_1} particles, Type 2: {self.sim.num_particles_2} particles")
    
    def show_legends(self):
        """Show plot legends"""
        if self.intensity_legend is not None:
            self.intensity_legend.setVisible(True)
        if self.autocorr_legend is not None:
            self.autocorr_legend.setVisible(True)
        
    def hide_legends(self):
        """Hide plot legends"""
        if self.intensity_legend is not None:
            self.intensity_legend.setVisible(False)
        if self.autocorr_legend is not None:
            self.autocorr_legend.setVisible(False)
    
    def toggle_separate_traces(self, state):
        """Toggle separate intensity traces"""
        self.sim.separate_traces = state == 2
        
        # Show/hide separate trace curves and legends
        if self.sim.separate_traces and self.sim.two_particle_mode:
            self.intensity_curve_1.show()
            self.intensity_curve_2.show()
            self.autocorr_curve_1.show()
            self.autocorr_curve_2.show()
            # Show legends
            self.show_legends()
        else:
            self.intensity_curve_1.hide()
            self.intensity_curve_2.hide()
            self.autocorr_curve_1.hide()
            self.autocorr_curve_2.hide()
            # Hide legends
            self.hide_legends()
            
        print(f"Separate traces: {'Enabled' if self.sim.separate_traces else 'Disabled'}")
    
    def update_concentration_2(self, value):
        """Update type 2 particle concentration"""
        self.sim.concentration_nM_2 = value
        old_num = self.sim.num_particles
        self.sim.update_particle_count_from_concentration()
        
        if self.sim.num_particles != old_num:
            self.reset_simulation()
    
    def update_radius_2(self, value):
        """Update type 2 particle radius"""
        self.sim.hydrodynamic_radius_2 = value * 1e-9
        self.sim.update_diffusion_coefficient()
    
    def update_brightness_2(self, value):
        """Update type 2 particle brightness"""
        self.sim.brightness_2 = value

def main():
    print("Starting FCS PyQtGraph Simulation (Performance Optimized)...")
    print("=" * 65)
    print("Performance Optimizations:")
    print("- Variable frame rates: Physics (1x), Visual (2x), Trails (5x)")
    print("- Level-of-detail: Particle/trail count limits")
    print("- Performance presets: Balanced/High Quality/Max Speed")
    print("- Simplified wireframe confocal volume")
    print("- PyQtGraph OpenGL acceleration (10-20x faster)")
    print("- Vectorized particle physics calculations") 
    print("- Optimized trail and plot updates")
    print("=" * 65)
    print("Controls:")
    print("- Performance dropdown: Choose quality vs speed")
    print("- Max Trails: Limit number of particle trails")
    print("- Video Recording: Capture full GUI as video")
    print("- All standard FCS parameters available")
    print("=" * 65)
    print("Video Recording:")
    print("- Click 'Start Recording' to begin video capture")
    print("- Choose quality: Low/Medium/High")
    print("- Set FPS: 15-60 frames per second")
    print("- Videos saved to 'fcs_videos/' folder")
    print("- Captures entire GUI: 3D view + plots + controls")
    print("=" * 65)
    
    app = QtWidgets.QApplication(sys.argv)
    # use qdarkstyle
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='PyQt6'))
    except ImportError:
        pass

    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = FCSVisualizationPyQtGraph()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()