# Enhanced ABR Plotter

This script creates improved visualizations from pre-trained Bayesian ABR models without needing to re-run the stochastic analysis.

## Features

### 1. **Dual Format Output**
- **PNG Format**: 1200 DPI for high-resolution presentations and printing
- **EPS Format**: Vector format for publication-quality scaling
- Creates separate high-quality plots instead of subplots

### 2. **Enhanced ABR Profiles Plot**
- **Modern Colors**: Dark grey controls with light grey confidence intervals, warm orange mutants
- **Y-axis**: Standardized to -10 to 100 dB SPL with 10 dB markers
- **Legends**: Moved outside plot area for better presentation
- **Professional Styling**: Increased line weights and modern color palette

### 3. **Improved Posterior Distribution Plot**
- **HDI Visualization**: Vertical lines for mean and 95% HDI bounds with legend
- **Modern Colors**: Light blue histogram, dark grey mean line, grey HDI bounds
- **Clean Layout**: Histogram with proper density scaling and enhanced contrast

### 4. **Enhanced Effect Size Plot**
- **Flipped Axes**: Frequency on x-axis, effect size on y-axis (more intuitive)
- **Frequency Labels**: Clear 6, 12, 18, 24, 30 kHz labels (not hl_shift[0], [1], etc.)
- **Modern Styling**: Nord blue data points with enhanced error bars
- **Reference Line**: Horizontal line at zero effect with improved visibility

### 5. **Modern Nord Color Theme**
- **Inspired by Nord Theme**: Professional, accessible color palette
- **Blue Tones**: Nord blue for controls and primary data points
- **Orange Accents**: Warm orange for mutants and mean values
- **Green Highlights**: Sage green for HDI bounds and secondary elements
- **Grey Scales**: Refined grey tones for backgrounds and confidence intervals

### 6. **Comprehensive Titles**
- **Format**: Gene | Allele | Zygosity | Center | Gender
- **Example**: "Marveld2 | Marveld2<tm1b(EUCOMM)Wtsi> | homozygote | MRC Harwell | All"
- **Gender Detection**: Automatically detects Male/Female splits from directory names

## Usage

### Command Line
```bash
# Basic usage - creates plots in model_directory/enhanced_plots/
python scripts/enhanced_abr_plotter.py "path/to/model/directory"

# Specify custom output directory
python scripts/enhanced_abr_plotter.py "path/to/model/directory" --output-dir "custom/output/path"
```

### Example
```bash
python scripts/enhanced_abr_plotter.py "results/test_bayes/Marveld2_analysis_20250603_162914/Marveld2_Marveld2<tm1b(EUCOMM)Wtsi>_homozygote_MRC Harwell"
```

### Python API
```python
from enhanced_abr_plotter import EnhancedABRPlotter

# Initialize plotter
plotter = EnhancedABRPlotter("path/to/model/directory")
plotter.load_model()

# Create individual plots
plotter.plot_abr_profiles("abr_profiles.png")
plotter.plot_posterior_distribution("posterior.png") 
plotter.plot_effect_size("effect_size.png")

# Or create all plots at once
plotter.create_all_plots("output/directory")
```

## Input Requirements

The script expects a model directory containing:
- `trace.nc`: NetCDF file with MCMC samples and observed data
- `model_spec.json`: JSON file with model specifications and summary statistics

## Output Files

Each plot is saved in both formats:
- `abr_profiles_enhanced.png/.eps`: ABR threshold profiles comparison
- `posterior_distribution_enhanced.png/.eps`: Posterior probability of hearing loss  
- `effect_size_enhanced.png/.eps`: Frequency-specific hearing loss effects

**PNG files** (1200 DPI): High-resolution for presentations, printing, and detailed viewing
**EPS files** (vector): Perfect for publication-quality printing and infinite scaling

## Key Advantages

1. **Reproducible**: Uses saved model traces, avoiding stochastic variation
2. **High Quality**: 300 DPI publication-ready figures
3. **Customizable**: Easy to modify colors, layouts, and styling
4. **Fast**: No model refitting required
5. **Batch Processing**: Can process multiple models efficiently

## Dependencies

- `arviz`: Bayesian analysis visualization
- `matplotlib`: Plotting library
- `numpy`: Numerical computations
- Standard library: `json`, `pathlib`, `argparse`