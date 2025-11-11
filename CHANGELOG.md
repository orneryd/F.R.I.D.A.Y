# Changelog - FRIDAY AI Improvements

## Version 2.3 - Complete Overhaul (November 2024)

### CLI Improvements
- ✅ **Rich CLI**: Beautiful colored output with Rich library
- ✅ **Tables**: Statistics displayed in formatted tables
- ✅ **Icons**: Success (✓), Error (✗), Info (ℹ) indicators
- ✅ **Headers**: Styled panels for each command
- ✅ **Custom Help**: Improved --help with categories and examples
- ✅ **Fallback**: Works without Rich (plain text mode)
- ✅ **Better UX**: Clearer messages and formatting
- ✅ **Experimental Notice**: Clear indication of features in development

## Version 2.3 - Project Cleanup & Three.js Integration (November 2024)

### Major Changes
- ✅ **Three.js Viewer Integration**: High-performance 3D viewer now integrated into main system
  - New command: `python cli.py view`
  - Handles 100K+ synapses smoothly with WebGL
  - Interactive controls, real-time filtering
  - Click neurons for detailed information
  
- ✅ **Enhanced UI with Sidebar**: Professional interface with collapsible sidebar
  - Left sidebar with system logs (toggle with ☰ button)
  - Real-time logging of all operations
  - Clean, terminal-style formatting
  - Color-coded log levels (info, warning, error)
  - Neuron type statistics
  - Synapse weight distribution
  - No emojis, professional appearance

- ✅ **Integrated Training Interface**: Train directly in the browser
  - Training panel with dataset and manual modes
  - Dataset selection dropdown
  - Configurable sample count (100-10,000)
  - Manual knowledge input with tags
  - **Real-time Live Updates**: Watch neurons appear during training
  - WebSocket-based live visualization
  - Progress updates every 10 samples
  - Neurons appear in 3D graph in real-time
  - Training progress in logs and status bar
  - Automatic brain reload after training
  - REST API for training operations
  - No command line needed

- ✅ **HuggingFace Integration**: Access thousands of datasets
  - Token-based authentication
  - Login directly in the browser
  - **Search Bar**: Search any HuggingFace dataset in real-time
  - Shows downloads and likes for each dataset
  - Popular datasets pre-selected (SQuAD, WikiText, ELI5, etc.)
  - Streaming support for large datasets
  - Automatic dataset structure detection
  - Three training modes: Local, HuggingFace, Manual
  - Visual login status indicator
  - Secure token handling
  - Click-to-select from search results
  
- ✅ **Project Structure Cleanup**: Moved all utility scripts to `scripts/` directory
  - Root directory now clean with only main files
  - All test, demo, and utility scripts organized in `scripts/`
  - Added `scripts/README.md` with documentation
  
- ✅ **Integrated Visualization System**: 
  - `neuron_system/visualization/threejs_viewer.py` - Main viewer class
  - `neuron_system/visualization/templates/brain_viewer_threejs.html` - WebGL interface
  - Properly integrated into neuron_system package

### File Organization
**Root (Clean):**
- `cli.py` - Main CLI interface
- `train.py` - Training system
- `README.md`, `CLI.md`, `FEATURES.md`, `CHANGELOG.md` - Documentation
- `requirements.txt` - Dependencies

**Scripts Directory:**
- Visualization scripts (visualize.py, view_brain_*.py, live_viewer.py)
- Setup/Reset scripts (setup_model.py, reset_brain.py, fix_neuron_positions.py)
- Extract/Analyze scripts (extract_*.py, analyze_*.py)
- Demo/Test scripts (demo_*.py, test_*.py, quick_*.py)
- Legacy scripts (assimilate.py, train_fresh.py, check_*.py)

### Documentation Updates
- Updated README.md with 3D viewer usage
- Updated CLI.md with `view` command documentation
- Added scripts/README.md for utility scripts

## Version 2.2 - Live 3D Brain Viewer - Complete Overhaul

### Visualization Fixes
- ✅ Fixed neuron rendering (were invisible due to axis range issues)
- ✅ Centered brain visualization automatically
- ✅ Improved neuron visibility with Plasma colorscale
- ✅ Made synapses thinner and more transparent
- ✅ Fixed alignment between neurons and synapses
- ✅ Sidebar now hidden by default, shows on neuron click
- ✅ Reduced colorbar size (10px width, 30% height)

### Technical Improvements
- Fixed axis range to include negative coordinates (-120 to 120)
- Neurons and synapses now use same centered coordinate system
- Click handler properly attached after first render
- Initial data loaded immediately on connect

## Training Performance - Batch Processing

### New Features
- ✅ Added `learn_batch()` method to LanguageModel
- ✅ TRUE batch training - all neurons created at once
- ✅ Single database save instead of per-neuron saves
- ✅ Significantly faster training (10x+ speedup)

### Implementation
- `neuron_system/ai/language_model.py`: Added `learn_batch()` method
- `neuron_system/training/knowledge_extractor.py`: Uses batch training
- Fallback to individual training if batch fails

## Brain Layout Algorithms

### New File: `neuron_system/visualization/brain_layout.py`
- `generate_spherical_position()`: Fibonacci spiral on sphere
- `generate_brain_shaped_position()`: Brain-like ellipsoid
- `generate_clustered_position()`: Semantic clustering
- `recenter_positions()`: Center positions at origin

## Utility Scripts

### Kept
- `reset_brain.py`: Delete all neurons and synapses
- `train_fresh.py`: Train with Hebbian learning (no synapses)
- `fix_neuron_positions.py`: Assign positions to neurons
- `quick_start.py`: Automated workflow
- `demo_hebbian.py`: Demonstrate Hebbian learning

### Removed (Debug only)
- ~~`check_neuron_data.py`~~ (debug)
- ~~`debug_viewer_data.py`~~ (debug)

## Model Loading Improvements

### Local Model Manager
- Fixed model path detection for cached models
- Support for HuggingFace cache structure
- Automatic fallback to multiple paths
- Changed `torch_dtype` to `dtype` (deprecated warning fix)

## Documentation

### Updated
- `FRESH_START.md`: Complete Hebbian learning workflow
- `HEBBIAN_LEARNING.md`: Hebbian learning concepts
- `LIVE_VIEWER.md`: Live viewer usage

## Breaking Changes
None - all changes are backwards compatible

## Migration Guide
No migration needed. Existing databases work as-is.

## Performance Improvements
- Training: 10x+ faster with batch processing
- Visualization: Smooth rendering with optimized traces
- Database: Single save per batch instead of per-neuron

## Bug Fixes
- Fixed neurons not visible in 3D viewer
- Fixed misalignment between neurons and synapses
- Fixed sidebar always visible
- Fixed colorbar taking too much space
- Fixed axis range not including negative coordinates

## Next Steps
- Implement Hebbian learning during query processing
- Add synapse pruning for weak connections
- Improve clustering visualization
- Add animation for neuron activation
