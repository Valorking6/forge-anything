import os
import sys
import importlib.util

# Add the extension directory to the Python path
extension_dir = os.path.dirname(os.path.abspath(__file__))
if extension_dir not in sys.path:
    sys.path.append(extension_dir)

# Run the install script first to ensure all dependencies are installed
try:
    import install
except ImportError:
    pass

# Import the scripts module to ensure it's loaded
scripts_dir = os.path.join(extension_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Import the sam_tab module to register the UI tab
try:
    import scripts.sam_tab
except ImportError as e:
    print(f"Error importing sam_tab module: {e}")
