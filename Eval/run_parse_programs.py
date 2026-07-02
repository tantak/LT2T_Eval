"""
Wrapper script to run parse_programs.py with correct path setup.

This ensures that local logic_parser.py is found before the built-in parser module.
Note: parser.py was renamed to logic_parser.py to avoid conflict with Python's built-in parser module.
"""
import sys
import os
from pathlib import Path

# Get the LogicNLG directory (parent of this script's directory, or from env)
if len(sys.argv) > 1 and '--logicnlg_dir' in sys.argv:
    idx = sys.argv.index('--logicnlg_dir')
    logicnlg_dir = Path(sys.argv[idx + 1]).resolve()
    # Remove these args before passing to parse_programs
    sys.argv.pop(idx)
    sys.argv.pop(idx)
else:
    # Assume we're in the LogicNLG directory
    logicnlg_dir = Path(__file__).parent.parent / "LogicNLG-master"
    logicnlg_dir = logicnlg_dir.resolve()

# Ensure LogicNLG directory is FIRST in sys.path (before built-in modules)
if str(logicnlg_dir) not in sys.path:
    sys.path.insert(0, str(logicnlg_dir))

# Now import and run parse_programs
if __name__ == "__main__":
    # Change to LogicNLG directory
    os.chdir(str(logicnlg_dir))
    
    # Import parse_programs and run its main
    import parse_programs
    parse_programs.main()
