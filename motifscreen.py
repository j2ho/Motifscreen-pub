#!/usr/bin/env python
"""MotifScreen-Aff CLI entry point.

Usage:
    uv run python motifscreen.py prepare --protein receptor.pdb --ligands compounds.sdf --center 12.5,34.2,8.7 --output prepared/
    uv run python motifscreen.py predict --datapath prepared/ --checkpoint models/best.pkl --base-config configs/training/endtoend.yaml --output scores.csv
"""

import sys

COMMANDS = {
    'prepare': 'src.cli.prepare',
    'predict': 'src.cli.predict',
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: motifscreen.py <command> [options]")
        print()
        print("Commands:")
        print("  prepare   Prepare protein + ligands for inference (PDB + SDF/mol2 -> npz files)")
        print("  predict   Run inference on prepared data (npz files + checkpoint -> scores)")
        print()
        print("Run 'motifscreen.py <command> --help' for command-specific options.")
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(COMMANDS)}")
        sys.exit(1)

    # Remove the command from argv so argparse in submodules sees the right args
    sys.argv = [f"motifscreen {command}"] + sys.argv[2:]

    import importlib
    module = importlib.import_module(COMMANDS[command])
    module.main()


if __name__ == '__main__':
    main()
