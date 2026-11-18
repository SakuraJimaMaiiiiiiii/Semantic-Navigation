"""Start graph-node exploration from the project root.

Usage: python run_wildos_explore.py --sam3
Implementation and helpers live in exploration_wildos/.
"""
import runpy


def main():
    # Execute the module's CLI so --help is handled before flight SDK imports.
    runpy.run_module("exploration_wildos.explore", run_name="__main__")


if __name__ == "__main__":
    main()
