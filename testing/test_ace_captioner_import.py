import importlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    importlib.import_module("extensions_built_in.captioner.AceStepCaptioner")


if __name__ == "__main__":
    main()
