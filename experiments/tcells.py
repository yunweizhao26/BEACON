"""Run a frozen tcells experiment; use --case for another declared condition."""
from experiments.run import main
if __name__ == "__main__": main('external', context='tcell_resting')
