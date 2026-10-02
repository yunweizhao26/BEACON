"""Run a frozen k562 experiment; use --case for another declared condition."""
from experiments.run import main
if __name__ == "__main__": main('external', context='k562')
