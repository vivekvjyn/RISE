import argparse

from pitch import Logger
from pitch import varnam_svaras, varnam_svara_forms, cmmr_plausible_svaras

def main():
    args = parse_args()
    logger = Logger()

    varnam_svaras(args.smoothing_factor, args.interpolation_gap, logger)
    varnam_svara_forms(args.smoothing_factor, args.interpolation_gap, logger)
    cmmr_plausible_svaras(args.smoothing_factor, args.interpolation_gap, logger)

def parse_args():
    parser = argparse.ArgumentParser(description="svara representation learning for carnatic music transcription")
    parser.add_argument('--smoothing-factor', type=float, default=0.5, help='Smoothing factor for pitch curve smoothing')
    parser.add_argument('--interpolation-gap', type=float, default=0.02, help='Maximum gap duration for interpolation in seconds')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
