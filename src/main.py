import argparse
from data_loader import load_data
from analysis import analyze
from visualization import generate_plots
from report import generate_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to CSV input data')
    parser.add_argument('--output', required=True, help='Path to save report PDF')
    args = parser.parse_args()

    df = load_data(args.input)
    results = analyze(df)
    figs = generate_plots(df, results)
    generate_report(df, results, figs, args.output)

if __name__ == '__main__':
    main()
