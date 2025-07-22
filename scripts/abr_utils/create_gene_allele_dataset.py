#!/usr/bin/env python3
"""
Create gene or allele-specific ABR datasets for targeted analysis.

This script filters the full ABR dataset to create smaller datasets containing:
- Specific genes (all alleles) or specific alleles
- All control mice from the dataset

The output can be directly used with run_parallel_analysis.py for Bayesian analysis.

Usage examples:
    # Single gene
    python create_gene_allele_dataset.py --mode gene --genes Mettl5 --output data/processed/abr_mettl5_gene.csv

    # Multiple genes
    python create_gene_allele_dataset.py --mode gene --genes Mettl5,Pank2,Adgrb1 --output data/processed/abr_selected_genes.csv

    # Single allele
    python create_gene_allele_dataset.py --mode allele --alleles "Mettl5<em1(IMPC)Hmgu>" --output data/processed/abr_mettl5_allele.csv

    # From file
    python create_gene_allele_dataset.py --mode gene --input-file gene_list.txt --output data/processed/abr_gene_set.csv

For full run of datasubset, Bayes and enhanced plots:

```bash
#####################################

EXAMPLE WITH ASF1B GENE

#####################################

# Create gene/allele dataset
python scripts/abr_utils/create_gene_allele_dataset.py --mode gene --genes Asf1b --output data/processed/Asf1b_with_control.csv

# Run Bayesian analysis on subset
python run_parallel_analysis.py --data data/processed/Asf1b_with_control.csv --output results/example_bayes_asf1b

# Create enhanced plots
python scripts/abr_utils/enhanced_abr_plotter.py "path/to/results/model_dir" --output-dir "path/to/results/visuals"
```

Author: Liam Barrett
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def load_input_list(file_path):
    """Load a list of genes or alleles from a text file (one per line)."""
    items = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                items.append(line)
    return items


def create_dataset(mode, items, output_path, data_path=None):
    """
    Create a filtered dataset based on specified genes or alleles.

    Parameters:
    -----------
    mode : str
        Either 'gene' or 'allele' to specify filtering mode
    items : list
        List of gene symbols or allele symbols to filter for
    output_path : str or Path
        Path to save the output CSV file
    data_path : str or Path, optional
        Path to the full ABR data file (defaults to standard location)

    Returns:
    --------
    pd.DataFrame
        The filtered dataset
    """
    # Set default data path if not provided
    if data_path is None:
        data_path = Path('/home/liamb/impc-abr/data/processed/abr_full_data.csv')
    else:
        data_path = Path(data_path)

    # Load the full dataset
    print(f"Loading ABR data from {data_path}...")
    try:
        abr_data = pd.read_csv(data_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Could not find data file at {data_path}")
        sys.exit(1)

    print(f"Loaded {len(abr_data):,} total records")

    # Filter column based on mode
    filter_column = 'gene_symbol' if mode == 'gene' else 'allele_symbol'

    # Check which items are available in the dataset
    available_items = set(abr_data[filter_column].dropna().unique())
    requested_items = set(items)
    found_items = requested_items.intersection(available_items)
    missing_items = requested_items - available_items

    if missing_items:
        print(f"\nWarning: The following {mode}s were not found in the dataset:")
        for item in sorted(missing_items):
            print(f"  - {item}")

    if not found_items:
        print(f"\nError: None of the specified {mode}s were found in the dataset!")
        sys.exit(1)

    print(f"\nFiltering for {len(found_items)} {mode}(s):")
    for item in sorted(found_items):
        print(f"  - {item}")

    # Filter for experimental mice with specified genes/alleles
    experimental_data = abr_data[
        (abr_data[filter_column].isin(found_items)) &
        (abr_data['biological_sample_group'] == 'experimental')
    ].copy()

    # Get all control mice
    control_data = abr_data[
        abr_data['biological_sample_group'] == 'control'
    ].copy()

    print(f"\nFound {len(experimental_data):,} experimental records")
    print(f"Found {len(control_data):,} control records")

    # Show breakdown by item
    print(f"\nExperimental records by {mode}:")
    item_counts = experimental_data[filter_column].value_counts()
    for item in sorted(found_items):
        count = item_counts.get(item, 0)
        print(f"  {item}: {count:,}")

    # If in gene mode, show allele breakdown
    if mode == 'gene':
        print("\nAllele breakdown:")
        for gene in sorted(found_items):
            gene_data = experimental_data[experimental_data['gene_symbol'] == gene]
            allele_counts = gene_data['allele_symbol'].value_counts()
            print(f"\n  {gene}:")
            for allele, count in allele_counts.items():
                print(f"    {allele}: {count}")

    # Combine experimental and control data
    combined_data = pd.concat([experimental_data, control_data], ignore_index=True)

    # Sort by biological_sample_group to have controls first, then by gene/allele
    combined_data = combined_data.sort_values(
        by=['biological_sample_group', filter_column],
        ascending=[True, True]
    )

    # Save the combined dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_data.to_csv(output_path, index=False)

    print(f"\nSaved combined dataset to: {output_path}")
    print(f"Total records: {len(combined_data):,}")

    # Show sex breakdown
    print("\nSex breakdown:")
    sex_exp = experimental_data['sex'].value_counts()
    sex_ctrl = control_data['sex'].value_counts()
    print(f"  Experimental - Male: {sex_exp.get('male', 0):,}, Female: {sex_exp.get('female', 0):,}")
    print(f"  Control - Male: {sex_ctrl.get('male', 0):,}, Female: {sex_ctrl.get('female', 0):,}")

    # Show center breakdown for experimental data
    print("\nExperimental records by center:")
    center_counts = experimental_data['phenotyping_center'].value_counts()
    for center, count in center_counts.head(10).items():
        print(f"  {center}: {count:,}")
    if len(center_counts) > 10:
        print(f"  ... and {len(center_counts) - 10} more centers")

    return combined_data


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Create gene or allele-specific ABR datasets for targeted analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single gene
  %(prog)s --mode gene --genes Mettl5 --output data/processed/abr_mettl5_gene.csv

  # Multiple genes
  %(prog)s --mode gene --genes Mettl5,Pank2,Adgrb1 --output data/processed/abr_selected_genes.csv

  # Single allele
  %(prog)s --mode allele --alleles "Mettl5<em1(IMPC)Hmgu>" --output data/processed/abr_mettl5_allele.csv

  # From file
  %(prog)s --mode gene --input-file gene_list.txt --output data/processed/abr_gene_set.csv
        """
    )

    parser.add_argument('--mode', required=True, choices=['gene', 'allele'],
                        help='Filter mode: gene (all alleles) or allele (specific alleles)')

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--genes', type=str,
                            help='Comma-separated list of gene symbols')
    input_group.add_argument('--alleles', type=str,
                            help='Comma-separated list of allele symbols')
    input_group.add_argument('--input-file', type=str,
                            help='Text file with gene/allele symbols (one per line)')

    parser.add_argument('--output', required=True, type=str,
                        help='Output CSV file path')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to full ABR data file (default: data/processed/abr_full_data.csv)')

    args = parser.parse_args()

    # Validate mode matches input type
    if args.mode == 'gene' and args.alleles:
        parser.error("Mode 'gene' requires --genes or --input-file, not --alleles")
    elif args.mode == 'allele' and args.genes:
        parser.error("Mode 'allele' requires --alleles or --input-file, not --genes")

    # Get the list of items to filter
    if args.input_file:
        items = load_input_list(args.input_file)
        print(f"Loaded {len(items)} {args.mode}(s) from {args.input_file}")
    else:
        # Get from comma-separated command line argument
        items_str = args.genes if args.genes else args.alleles
        items = [item.strip() for item in items_str.split(',')]

    if not items:
        print("Error: No items to filter!")
        sys.exit(1)

    # Create the dataset
    create_dataset(args.mode, items, args.output, args.data)


if __name__ == "__main__":
    main()