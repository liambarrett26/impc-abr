#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fixed Bootstrap Script - Handles Zygosity Properly

This version correctly separates different zygosity groups and ensures
the right models are saved to the right directories.

Author: Liam Barrett
Version: 1.0.0
"""

import sys
import time
import argparse
from pathlib import Path
import tempfile
import shutil
import json

# Add the package directory to the path
script_dir = Path(__file__).parent
package_dir = script_dir.parent
sys.path.insert(0, str(package_dir))

from abr_analysis.analysis.parallel_executor import process_gene_groups

# Complete target statistics from concatenated_results_v6.csv
TARGET_STATS = {
    'Pank2': {
        'Pank2<em1(IMPC)Ccpcz>_homozygote_CCP-IMG': {
            'all': {'p_hearing_loss': 0.644, 'hdi_lower': 0.404, 'hdi_upper': 0.885},
            'male': {'p_hearing_loss': 0.536, 'hdi_lower': 0.232, 'hdi_upper': 0.830},
            'female': {'p_hearing_loss': 0.522, 'hdi_lower': 0.198, 'hdi_upper': 0.853}
        }
    },
    'Mettl5': {
        'Mettl5<em1(IMPC)Hmgu>_heterozygote_HMGU': {
            'all': {'p_hearing_loss': 0.857, 'hdi_lower': 0.724, 'hdi_upper': 0.980},
            'male': {'p_hearing_loss': 0.786, 'hdi_lower': 0.588, 'hdi_upper': 0.961},
            'female': {'p_hearing_loss': 0.651, 'hdi_lower': 0.376, 'hdi_upper': 0.910}
        },
        'Mettl5<em1(IMPC)Hmgu>_homozygote_HMGU': {
            'all': {'p_hearing_loss': 0.734, 'hdi_lower': 0.516, 'hdi_upper': 0.952},
            'male': {'p_hearing_loss': 0.685, 'hdi_lower': 0.422, 'hdi_upper': 0.945},
            'female': {'p_hearing_loss': 0.573, 'hdi_lower': 0.252, 'hdi_upper': 0.880}
        }
    },
    'Adgrb1': {
        'Adgrb1<tm2a(EUCOMM)Wtsi>_homozygote_WTSI': {
            'all': {'p_hearing_loss': 0.750, 'hdi_lower': 0.539, 'hdi_upper': 0.964},
            'male': {'p_hearing_loss': 0.698, 'hdi_lower': 0.441, 'hdi_upper': 0.933}
        },
        'Adgrb1<tm2a(EUCOMM)Wtsi>_heterozygote_WTSI': {
            'all': {'p_hearing_loss': 0.125, 'hdi_lower': 0.000, 'hdi_upper': 0.331}
        }
    },
    'Wdtc1': {
        'Wdtc1<tm1a(KOMP)Wtsi>_homozygote_WTSI': {
            'all': {'p_hearing_loss': 0.454, 'hdi_lower': 0.195, 'hdi_upper': 0.723},
            'male': {'p_hearing_loss': 0.143, 'hdi_lower': 0.000, 'hdi_upper': 0.372},
            'female': {'p_hearing_loss': 0.626, 'hdi_lower': 0.323, 'hdi_upper': 0.910}
        }
    }
}

TOLERANCE = 0.015  # 1.5% tolerance

class ZygosityAwareBootstrapper:
    """Manages bootstrap with proper zygosity separation."""
    
    def __init__(self, gene, output_dir):
        self.gene = gene
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track what we've already saved
        self.saved_models = {}  # allele_key -> {analysis_type: True/False}
        self.load_existing_progress()
    
    def load_existing_progress(self):
        """Load any existing saved models to avoid re-doing work."""
        progress_file = self.output_dir / "progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                self.saved_models = json.load(f)
            print(f"  Loaded existing progress: {sum(len(allele.values()) for allele in self.saved_models.values())} models saved")
    
    def save_progress(self):
        """Save progress to avoid losing work."""
        progress_file = self.output_dir / "progress.json"
        with open(progress_file, 'w') as f:
            json.dump(self.saved_models, f, indent=2)
    
    def is_already_saved(self, allele_key, analysis_type):
        """Check if this model is already saved."""
        return self.saved_models.get(allele_key, {}).get(analysis_type, False)
    
    def mark_as_saved(self, allele_key, analysis_type):
        """Mark a model as saved."""
        if allele_key not in self.saved_models:
            self.saved_models[allele_key] = {}
        self.saved_models[allele_key][analysis_type] = True
        self.save_progress()
    
    def find_specific_model_dir(self, temp_dir, target_allele_key, analysis_type):
        """Find the specific model directory for the given allele and analysis type."""
        # Parse the target allele key to extract zygosity
        parts = target_allele_key.split('_')
        target_zygosity = parts[1]  # e.g., 'homozygote' or 'heterozygote'
        
        temp_path = Path(temp_dir)
        models_dir = temp_path / "models" / self.gene
        
        if not models_dir.exists():
            return None
        
        # Map analysis types to suffixes
        analysis_suffix_map = {
            'all': '_all',
            'male': '_male', 
            'female': '_female'
        }
        suffix = analysis_suffix_map[analysis_type]
        
        # Find directories that end with the right suffix AND contain the right zygosity
        for item in models_dir.iterdir():
            if item.is_dir() and item.name.endswith(suffix):
                # Check if this directory corresponds to the target zygosity
                # The directory name should contain the zygosity
                if target_zygosity in item.name:
                    return item
        
        print(f"    Warning: Could not find model directory for {target_allele_key}_{analysis_type}")
        print(f"    Available directories in {models_dir}:")
        if models_dir.exists():
            for item in models_dir.iterdir():
                if item.is_dir():
                    print(f"      {item.name}")
        return None
    
    def copy_model_files(self, temp_dir, allele_key, analysis_type):
        """Copy model files for a specific analysis to permanent storage."""
        source_dir = self.find_specific_model_dir(temp_dir, allele_key, analysis_type)
        
        if not source_dir:
            return False
        
        # Copy to permanent location
        dest_dir = self.output_dir / allele_key / f"{allele_key}_{analysis_type}"
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(source_dir, dest_dir)
            print(f"    ✅ SAVED: {allele_key}_{analysis_type}")
            print(f"       From: {source_dir.name}")
            return True
        except Exception as e:
            print(f"    Error saving {allele_key}_{analysis_type}: {e}")
            return False
    
    def get_completion_status(self):
        """Get completion status for all target alleles."""
        target_alleles = TARGET_STATS[self.gene]
        total_targets = 0
        completed_targets = 0
        
        for allele_key, target_analyses in target_alleles.items():
            for analysis_type in target_analyses.keys():
                total_targets += 1
                if self.is_already_saved(allele_key, analysis_type):
                    completed_targets += 1
        
        return completed_targets, total_targets

def extract_allele_stats_with_zygosity(results):
    """Extract statistics organized by allele, ensuring zygosity separation."""
    allele_stats = {}
    
    for result in results:
        if not result:
            continue
        
        # Create allele key with explicit zygosity
        allele = result.get('allele_symbol', '')
        zygosity = result.get('zygosity', '')
        center = result.get('center', '')
        allele_key = f"{allele}_{zygosity}_{center}"
        
        if allele_key not in allele_stats:
            allele_stats[allele_key] = {}
        
        # Extract all, male, female stats
        for analysis_type in ['all', 'male', 'female']:
            p_col = f'{analysis_type}_p_hearing_loss'
            if result.get(p_col) is not None:
                allele_stats[allele_key][analysis_type] = {
                    'p_hearing_loss': result.get(p_col),
                    'hdi_lower': result.get(f'{analysis_type}_hdi_lower'),
                    'hdi_upper': result.get(f'{analysis_type}_hdi_upper'),
                    'zygosity': zygosity,  # Track zygosity for debugging
                    'center': center
                }
    
    return allele_stats

def check_stats_match(result_stats, target_stats):
    """Check if statistics match within tolerance."""
    for stat in ['p_hearing_loss', 'hdi_lower', 'hdi_upper']:
        if stat not in result_stats or stat not in target_stats:
            return False
        
        result_val = result_stats[stat]
        target_val = target_stats[stat]
        
        if result_val is None or target_val is None:
            return False
        
        # Calculate relative difference
        if abs(target_val) < 1e-6:
            diff = abs(result_val - target_val)
        else:
            diff = abs(result_val - target_val) / abs(target_val)
        
        if diff > TOLERANCE:
            return False
    
    return True

def bootstrap_gene_zygosity_aware(gene, data_path, max_iterations):
    """Bootstrap with proper zygosity handling."""
    print(f"\nBootstrapping gene: {gene}")
    
    if gene not in TARGET_STATS:
        print(f"No target statistics for {gene}")
        return False
    
    # Create bootstrapper
    output_dir = f"results/bootstrap_zygosity_fixed_{gene}"
    bootstrapper = ZygosityAwareBootstrapper(gene, output_dir)
    
    # Check initial completion status
    completed, total = bootstrapper.get_completion_status()
    print(f"Progress: {completed}/{total} models already saved")
    
    if completed == total:
        print(f"✅ All models already completed for {gene}!")
        return True
    
    target_alleles = TARGET_STATS[gene]
    print(f"Target alleles: {list(target_alleles.keys())}")
    
    for iteration in range(max_iterations):
        print(f"  Iteration {iteration + 1}/{max_iterations}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Run analysis
                results = process_gene_groups(gene, data_path, temp_dir)
                
                if not results:
                    print("    No results")
                    continue
                
                # Extract statistics by allele with zygosity awareness
                allele_results = extract_allele_stats_with_zygosity(results)
                
                if not allele_results:
                    print("    Could not extract statistics")
                    continue
                
                print(f"    Found results for: {list(allele_results.keys())}")
                
                # Check each target allele and analysis type
                new_saves = 0
                
                for target_allele_key, target_allele_stats in target_alleles.items():
                    if target_allele_key in allele_results:
                        result_allele_stats = allele_results[target_allele_key]
                        
                        for analysis_type, target_analysis_stats in target_allele_stats.items():
                            # Skip if already saved
                            if bootstrapper.is_already_saved(target_allele_key, analysis_type):
                                continue
                            
                            if analysis_type in result_allele_stats:
                                result_analysis_stats = result_allele_stats[analysis_type]
                                if check_stats_match(result_analysis_stats, target_analysis_stats):
                                    # Found a match! Save it immediately
                                    if bootstrapper.copy_model_files(temp_dir, target_allele_key, analysis_type):
                                        bootstrapper.mark_as_saved(target_allele_key, analysis_type)
                                        new_saves += 1
                                        print(f"    🎯 MATCH: {target_allele_key[:30]}...{analysis_type}")
                                        print(f"        p_hl={result_analysis_stats['p_hearing_loss']:.3f} (zygosity: {result_analysis_stats['zygosity']})")
                                else:
                                    print(f"    ❌ {target_allele_key[:30]}...{analysis_type}: p_hl={result_analysis_stats['p_hearing_loss']:.3f} (target: {target_analysis_stats['p_hearing_loss']:.3f})")
                            else:
                                print(f"    ❓ {target_allele_key[:30]}...{analysis_type}: No results")
                    else:
                        print(f"    ❌ {target_allele_key}: Not found in results")
                
                # Check if we're done
                completed, total = bootstrapper.get_completion_status()
                print(f"    Progress: {completed}/{total} (+{new_saves} new)")
                
                if completed == total:
                    print(f"  🎉 SUCCESS: All models completed for {gene}!")
                    return True
                    
            except Exception as e:
                print(f"    Error: {str(e)[:100]}...")
                continue
    
    # Final status
    completed, total = bootstrapper.get_completion_status()
    print(f"  Final progress: {completed}/{total} models saved")
    
    if completed == total:
        print(f"  🎉 SUCCESS: All models completed for {gene}!")
        return True
    else:
        print(f"  ⚠️  PARTIAL: {completed}/{total} models completed for {gene}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Zygosity-aware bootstrap for missing model configurations")
    parser.add_argument("--data", required=True, help="Path to ABR data CSV file")
    parser.add_argument("--gene", help="Single gene to bootstrap")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum iterations per gene")
    
    args = parser.parse_args()
    
    # Validate data file
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)
    
    # Determine genes to process
    if args.gene:
        if args.gene not in TARGET_STATS:
            print(f"ERROR: No target statistics for gene {args.gene}")
            print(f"Available genes: {list(TARGET_STATS.keys())}")
            sys.exit(1)
        genes_to_process = [args.gene]
    else:
        genes_to_process = list(TARGET_STATS.keys())
    
    print(f"Processing genes: {genes_to_process}")
    print(f"Max iterations per gene: {args.max_iterations}")
    print(f"Tolerance: {TOLERANCE}")
    
    start_time = time.time()
    
    # Process genes
    results = {}
    for gene in genes_to_process:
        success = bootstrap_gene_zygosity_aware(gene, str(data_path), args.max_iterations)
        results[gene] = success
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("ZYGOSITY-AWARE BOOTSTRAP SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    successful = [gene for gene, success in results.items() if success]
    failed = [gene for gene, success in results.items() if not success]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"SUCCESS: {', '.join(successful)}")
    
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    
    sys.exit(0 if len(failed) == 0 else 1)

if __name__ == "__main__":
    main()