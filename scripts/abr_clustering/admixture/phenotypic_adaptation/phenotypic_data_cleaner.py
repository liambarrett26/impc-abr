"""
Missing Data Handler for IMPC Neural ADMIXTURE Pipeline

This module provides comprehensive missing data detection, analysis, and handling
specifically designed for the IMPC ABR dataset and Neural ADMIXTURE pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class MissingDataAnalyzer:
    """
    Comprehensive missing data analysis and handling for IMPC data.
    """

    def __init__(self):
        self.missing_patterns = {}
        self.drop_stats = {}
        self.feature_missing_rates = {}

    def analyze_missing_data(self, df: pd.DataFrame,
                           feature_columns: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of missing data patterns.

        Args:
            df: Input dataframe
            feature_columns: Specific columns to analyze (if None, analyze all)

        Returns:
            Dictionary with missing data analysis results
        """
        if feature_columns is None:
            feature_columns = df.columns.tolist()

        logger.info(f"Analyzing missing data for {len(feature_columns)} features in {len(df)} samples")

        analysis = {
            'total_samples': len(df),
            'total_features': len(feature_columns),
            'missing_by_feature': {},
            'missing_by_sample': {},
            'missing_patterns': {},
            'recommendations': []
        }

        # Feature-level missing data analysis
        for col in feature_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_rate = missing_count / len(df)

                analysis['missing_by_feature'][col] = {
                    'count': int(missing_count),
                    'rate': float(missing_rate),
                    'dtype': str(df[col].dtype)
                }

                if missing_rate > 0.1:  # More than 10% missing
                    analysis['recommendations'].append(
                        f"Feature '{col}' has {missing_rate*100:.1f}% missing values - consider imputation or removal"
                    )
            else:
                logger.warning(f"Feature '{col}' not found in dataframe")

        # Sample-level missing data analysis
        df_subset = df[feature_columns] if feature_columns else df
        missing_per_sample = df_subset.isna().sum(axis=1)

        analysis['missing_by_sample'] = {
            'samples_with_missing': int((missing_per_sample > 0).sum()),
            'max_missing_per_sample': int(missing_per_sample.max()),
            'mean_missing_per_sample': float(missing_per_sample.mean()),
            'samples_complete': int((missing_per_sample == 0).sum())
        }

        # Missing data patterns
        missing_patterns = df_subset.isna().groupby(df_subset.isna().columns.tolist()).size().reset_index()
        missing_patterns.columns = list(missing_patterns.columns[:-1]) + ['count']

        # Convert to readable patterns
        pattern_summary = []
        for _, row in missing_patterns.iterrows():
            pattern = {}
            for col in feature_columns:
                if col in row.index:
                    pattern[col] = bool(row[col])
            pattern_summary.append({
                'pattern': pattern,
                'count': int(row['count']),
                'percentage': float(row['count'] / len(df) * 100)
            })

        analysis['missing_patterns']['top_patterns'] = sorted(
            pattern_summary, key=lambda x: x['count'], reverse=True
        )[:10]  # Top 10 patterns

        # Store for later use
        self.feature_missing_rates = analysis['missing_by_feature']

        return analysis

    def identify_problematic_features(self, analysis: Dict[str, Any],
                                    missing_threshold: float = 0.05) -> List[str]:
        """
        Identify features with high missing rates.

        Args:
            analysis: Output from analyze_missing_data
            missing_threshold: Threshold for considering a feature problematic

        Returns:
            List of problematic feature names
        """
        problematic = []
        for feature, stats in analysis['missing_by_feature'].items():
            if stats['rate'] > missing_threshold:
                problematic.append(feature)
                logger.warning(f"Feature '{feature}' has {stats['rate']*100:.1f}% missing values")

        return problematic

    def clean_data_drop_missing(self, df: pd.DataFrame,
                              feature_columns: List[str],
                              strategy: str = 'any') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean data by dropping samples with missing values.

        Args:
            df: Input dataframe
            feature_columns: Columns to check for missing values
            strategy: 'any' (drop if any missing) or 'all' (drop if all missing)

        Returns:
            Tuple of (cleaned_dataframe, drop_statistics)
        """
        logger.info(f"Cleaning data using '{strategy}' strategy for missing values")

        initial_count = len(df)

        # Subset to feature columns for missing value detection
        df_features = df[feature_columns].copy()

        if strategy == 'any':
            # Drop rows with any missing values in feature columns
            mask = df_features.notna().all(axis=1)
        elif strategy == 'all':
            # Drop rows with all missing values in feature columns
            mask = df_features.notna().any(axis=1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Apply mask to full dataframe
        df_clean = df[mask].copy()
        final_count = len(df_clean)

        # Calculate statistics
        drop_stats = {
            'initial_samples': initial_count,
            'final_samples': final_count,
            'dropped_samples': initial_count - final_count,
            'retention_rate': final_count / initial_count if initial_count > 0 else 0,
            'strategy': strategy,
            'feature_columns_checked': feature_columns
        }

        logger.info(f"Data cleaning complete: {initial_count} → {final_count} samples "
                   f"({drop_stats['retention_rate']*100:.1f}% retained)")

        if drop_stats['retention_rate'] < 0.9:
            logger.warning(f"Significant data loss: {(1-drop_stats['retention_rate'])*100:.1f}% of samples dropped")

        self.drop_stats = drop_stats
        return df_clean, drop_stats

    def impute_missing_values(self, df: pd.DataFrame,
                            feature_columns: List[str],
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Impute missing values using specified strategies.

        Args:
            df: Input dataframe
            feature_columns: Columns to impute
            strategy: Dictionary mapping feature names to imputation strategies
                     ('mean', 'median', 'mode', 'forward_fill', 'backward_fill')

        Returns:
            Dataframe with imputed values
        """
        if strategy is None:
            # Default strategies based on data type
            strategy = {}
            for col in feature_columns:
                if col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        strategy[col] = 'median'
                    else:
                        strategy[col] = 'mode'

        df_imputed = df.copy()

        for col in feature_columns:
            if col not in df.columns:
                continue

            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue

            impute_method = strategy.get(col, 'median')

            logger.info(f"Imputing {missing_count} missing values in '{col}' using '{impute_method}'")

            if impute_method == 'mean':
                fill_value = df[col].mean()
            elif impute_method == 'median':
                fill_value = df[col].median()
            elif impute_method == 'mode':
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else df[col].iloc[0]
            elif impute_method == 'forward_fill':
                df_imputed[col] = df_imputed[col].fillna(method='ffill')
                continue
            elif impute_method == 'backward_fill':
                df_imputed[col] = df_imputed[col].fillna(method='bfill')
                continue
            else:
                logger.warning(f"Unknown imputation method '{impute_method}' for column '{col}'")
                continue

            df_imputed[col] = df_imputed[col].fillna(fill_value)

        return df_imputed

    def create_missing_data_report(self, analysis: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive missing data report.

        Args:
            analysis: Output from analyze_missing_data
            save_path: Optional path to save the report

        Returns:
            Report as string
        """
        report_lines = [
            "="*80,
            "MISSING DATA ANALYSIS REPORT",
            "="*80,
            "",
            f"Dataset Overview:",
            f"  Total Samples: {analysis['total_samples']:,}",
            f"  Total Features: {analysis['total_features']}",
            f"  Complete Samples: {analysis['missing_by_sample']['samples_complete']:,}",
            f"  Samples with Missing: {analysis['missing_by_sample']['samples_with_missing']:,}",
            "",
            "Feature-Level Missing Data:",
            "-" * 40
        ]

        # Sort features by missing rate
        sorted_features = sorted(
            analysis['missing_by_feature'].items(),
            key=lambda x: x[1]['rate'],
            reverse=True
        )

        for feature, stats in sorted_features:
            if stats['count'] > 0:
                report_lines.append(
                    f"  {feature:<30} {stats['count']:>6} ({stats['rate']*100:>5.1f}%)"
                )

        # Top missing patterns
        report_lines.extend([
            "",
            "Top Missing Patterns:",
            "-" * 40
        ])

        for i, pattern_info in enumerate(analysis['missing_patterns']['top_patterns'][:5]):
            missing_features = [f for f, is_missing in pattern_info['pattern'].items() if is_missing]
            if missing_features:
                report_lines.append(
                    f"  Pattern {i+1}: {len(missing_features)} features missing "
                    f"({pattern_info['percentage']:.1f}% of samples)"
                )
                report_lines.append(f"    Missing: {', '.join(missing_features)}")
            else:
                report_lines.append(
                    f"  Pattern {i+1}: Complete data ({pattern_info['percentage']:.1f}% of samples)"
                )

        # Recommendations
        if analysis['recommendations']:
            report_lines.extend([
                "",
                "Recommendations:",
                "-" * 40
            ])
            for rec in analysis['recommendations']:
                report_lines.append(f"  • {rec}")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Missing data report saved to {save_path}")

        return report

    def visualize_missing_patterns(self, df: pd.DataFrame,
                                 feature_columns: List[str],
                                 save_path: Optional[str] = None):
        """
        Create visualizations of missing data patterns.

        Args:
            df: Input dataframe
            feature_columns: Columns to visualize
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Subset data
        df_subset = df[feature_columns].copy()

        # 1. Missing data heatmap
        missing_data = df_subset.isna()
        sns.heatmap(missing_data.iloc[:1000].T, cbar=True, cmap='viridis',
                   ax=axes[0, 0], xticklabels=False, yticklabels=True)
        axes[0, 0].set_title('Missing Data Pattern (First 1000 samples)')
        axes[0, 0].set_xlabel('Samples')

        # 2. Missing data by feature
        missing_by_feature = missing_data.sum().sort_values(ascending=True)
        missing_by_feature.plot(kind='barh', ax=axes[0, 1])
        axes[0, 1].set_title('Missing Values by Feature')
        axes[0, 1].set_xlabel('Number of Missing Values')

        # 3. Missing data by sample
        missing_by_sample = missing_data.sum(axis=1)
        axes[1, 0].hist(missing_by_sample, bins=50, alpha=0.7)
        axes[1, 0].set_title('Distribution of Missing Values per Sample')
        axes[1, 0].set_xlabel('Number of Missing Features')
        axes[1, 0].set_ylabel('Number of Samples')

        # 4. Missing data correlation
        if len(feature_columns) > 1:
            missing_corr = missing_data.corr()
            sns.heatmap(missing_corr, annot=False, cmap='coolwarm', center=0,
                       ax=axes[1, 1], cbar=True)
            axes[1, 1].set_title('Missing Data Correlation')
        else:
            axes[1, 1].text(0.5, 0.5, 'Need >1 feature\nfor correlation',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Missing data visualization saved to {save_path}")

        return fig


def integrate_missing_data_handling(preprocessor_class):
    """
    Decorator to integrate missing data handling into existing preprocessor classes.

    Args:
        preprocessor_class: The class to enhance with missing data handling

    Returns:
        Enhanced class with missing data capabilities
    """

    class EnhancedPreprocessor(preprocessor_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.missing_analyzer = MissingDataAnalyzer()
            self.missing_handling_strategy = 'drop'  # 'drop' or 'impute'

        def handle_missing_data(self, df: pd.DataFrame,
                              feature_columns: List[str],
                              strategy: str = 'drop',
                              missing_threshold: float = 0.05) -> Tuple[pd.DataFrame, Dict[str, Any]]:
            """
            Handle missing data in the preprocessing pipeline.

            Args:
                df: Input dataframe
                feature_columns: Columns to check for missing values
                strategy: 'drop' or 'impute'
                missing_threshold: Threshold for dropping features with high missing rates

            Returns:
                Tuple of (processed_dataframe, processing_statistics)
            """
            logger.info(f"Handling missing data using strategy: {strategy}")

            # Analyze missing data
            analysis = self.missing_analyzer.analyze_missing_data(df, feature_columns)

            # Create report
            report = self.missing_analyzer.create_missing_data_report(analysis)
            logger.info("Missing Data Analysis:\n" + report)

            # Identify problematic features
            problematic_features = self.missing_analyzer.identify_problematic_features(
                analysis, missing_threshold
            )

            if problematic_features:
                logger.warning(f"Found {len(problematic_features)} features with high missing rates")

            stats = {
                'analysis': analysis,
                'problematic_features': problematic_features,
                'strategy_used': strategy
            }

            if strategy == 'drop':
                # Drop samples with missing values
                df_clean, drop_stats = self.missing_analyzer.clean_data_drop_missing(
                    df, feature_columns, strategy='any'
                )
                stats['drop_stats'] = drop_stats
                return df_clean, stats

            elif strategy == 'impute':
                # Impute missing values
                df_imputed = self.missing_analyzer.impute_missing_values(df, feature_columns)
                return df_imputed, stats

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

    return EnhancedPreprocessor


# Specific function for your IMPC ABR pipeline
def clean_impc_data_for_neural_admixture(df: pd.DataFrame,
                                        abr_columns: List[str] = None,
                                        metadata_columns: List[str] = None,
                                        save_report: bool = True,
                                        output_dir: str = "outputs/preprocessing_outputs") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean IMPC data specifically for Neural ADMIXTURE pipeline.

    Args:
        df: Raw IMPC dataframe
        abr_columns: ABR threshold columns (if None, will detect automatically)
        metadata_columns: Metadata columns to include in analysis
        save_report: Whether to save missing data report
        output_dir: Output directory for reports

    Returns:
        Tuple of (cleaned_dataframe, cleaning_statistics)
    """
    if abr_columns is None:
        # Detect ABR columns automatically
        potential_abr_cols = [col for col in df.columns if 'threshold' in col.lower() or 'abr' in col.lower()]
        abr_columns = potential_abr_cols

    if metadata_columns is None:
        # Standard IMPC metadata columns
        metadata_columns = [
            'age_in_days', 'age_in_weeks', 'weight', 'sex', 'zygosity',
            'genetic_background', 'phenotyping_center', 'date_of_birth', 'date_of_experiment'
        ]
        # Filter to columns that actually exist
        metadata_columns = [col for col in metadata_columns if col in df.columns]

    all_feature_columns = abr_columns + metadata_columns

    logger.info(f"Cleaning IMPC data for Neural ADMIXTURE")
    logger.info(f"ABR columns: {abr_columns}")
    logger.info(f"Metadata columns: {metadata_columns}")

    # Initialize analyzer
    analyzer = MissingDataAnalyzer()

    # Analyze missing data
    analysis = analyzer.analyze_missing_data(df, all_feature_columns)

    # Create and save report
    if save_report:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(output_dir) / "missing_data_report.txt"
        report = analyzer.create_missing_data_report(analysis, str(report_path))

        # Save visualization
        viz_path = Path(output_dir) / "missing_data_patterns.png"
        analyzer.visualize_missing_patterns(df, all_feature_columns, str(viz_path))

    # Clean data by dropping samples with missing values
    df_clean, drop_stats = analyzer.clean_data_drop_missing(
        df, all_feature_columns, strategy='any'
    )

    # Check if data loss is acceptable
    if drop_stats['retention_rate'] < 0.8:
        logger.error(f"Excessive data loss: {(1-drop_stats['retention_rate'])*100:.1f}% of samples would be dropped")
        logger.info("Consider using imputation strategy instead")

        # Offer imputation as alternative
        df_imputed = analyzer.impute_missing_values(df, all_feature_columns)

        return df_imputed, {
            'method': 'imputation',
            'analysis': analysis,
            'drop_stats': drop_stats,
            'imputation_applied': True,
            'cleaned_df_shape': df_imputed.shape,
            'original_df_shape': df.shape
        }

    return df_clean, {
        'method': 'drop_missing',
        'analysis': analysis,
        'drop_stats': drop_stats,
        'imputation_applied': False,
        'cleaned_df_shape': df_clean.shape,
        'original_df_shape': df.shape
    }