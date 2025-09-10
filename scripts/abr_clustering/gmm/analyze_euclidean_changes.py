import pandas as pd

# Load the Euclidean assignments
df = pd.read_csv('results/euclidean_assignment/gene_cluster_associations_euclidean.csv')

# Find genes with changed assignments
changed = df[df['changed_assignment']]
print(f'Total genes with changed assignments: {len(changed)}')
print(f'\nGenes that moved TO cluster 3 (high-frequency loss):')

moved_to_3 = changed[(changed['dominant_cluster_euclidean'] == 3) & (changed['dominant_cluster_original'] != 3)]
for _, row in moved_to_3.iterrows():
    print(f"  {row['gene_symbol']} ({row['allele_symbol'][:20]}...): Cluster {row['dominant_cluster_original']} -> 3, consistency {row['consistency_score_original']:.2f} -> {row['consistency_score_euclidean']:.2f}")

print(f'\n\nGenes with largest consistency improvements:')
df['consistency_improvement'] = df['consistency_score_euclidean'] - df['consistency_score_original']
top_improved = df.nlargest(10, 'consistency_improvement')
for _, row in top_improved.iterrows():
    print(f"  {row['gene_symbol']}: {row['consistency_score_original']:.2f} -> {row['consistency_score_euclidean']:.2f} (+{row['consistency_improvement']:.2f}), Cluster {row['dominant_cluster_original']} -> {row['dominant_cluster_euclidean']}")