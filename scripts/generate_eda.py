import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA

# Set numpy to not print scientific notation
np.set_printoptions(suppress=True)

# Load the processed data
print("Loading processed GWAS data...")
data = pd.read_csv('processed_data/multitask_gwas_data.csv')
print(f"Dataset shape: {data.shape}")

# Check for NaN or infinite values
print("\nChecking for NaN or infinite values...")
for col in data.columns:
    nan_count = data[col].isna().sum()
    inf_count = np.isinf(data[col].fillna(0)).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"Column {col}: {nan_count} NaNs, {inf_count} infinites")
        # Replace infinites with NaN, then fill NaNs with column mean
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        data[col] = data[col].fillna(data[col].mean())

# Create directory for EDA visualizations
os.makedirs('eda_plots', exist_ok=True)

# 1. Distribution of effect sizes for each disease
print("\nGenerating effect size distribution plots...")
plt.figure(figsize=(12, 6))
for i, disease in enumerate(['cardio', 't2d', 'cancer']):
    plt.subplot(1, 3, i+1)
    sns.histplot(data[f'log_odds_{disease}'], kde=True)
    plt.title(f'{disease.capitalize()} Effect Size Distribution')
    plt.xlabel('Log Odds')
    plt.tight_layout()
plt.savefig('eda_plots/effect_size_distributions.png')
plt.close()

# 2. P-value distributions
print("\nGenerating p-value distributions...")
plt.figure(figsize=(12, 6))
for i, disease in enumerate(['cardio', 't2d', 'cancer']):
    plt.subplot(1, 3, i+1)
    # Ensure p-values are between 0 and 1
    p_values = data[f'pvalue_{disease}'].clip(0, 1)
    # Ensure we don't have zeros that would cause -inf when log transformed
    p_values = p_values.replace(0, 1e-10)
    sns.histplot(-np.log10(p_values), kde=True)
    plt.title(f'{disease.capitalize()} -log10(p-value) Distribution')
    plt.xlabel('-log10(p-value)')
    plt.axvline(-np.log10(0.05), color='red', linestyle='--')
    plt.tight_layout()
plt.savefig('eda_plots/pvalue_distributions.png')
plt.close()

# 3. Correlation matrix between features
print("\nGenerating correlation matrix...")
features = []
for disease in ['cardio', 't2d', 'cancer']:
    features.extend([f'log_odds_{disease}', f'log_odds_se_{disease}', f'pvalue_{disease}'])

correlation_matrix = data[features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of GWAS Features')
plt.tight_layout()
plt.savefig('eda_plots/feature_correlation_matrix.png')
plt.close()

# 4. Chromosome distribution of significant SNPs
print("\nGenerating chromosome distribution plot...")
plt.figure(figsize=(14, 6))
chrom_counts = data['chromosome'].value_counts().sort_index()
sns.barplot(x=chrom_counts.index, y=chrom_counts.values)
plt.title('Distribution of SNPs Across Chromosomes')
plt.xlabel('Chromosome')
plt.ylabel('Number of SNPs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_plots/chromosome_distribution.png')
plt.close()

# 5. Feature dimensionality reduction with PCA for visualization
print("\nGenerating PCA visualization...")
# Prepare features for PCA
X = data[features].values
# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with PCA results and risk categories
pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cardio_Risk': data['high_risk_cardio'],
    'T2D_Risk': data['high_risk_t2d'],
    'Cancer_Risk': data['high_risk_cancer']
})

# Plot PCA results colored by risk category for each disease
diseases = ['Cardio_Risk', 'T2D_Risk', 'Cancer_Risk']
disease_names = ['Cardiovascular Disease', 'Type 2 Diabetes', 'Cancer']

plt.figure(figsize=(18, 6))
for i, (disease, name) in enumerate(zip(diseases, disease_names)):
    plt.subplot(1, 3, i+1)
    for risk in [0, 1]:
        subset = pca_df[pca_df[disease] == risk]
        plt.scatter(subset['PC1'], subset['PC2'], 
                   label=f'{"High" if risk else "Low"} Risk',
                   alpha=0.7, s=10)
    plt.title(f'PCA of GWAS Features colored by {name} Risk')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.tight_layout()
plt.savefig('eda_plots/pca_visualization.png')
plt.close()

# 6. Visualize the relationship between effect size and standard error
print("\nGenerating effect size vs. standard error plots...")
plt.figure(figsize=(18, 6))
for i, disease in enumerate(['cardio', 't2d', 'cancer']):
    plt.subplot(1, 3, i+1)
    plt.scatter(data[f'log_odds_{disease}'], data[f'log_odds_se_{disease}'], 
               alpha=0.5, s=10, c=data[f'high_risk_{disease}'], cmap='coolwarm')
    plt.colorbar(label='High Risk')
    plt.title(f'{disease.capitalize()} Effect Size vs. Standard Error')
    plt.xlabel('Log Odds (Effect Size)')
    plt.ylabel('Standard Error')
    plt.tight_layout()
plt.savefig('eda_plots/effect_size_vs_se.png')
plt.close()

# 7. Manhattan plots for each disease - simplified version to avoid finite value errors
print("\nGenerating simplified Manhattan plots...")
def safe_manhattan_plot(data, pvalue_col, title, outfile):
    # Create a copy and ensure no NaN or infinite values
    plot_data = data.copy()
    
    # Ensure p-values are valid (between 0 and 1) and not zero
    plot_data[pvalue_col] = plot_data[pvalue_col].clip(1e-10, 1)
    
    # -log10 transform p-values
    plot_data['log_pvalue'] = -np.log10(plot_data[pvalue_col])
    
    # Ensure chromosomes are treated as strings for categorical ordering
    plot_data['chromosome'] = plot_data['chromosome'].astype(str)
    
    # Simple scatterplot by chromosome
    plt.figure(figsize=(14, 6))
    
    # Group by chromosome for coloring
    for i, (chrom, group) in enumerate(plot_data.groupby('chromosome')):
        color = 'blue' if i % 2 == 0 else 'skyblue'
        plt.scatter(group.index, group['log_pvalue'], color=color, alpha=0.8, s=5, label=chrom if i < 10 else "")
    
    # Add reference lines
    plt.axhline(y=-np.log10(5e-8), color='red', linestyle='--', alpha=0.5, label='p = 5e-8')
    plt.axhline(y=-np.log10(0.05), color='green', linestyle='--', alpha=0.5, label='p = 0.05')
    
    plt.title(title)
    plt.xlabel('SNP Index')
    plt.ylabel('-log10(p-value)')
    plt.legend(loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

# Create Manhattan plots for each disease
safe_manhattan_plot(data, 'pvalue_cardio', 
                   'Manhattan Plot: Cardiovascular Disease', 
                   'eda_plots/manhattan_cardio.png')

safe_manhattan_plot(data, 'pvalue_t2d', 
                   'Manhattan Plot: Type 2 Diabetes', 
                   'eda_plots/manhattan_t2d.png')

safe_manhattan_plot(data, 'pvalue_cancer', 
                   'Manhattan Plot: Cancer', 
                   'eda_plots/manhattan_cancer.png')

# 8. Create a custom Venn diagram-like visualization (without matplotlib_venn)
print("\nGenerating custom Venn diagram alternative...")

# Count overlap between high-risk groups
cardio_high = set(data[data['high_risk_cardio'] == 1].index)
t2d_high = set(data[data['high_risk_t2d'] == 1].index)
cancer_high = set(data[data['high_risk_cancer'] == 1].index)

# Calculate overlaps
cardio_only = len(cardio_high - t2d_high - cancer_high)
t2d_only = len(t2d_high - cardio_high - cancer_high)
cancer_only = len(cancer_high - cardio_high - t2d_high)
cardio_t2d = len(cardio_high & t2d_high - cancer_high)
cardio_cancer = len(cardio_high & cancer_high - t2d_high)
t2d_cancer = len(t2d_high & cancer_high - cardio_high)
all_three = len(cardio_high & t2d_high & cancer_high)

# Create a bar chart of overlaps
overlap_data = {
    'Category': ['Cardio Only', 'T2D Only', 'Cancer Only', 
                'Cardio & T2D', 'Cardio & Cancer', 'T2D & Cancer', 'All Three'],
    'Count': [cardio_only, t2d_only, cancer_only, 
             cardio_t2d, cardio_cancer, t2d_cancer, all_three]
}
overlap_df = pd.DataFrame(overlap_data)

plt.figure(figsize=(12, 6))
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'black']
sns.barplot(x='Category', y='Count', data=overlap_df, palette=colors)
plt.title('Overlap of High-Risk SNPs Across Diseases')
plt.xlabel('Disease Category')
plt.ylabel('Number of SNPs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_plots/high_risk_overlap.png')
plt.close()

print("\nEDA visualizations complete!")
