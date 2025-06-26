import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import pearsonr, spearmanr, kendalltau

df = pd.read_csv('all_portfolios_returns.csv')
# df = pd.read_csv('all_risk_factors_net_returns.csv')

# Convert 'Date' column to datetime format and set it as index
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.set_index('Date', inplace=True)

# Convert percentage strings to floats for all non-Date columns
mask = df.dtypes == 'object'
df.loc[:, mask] = df.loc[:, mask].apply(lambda x: x.str.rstrip('%').astype(float) / 100)

# Drop rows with any missing values
df = df.dropna()

print("\nNumber of rows after dropping missing values:", len(df))

# Calculate correlation matrices
corr_methods = {
    'Pearson': df.corr(method='pearson'),
    'Spearman': df.corr(method='spearman'),
    'Kendall': df.corr(method='kendall')
}

# Prepare for plotting
num_columns = df.shape[1]
figsize = (max(15, num_columns * 1.5), max(5, num_columns * 0.5))
fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.05})

# Define a single colorbar axis
cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))  # (left, bottom, width, height)

# Plot heatmaps with shared colorbar (mask upper triangle to hide duplicate numbers)
for ax, (name, corr) in zip(axes, corr_methods.items()):
    annot = corr.round(2).astype(str)
    # Create a mask for the upper triangle
    mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, annot=annot, fmt='s', cmap='coolwarm', vmin=-1, vmax=1, center=0,
        mask=mask_upper, square=True, cbar=False, annot_kws={'size': max(6, 10 - num_columns * 0.3)}
    )
    ax.set_title(name)
    ax.tick_params(axis='x', rotation=45)

# Add a single colorbar for all heatmaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sm = ScalarMappable(cmap='coolwarm', norm=Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Correlation')

# Adjust layout and save figure
plt.tight_layout(rect=(0, 0, 0.9, 1))  # Leave space for colorbar
# plt.savefig('portfolios_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
# plt.savefig('risk_factors_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()
