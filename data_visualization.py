import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

def load_data(filepath='Dataset/Data.csv'):
    """Load the dataset"""
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    return df

def create_overview_stats(df):
    """Generate and visualize dataset overview statistics"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Dataset Overview & Statistics', fontsize=20, fontweight='bold', y=0.98)

    # 1. Dataset Info Box
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    info_text = f"""
    Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
    Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB
    Missing Values: {df.isnull().sum().sum()}
    Duplicate Rows: {df.duplicated().sum()}
    """
    ax1.text(0.5, 0.5, info_text, ha='center', va='center',
             fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Machine Failure Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    failure_counts = df['Machine failure'].value_counts()
    colors_pie = ['#2ECC71', '#E74C3C']
    wedges, texts, autotexts = ax2.pie(failure_counts.values,
                                         labels=['No Failure', 'Failure'],
                                         autopct='%1.1f%%',
                                         colors=colors_pie,
                                         explode=(0, 0.1),
                                         shadow=True,
                                         startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_weight('bold')
    ax2.set_title('Machine Failure Distribution', fontsize=14, fontweight='bold')

    # 3. Product Type Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    type_counts = df['Type'].value_counts()
    bars = ax3.bar(type_counts.index, type_counts.values, color=COLORS[:3], edgecolor='black', linewidth=1.2)
    ax3.set_title('Product Type Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Type', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Failure Types Distribution
    ax4 = fig.add_subplot(gs[1, 2])
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    failure_counts = [df[ft].sum() for ft in failure_types]
    bars = ax4.barh(failure_types, failure_counts, color=COLORS, edgecolor='black', linewidth=1.2)
    ax4.set_title('Failure Type Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Count', fontsize=11)
    for i, (bar, count) in enumerate(zip(bars, failure_counts)):
        ax4.text(count + 5, i, str(count), va='center', fontsize=10, fontweight='bold')

    # 5. Numerical Features Statistics Table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')

    numerical_cols = ['Air temperature [K]', 'Process temperature [K]',
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    stats_data = []
    for col in numerical_cols:
        stats_data.append([
            col,
            f"{df[col].mean():.2f}",
            f"{df[col].std():.2f}",
            f"{df[col].min():.2f}",
            f"{df[col].quantile(0.25):.2f}",
            f"{df[col].median():.2f}",
            f"{df[col].quantile(0.75):.2f}",
            f"{df[col].max():.2f}"
        ])

    table = ax5.table(cellText=stats_data,
                      colLabels=['Feature', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.2] + [0.1]*7)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(8):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_data) + 1):
        for j in range(8):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')

    plt.savefig('01_dataset_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: 01_dataset_overview.png")

def plot_distributions(df):
    """Create beautiful distribution plots for all numerical features"""
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]',
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Feature Distributions with KDE', fontsize=20, fontweight='bold')
    axes = axes.flatten()

    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]

        # Histogram with KDE
        ax.hist(df[col], bins=50, alpha=0.6, color=COLORS[idx], edgecolor='black', density=True)

        # KDE overlay
        df[col].plot(kind='kde', ax=ax, color='darkred', linewidth=2.5, label='KDE')

        # Styling
        ax.set_title(col, fontsize=14, fontweight='bold')
        ax.set_xlabel('Value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f"μ={df[col].mean():.2f}\nσ={df[col].std():.2f}\nmedian={df[col].median():.2f}"
        ax.text(0.72, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig('02_feature_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: 02_feature_distributions.png")

def plot_boxplots(df):
    """Create box plots comparing features by machine failure"""
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]',
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Feature Distribution by Machine Failure Status', fontsize=20, fontweight='bold')
    axes = axes.flatten()

    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]

        # Create box plot
        bp = ax.boxplot([df[df['Machine failure']==0][col],
                         df[df['Machine failure']==1][col]],
                        labels=['No Failure', 'Failure'],
                        patch_artist=True,
                        notch=True,
                        widths=0.6)

        # Color the boxes
        colors = ['#2ECC71', '#E74C3C']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Styling
        ax.set_title(col, fontsize=13, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig('03_feature_boxplots.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: 03_feature_boxplots.png")

def plot_correlation_heatmap(df):
    """Create a beautiful correlation heatmap"""
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]',
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                      'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    fig, ax = plt.subplots(figsize=(14, 11))

    # Calculate correlation
    corr = df[numerical_cols].corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlBu_r', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)

    ax.set_title('Feature Correlation Heatmap', fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig('04_correlation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: 04_correlation_heatmap.png")

def plot_pairplot(df):
    """Create pair plot for key features"""
    print("Creating pair plot (this may take a moment)...")

    # Sample data for faster plotting if dataset is large
    df_sample = df.sample(n=min(2000, len(df)), random_state=42)

    selected_features = ['Air temperature [K]', 'Process temperature [K]',
                         'Rotational speed [rpm]', 'Torque [Nm]', 'Machine failure']

    g = sns.pairplot(df_sample[selected_features],
                     hue='Machine failure',
                     palette={0: '#2ECC71', 1: '#E74C3C'},
                     diag_kind='kde',
                     plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
                     diag_kws={'linewidth': 2})

    g.fig.suptitle('Pairwise Feature Relationships', fontsize=18, fontweight='bold', y=1.01)

    plt.savefig('05_pairplot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: 05_pairplot.png")

def plot_failure_analysis(df):
    """Detailed failure type analysis"""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Comprehensive Failure Type Analysis', fontsize=20, fontweight='bold')

    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    # 1. Failure type percentages
    ax1 = fig.add_subplot(gs[0, 0])
    total_failures = sum([df[ft].sum() for ft in failure_types])
    percentages = [(df[ft].sum() / total_failures * 100) for ft in failure_types]
    bars = ax1.bar(failure_types, percentages, color=COLORS, edgecolor='black', linewidth=1.2)
    ax1.set_title('Failure Type Distribution (%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Failure types by Product Type
    ax2 = fig.add_subplot(gs[0, 1])
    product_types = df['Type'].unique()
    failure_by_type = []
    for pt in product_types:
        counts = [df[df['Type']==pt][ft].sum() for ft in failure_types]
        failure_by_type.append(counts)

    x = np.arange(len(failure_types))
    width = 0.25
    for i, pt in enumerate(product_types):
        ax2.bar(x + i*width, failure_by_type[i], width, label=pt,
                color=COLORS[i], edgecolor='black', linewidth=1)

    ax2.set_title('Failure Types by Product Category', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(failure_types)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Tool wear vs failures
    ax3 = fig.add_subplot(gs[0, 2])
    scatter = ax3.scatter(df['Tool wear [min]'], df['Torque [Nm]'],
                         c=df['Machine failure'], cmap='RdYlGn_r',
                         alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    ax3.set_title('Tool Wear vs Torque (colored by failure)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Tool Wear [min]', fontsize=11)
    ax3.set_ylabel('Torque [Nm]', fontsize=11)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Failure', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Temperature relationship
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(df['Air temperature [K]'], df['Process temperature [K]'],
                         c=df['Machine failure'], cmap='RdYlGn_r',
                         alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    ax4.set_title('Temperature Correlation', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Air Temperature [K]', fontsize=11)
    ax4.set_ylabel('Process Temperature [K]', fontsize=11)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Failure', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Rotational speed distribution by failure
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(df[df['Machine failure']==0]['Rotational speed [rpm]'],
             bins=50, alpha=0.7, label='No Failure', color='#2ECC71', edgecolor='black')
    ax5.hist(df[df['Machine failure']==1]['Rotational speed [rpm]'],
             bins=50, alpha=0.7, label='Failure', color='#E74C3C', edgecolor='black')
    ax5.set_title('Rotational Speed Distribution', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Rotational Speed [rpm]', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Failure rate by product type
    ax6 = fig.add_subplot(gs[1, 2])
    failure_rates = []
    for pt in product_types:
        rate = (df[df['Type']==pt]['Machine failure'].sum() / len(df[df['Type']==pt]) * 100)
        failure_rates.append(rate)

    bars = ax6.bar(product_types, failure_rates, color=COLORS[:3],
                   edgecolor='black', linewidth=1.2)
    ax6.set_title('Failure Rate by Product Type', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Failure Rate (%)', fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, rate in zip(bars, failure_rates):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.savefig('06_failure_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: 06_failure_analysis.png")

def plot_violin_plots(df):
    """Create beautiful violin plots"""
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]',
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Violin Plots: Feature Distributions by Failure Status',
                 fontsize=20, fontweight='bold')
    axes = axes.flatten()

    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]

        # Create violin plot
        parts = ax.violinplot([df[df['Machine failure']==0][col],
                               df[df['Machine failure']==1][col]],
                              positions=[1, 2],
                              showmeans=True,
                              showmedians=True,
                              widths=0.7)

        # Color the violins
        colors = ['#2ECC71', '#E74C3C']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        # Styling
        ax.set_title(col, fontsize=13, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['No Failure', 'Failure'])
        ax.grid(True, alpha=0.3, axis='y')

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig('07_violin_plots.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: 07_violin_plots.png")

def plot_3d_analysis(df):
    """Create 3D scatter plots for multi-dimensional analysis"""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('3D Feature Space Analysis', fontsize=20, fontweight='bold')

    # Sample for better visualization
    df_sample = df.sample(n=min(3000, len(df)), random_state=42)

    # Plot 1: Temperature, Speed, Torque
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(df_sample['Air temperature [K]'],
                           df_sample['Rotational speed [rpm]'],
                           df_sample['Torque [Nm]'],
                           c=df_sample['Machine failure'],
                           cmap='RdYlGn_r',
                           alpha=0.6,
                           s=20)
    ax1.set_xlabel('Air Temp [K]', fontsize=9)
    ax1.set_ylabel('Rotation [rpm]', fontsize=9)
    ax1.set_zlabel('Torque [Nm]', fontsize=9)
    ax1.set_title('Temperature-Speed-Torque', fontsize=12, fontweight='bold')

    # Plot 2: Process temp, Tool wear, Speed
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(df_sample['Process temperature [K]'],
                           df_sample['Tool wear [min]'],
                           df_sample['Rotational speed [rpm]'],
                           c=df_sample['Machine failure'],
                           cmap='RdYlGn_r',
                           alpha=0.6,
                           s=20)
    ax2.set_xlabel('Process Temp [K]', fontsize=9)
    ax2.set_ylabel('Tool Wear [min]', fontsize=9)
    ax2.set_zlabel('Rotation [rpm]', fontsize=9)
    ax2.set_title('Process-Wear-Speed', fontsize=12, fontweight='bold')

    # Plot 3: Torque, Tool wear, Air temp
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(df_sample['Torque [Nm]'],
                           df_sample['Tool wear [min]'],
                           df_sample['Air temperature [K]'],
                           c=df_sample['Machine failure'],
                           cmap='RdYlGn_r',
                           alpha=0.6,
                           s=20)
    ax3.set_xlabel('Torque [Nm]', fontsize=9)
    ax3.set_ylabel('Tool Wear [min]', fontsize=9)
    ax3.set_zlabel('Air Temp [K]', fontsize=9)
    ax3.set_title('Torque-Wear-Temperature', fontsize=12, fontweight='bold')

    # Add colorbar
    fig.colorbar(scatter3, ax=[ax1, ax2, ax3], label='Machine Failure',
                 shrink=0.6, pad=0.1)

    plt.savefig('08_3d_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: 08_3d_analysis.png")

def plot_time_series_simulation(df):
    """Create time-series style visualization"""
    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    fig.suptitle('Sequential Data Patterns (First 1000 samples)',
                 fontsize=20, fontweight='bold')

    # Take first 1000 samples to simulate time series
    df_subset = df.head(1000).copy()
    df_subset['Index'] = range(len(df_subset))

    # Plot 1: Temperatures over index
    ax1 = axes[0]
    ax1.plot(df_subset['Index'], df_subset['Air temperature [K]'],
             label='Air Temp', color='#FF6B6B', linewidth=1.5, alpha=0.8)
    ax1.plot(df_subset['Index'], df_subset['Process temperature [K]'],
             label='Process Temp', color='#4ECDC4', linewidth=1.5, alpha=0.8)

    # Highlight failures
    failure_indices = df_subset[df_subset['Machine failure']==1]['Index']
    for idx in failure_indices:
        ax1.axvline(x=idx, color='red', alpha=0.3, linewidth=2)

    ax1.set_ylabel('Temperature [K]', fontsize=12)
    ax1.set_title('Temperature Patterns', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mechanical parameters
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.plot(df_subset['Index'], df_subset['Rotational speed [rpm]'],
             label='Rotation Speed', color='#45B7D1', linewidth=1.5, alpha=0.8)
    ax2_twin.plot(df_subset['Index'], df_subset['Torque [Nm]'],
                  label='Torque', color='#FFA07A', linewidth=1.5, alpha=0.8)

    # Highlight failures
    for idx in failure_indices:
        ax2.axvline(x=idx, color='red', alpha=0.3, linewidth=2)

    ax2.set_ylabel('Rotational Speed [rpm]', fontsize=12, color='#45B7D1')
    ax2_twin.set_ylabel('Torque [Nm]', fontsize=12, color='#FFA07A')
    ax2.set_title('Mechanical Parameters', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Tool wear and failures
    ax3 = axes[2]
    ax3.fill_between(df_subset['Index'], df_subset['Tool wear [min]'],
                     alpha=0.5, color='#98D8C8', label='Tool Wear')
    ax3.plot(df_subset['Index'], df_subset['Tool wear [min]'],
             color='#2C3E50', linewidth=2)

    # Mark failures
    failure_data = df_subset[df_subset['Machine failure']==1]
    ax3.scatter(failure_data['Index'], failure_data['Tool wear [min]'],
                color='red', s=100, marker='X', label='Failure Event',
                edgecolors='black', linewidth=1.5, zorder=5)

    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.set_ylabel('Tool Wear [min]', fontsize=12)
    ax3.set_title('Tool Wear Progression & Failure Events', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('09_sequential_patterns.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: 09_sequential_patterns.png")

def create_summary_dashboard(df):
    """Create a comprehensive summary dashboard"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    fig.suptitle('FogBayes Dataset: Complete Summary Dashboard',
                 fontsize=22, fontweight='bold', y=0.98)

    # 1. Key Metrics
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')
    metrics = {
        'Total Samples': f"{len(df):,}",
        'Total Failures': f"{df['Machine failure'].sum():,}",
        'Failure Rate': f"{(df['Machine failure'].sum()/len(df)*100):.2f}%",
        'Product Types': len(df['Type'].unique()),
        'Features': len(df.columns),
    }

    y_pos = 0.8
    for key, value in metrics.items():
        ax1.text(0.2, y_pos, key + ':', fontsize=14, fontweight='bold')
        ax1.text(0.6, y_pos, str(value), fontsize=14, color='#E74C3C')
        y_pos -= 0.15

    ax1.set_title('Key Dataset Metrics', fontsize=16, fontweight='bold', pad=10)

    # 2. Failure types pie
    ax2 = fig.add_subplot(gs[0, 2:])
    failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    failure_counts = [df[ft].sum() for ft in failure_types]
    wedges, texts, autotexts = ax2.pie(failure_counts, labels=failure_types,
                                         autopct='%1.1f%%', colors=COLORS,
                                         explode=[0.05]*5, shadow=True)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
    ax2.set_title('Failure Type Distribution', fontsize=14, fontweight='bold')

    # 3-6. Small distribution plots
    numerical_cols = ['Air temperature [K]', 'Rotational speed [rpm]',
                      'Torque [Nm]', 'Tool wear [min]']

    for idx, col in enumerate(numerical_cols):
        ax = fig.add_subplot(gs[1, idx])
        ax.hist(df[col], bins=40, color=COLORS[idx], alpha=0.7, edgecolor='black')
        ax.set_title(col.split('[')[0].strip(), fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    # 7. Correlation mini heatmap
    ax7 = fig.add_subplot(gs[2, :2])
    corr_cols = ['Air temperature [K]', 'Rotational speed [rpm]',
                 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax7)
    ax7.set_title('Feature Correlations', fontsize=12, fontweight='bold')

    # 8. Failure by product type
    ax8 = fig.add_subplot(gs[2, 2])
    product_failure = df.groupby('Type')['Machine failure'].agg(['sum', 'count'])
    product_failure['rate'] = (product_failure['sum'] / product_failure['count'] * 100)
    bars = ax8.bar(product_failure.index, product_failure['rate'],
                   color=COLORS[:3], edgecolor='black', linewidth=1.2)
    ax8.set_title('Failure Rate by Type', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Failure Rate (%)', fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

    # 9. Feature importance proxy
    ax9 = fig.add_subplot(gs[2, 3])
    feature_corr = df[corr_cols[:-1]].corrwith(df['Machine failure']).abs().sort_values(ascending=True)
    bars = ax9.barh(range(len(feature_corr)), feature_corr.values, color=COLORS)
    ax9.set_yticks(range(len(feature_corr)))
    ax9.set_yticklabels([col.split('[')[0].strip() for col in feature_corr.index], fontsize=9)
    ax9.set_title('Feature-Target Correlation', fontsize=12, fontweight='bold')
    ax9.set_xlabel('|Correlation|', fontsize=10)
    ax9.grid(True, alpha=0.3, axis='x')

    plt.savefig('10_summary_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: 10_summary_dashboard.png")

def main():
    """Main execution function"""
    print("="*60)
    print("FogBayes Dataset - Beautiful Data Visualization")
    print("="*60)
    print("\nLoading dataset...")

    df = load_data()
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    print("Generating visualizations...\n")

    # Generate all visualizations
    create_overview_stats(df)
    plot_distributions(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    plot_pairplot(df)
    plot_failure_analysis(df)
    plot_violin_plots(df)
    plot_3d_analysis(df)
    plot_time_series_simulation(df)
    create_summary_dashboard(df)

    print("\n" + "="*60)
    print("All visualizations created successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  01_dataset_overview.png       - Dataset statistics & overview")
    print("  02_feature_distributions.png  - Feature distribution plots")
    print("  03_feature_boxplots.png       - Box plots by failure status")
    print("  04_correlation_heatmap.png    - Feature correlation matrix")
    print("  05_pairplot.png               - Pairwise relationships")
    print("  06_failure_analysis.png       - Detailed failure analysis")
    print("  07_violin_plots.png           - Violin plots by failure")
    print("  08_3d_analysis.png            - 3D feature space analysis")
    print("  09_sequential_patterns.png    - Sequential data patterns")
    print("  10_summary_dashboard.png      - Complete summary dashboard")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
