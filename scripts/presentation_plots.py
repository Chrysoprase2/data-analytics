import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import os

def create_presentation_plots():
    # Setup directories
    output_dir = 'results/plots/presentation/'
    os.makedirs(output_dir, exist_ok=True)

    # Load Data
    classes_df = pd.read_csv('data/raw/classes_boroughs_total.csv')
    professions_df = pd.read_csv('data/raw/professions_total.csv')

    # Colors (Booth-Palette)
    color_poverty = '#0a1d37' # Dark Blue/Black
    color_above = '#d69f4c' # Gold
    color_highlight = '#b53131' # Red

    # -------------------------------------------------------------
    # 1. Das Londoner Armuts-Rad (Donut Chart)
    # -------------------------------------------------------------
    total_poverty = classes_df[['A', 'B', 'C', 'D']].sum().sum()
    total_above = classes_df[['E', 'F', 'G', 'H']].sum().sum()

    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = [total_poverty, total_above]
    labels = ['In Poverty (A-D)', 'Above Poverty (E-H)']
    colors = [color_poverty, color_above]

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=140, pctdistance=0.85,
                                      textprops={'fontsize': 14, 'weight': 'bold'},
                                      wedgeprops=dict(width=0.3, edgecolor='w'))

    for text in texts:
        text.set_fontsize(16)

    ax.set_title('Das Londoner Armuts-Rad', fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'poverty_donut.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 2. Die Job-Landschaft (Treemap)
    # -------------------------------------------------------------
    # Filter professions_total to exclude the grand total row
    job_df = professions_df.dropna(subset=['Description']).copy()
    job_df = job_df[job_df['Description'] != 'Total']

    # Fill class if needed
    job_df['Class'] = job_df['Class'].ffill()

    # Wir brauchen die Verteilung der 'Heads of Families' nach Sektionen (Description)
    job_df['Heads of Famlies'] = pd.to_numeric(job_df['Heads of Famlies'], errors='coerce').fillna(0)
    job_df = job_df[job_df['Heads of Famlies'] > 0]

    # Grouping to avoid too many small boxes (Optional: taking top 20 or grouping)
    job_data = job_df.sort_values(by='Heads of Famlies', ascending=False)

    plt.figure(figsize=(16, 10))
    cmap = plt.cm.get_cmap('YlOrRd')
    mini = min(job_data['Heads of Famlies'])
    maxi = max(job_data['Heads of Famlies'])
    norm = plt.Normalize(vmin=mini, vmax=maxi)
    colors = [cmap(norm(value)) for value in job_data['Heads of Famlies']]

    squarify.plot(sizes=job_data['Heads of Famlies'],
                  label=job_data['Description'],
                  alpha=0.8,
                  color=colors,
                  text_kwargs={'fontsize':9, 'weight':'bold', 'wrap':True})
    plt.title('Die Job-Landschaft (Heads of Families)', fontsize=24, weight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'job_landscape_treemap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 3. Vergleich der Lebenswelten (Pictogram-Style Bar Chart)
    # -------------------------------------------------------------
    # IDs in the raw file are in the 'Unnamed: 1' column which has index 1.
    id_col = professions_df.columns[1]

    # Clean up IDs
    prof_df_clean = professions_df.copy()
    prof_df_clean[id_col] = pd.to_numeric(prof_df_clean[id_col], errors='coerce')

    labour_ids = [1, 2, 3, 4, 5]
    salaried_ids = [28, 29, 30]

    labour_df = prof_df_clean[prof_df_clean[id_col].isin(labour_ids)]
    salaried_df = prof_df_clean[prof_df_clean[id_col].isin(salaried_ids)]

    labour_kids = pd.to_numeric(labour_df['Children -15'], errors='coerce').sum()
    labour_heads = pd.to_numeric(labour_df['Heads of Famlies'], errors='coerce').sum()
    labour_avg = labour_kids / labour_heads if labour_heads > 0 else 0

    salaried_kids = pd.to_numeric(salaried_df['Children -15'], errors='coerce').sum()
    salaried_heads = pd.to_numeric(salaried_df['Heads of Famlies'], errors='coerce').sum()
    salaried_avg = salaried_kids / salaried_heads if salaried_heads > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    groups = ['Arbeiterfamilie\n(Labour)', 'Angestelltenfamilie\n(Salaried)']
    avgs = [labour_avg, salaried_avg]
    colors = [color_poverty, color_above]

    bars = ax.barh(groups, avgs, color=colors, height=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.2f} Kinder',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(10, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=14, weight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.tick_params(axis='y', labelsize=16)

    plt.title('Vergleich der Lebenswelten: Kinder pro Haushalt', fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'living_environments_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 4. Die 'unsichtbaren' Frauen (Horizontal Bar Chart)
    # -------------------------------------------------------------
    female_ids = [33, 34, 35, 36, 37, 38]
    female_df = prof_df_clean[prof_df_clean[id_col].isin(female_ids)].copy()
    female_df['Heads of Famlies'] = pd.to_numeric(female_df['Heads of Famlies'], errors='coerce').fillna(0)
    female_df = female_df.sort_values(by='Heads of Famlies', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(female_df['Description'], female_df['Heads of Famlies'], color=color_highlight)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{int(width):,}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Anzahl der weiblichen Haushaltsvorstände', fontsize=14)

    plt.title('Die "unsichtbaren" Frauen (Alleinverdiener)', fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'invisible_women_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 5. Stadt der Kontraste (Gauge/Tacho-Chart Alternative)
    # -------------------------------------------------------------
    # To create a high-contrast comparison (St George vs Hackney)
    st_george_row = classes_df[classes_df['Borough'] == "St George's-in-the-East"]
    hackney_row = classes_df[classes_df['Borough'] == 'Hackney']

    def get_pov_rate(row):
        pov = row[['A', 'B', 'C', 'D']].sum(axis=1).values[0]
        tot = row['Total'].values[0]
        return pov / tot * 100

    rate_st_george = get_pov_rate(st_george_row)
    rate_hackney = get_pov_rate(hackney_row)

    fig, ax = plt.subplots(figsize=(10, 6))

    # We will use a dual bar chart to represent the contrast clearly
    y_pos = np.arange(2)
    rates = [rate_st_george, rate_hackney]
    boroughs = ["St George's-in-the-East", "Hackney"]
    colors = [color_poverty, color_above]

    bars = ax.barh(y_pos, rates, color=colors, height=0.6)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.1f}% Armut',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(-10, 0),
                    textcoords="offset points",
                    ha='right', va='center', color='white', fontsize=16, weight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(boroughs, fontsize=16, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)

    plt.title('Stadt der Kontraste: Armutsquote', fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'city_contrasts_gauge.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Presentation plots created successfully.")

if __name__ == "__main__":
    create_presentation_plots()
