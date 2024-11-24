import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 Importar archivo
df = pd.read_csv('medical_examination.csv')
print(df.head(10))

#Revisar datos
print("info: ", df.info())
print("Describe: ", df.describe())
print("columns: ", df.columns)
print("DATOS NULOS: ", df.isnull().any())

# 2 calcualr IMC agregando nueca columna overweight
df['overweight'] = ((df['weight']/(df['height']/100)**2)>25).astype(int)
print("df['overweight']: ", df['overweight'])

# 3: Normalizar los datos para cholesterol y gluc
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4: Grafico Categórico
def draw_cat_plot():
    # 5: crear un df para el grafico categorico usando pd.melt (esto transforma a formato long)
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )
   
    # 6: Renombrar las columnas para que seaborn.catplot funcione correctamente
    df_cat = df_cat.rename(columns={'value': 'value'})
    # 7: Agrupar los datos por cardio, característica y valor
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')


    # 8
    fig = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar',
        height=5,
        aspect=1.2
    ).fig


    # 9
    fig.savefig('catplot.png')
    return fig


# Grafico Mapa de Calor
def draw_heat_map():
    # 11 Limpiar los datos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12: Matriz de correlación
    corr = df_heat.corr()

    # 13: Generar una máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))


    # 14: Configurar la figura de matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15 Dibujar el mapa de calor
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap='coolwarm',
        square=True,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )


    # 16
    fig.savefig('heatmap.png')
    return fig