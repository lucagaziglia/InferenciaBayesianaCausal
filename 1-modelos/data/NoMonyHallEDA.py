#EDA

#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

df = pd.read_csv("NoMontyHall.csv")
df.describe

#%%

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cols = ['c', 's', 'r']
titles = ['Caja elegida (c)', 'Caja señalada (s)', 'Caja con regalo (r)']

for i, col in enumerate(cols):
    sns.countplot(x=df[col], ax=axes[i])
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Número de caja')
    axes[i].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

#%%

df['c_equals_s'] = df['c'] == df['s']
df['c_equals_r'] = df['c'] == df['r']
df['s_equals_r'] = df['s'] == df['r']

coincidencias = df[['c_equals_s', 'c_equals_r', 's_equals_r']].melt()
sns.countplot(x='variable', hue='value', data=coincidencias)
plt.title('Coincidencias entre columnas')
plt.xlabel('Comparación')
plt.ylabel('Cantidad de casos')
plt.legend(title='¿Coinciden?')
plt.show()

#%%

pivot_c_s = df.pivot_table(index='c', columns='s', aggfunc='size', fill_value=0)

sns.heatmap(pivot_c_s, annot=True, fmt='d', cmap='Greens') 
plt.title('Frecuencia de combinaciones: Caja elegida (c) vs Señalada (s)')
plt.xlabel('Caja señalada')
plt.ylabel('Caja elegida')
plt.show()

#%%
import plotly.graph_objects as go
from collections import Counter

df['c_str'] = 'Elegida: ' + df['c'].astype(str)
df['s_str'] = 'Señalada: ' + df['s'].astype(str)
df['r_str'] = 'Regalo: ' + df['r'].astype(str)

links_cs = Counter(zip(df['c_str'], df['s_str']))
links_sr = Counter(zip(df['s_str'], df['r_str']))

orden_c = [f'Elegida: {i}' for i in range(3)]
orden_s = [f'Señalada: {i}' for i in range(3)]
orden_r = [f'Regalo: {i}' for i in range(3)]
labels = orden_c + orden_s + orden_r 

label_map = {label: i for i, label in enumerate(labels)}

source = []
target = []
value = []

for (src, tgt), val in list(links_cs.items()) + list(links_sr.items()):
    source.append(label_map[src])
    target.append(label_map[tgt])
    value.append(val)

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",  
    node=dict(label=labels, pad=20, thickness=20),
    link=dict(source=source, target=target, value=value)
)])

fig.update_layout(title_text="Flujo de cajas: Elegida → Señalada → Regalo", font_size=12)
fig.show()

# %%
