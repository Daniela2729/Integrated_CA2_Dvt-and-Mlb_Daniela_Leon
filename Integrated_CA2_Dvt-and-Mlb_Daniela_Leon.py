import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import time
from collections import Counter
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Market Basket", layout="wide")
st.title("Integrated e-Commerce Dashboard")
st.markdown("This dashboard shows the main business insights in a simple, user-friendly interface.", unsafe_allow_html=True)


df1 = pd.read_csv("Online-eCommerce.csv")
df2 = pd.read_csv("products.csv")


df1['Product'] = df1['Product'].astype(str).str.strip().str.lower()
df1['Category'] = df1['Category'].astype(str).str.strip().str.lower()
df1['Brand'] = df1['Brand'].astype(str).str.strip().str.lower()
df1['Customer_Name'] = df1['Customer_Name'].astype(str).str.strip()
df1 = df1[['Customer_Name', 'Product', 'Category', 'Brand', 'Quantity', 'Total_Sales']].copy()
df1.dropna(inplace=True)


col1, col2, col3 = st.columns(3)
col1.metric("Total de Ventas", f"${df1['Total_Sales'].sum():,.0f}")
col2.metric("Productos únicos", df1['Product'].nunique())
col3.metric("Categorías", df1['Category'].nunique())


cat_sales = df1.groupby("Category")["Quantity"].sum().reset_index()
st.subheader("Top Selling Categories (Interactive)")
fig = px.bar(
cat_sales,
x="Category",
y="Quantity",
color="Quantity",
hover_data=["Quantity"],
labels={"Quantity":"Cantidad vendida","Category":"Categoría"},
title="Ventas por Categoría"
)
st.plotly_chart(fig, use_container_width=True)


st.subheader("Sales per Customer (Interactive)")
customer_sales = df1.groupby('Customer_Name')['Total_Sales'].sum().reset_index()
fig = px.scatter(
customer_sales,
x="Customer_Name",
y="Total_Sales",
size="Total_Sales",
color="Total_Sales",
hover_data=["Total_Sales"],
title="Ventas Totales por Cliente"
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)


check_final = df1.pivot_table(index='Customer_Name', columns='Category', values='Quantity')
check_final = check_final.apply(lambda row: row.fillna(row.mean()), axis=1)
similarity_with_user = pd.DataFrame(cosine_similarity(check_final), index=check_final.index, columns=check_final.index)

def find_n_neighbours_user_ids(similarity_with_user, n):
top_users = similarity_with_user.apply(
lambda row: pd.Series(
row.sort_values(ascending=False).iloc[1:n+1].index,
index=['top{}'.format(i) for i in range(1, n+1)]
),
axis=1
)
return top_users

top_n_users = 30
sim_user_30_u = find_n_neighbours_user_ids(similarity_with_user, top_n_users)

def User_item_score1(user, top_n=5):
score = []
for item in check_final.columns:
neighbors = sim_user_30_u.loc[user].values.squeeze().tolist()
item_ratings = check_final.loc[neighbors, item]
item_ratings = item_ratings[item_ratings.notnull()]
avg_user = check_final.loc[user].mean()
if len(item_ratings) == 0:
final_score = avg_user
else:
index = item_ratings.index.values.squeeze().tolist() if len(item_ratings) > 1 else [item_ratings.index[0]]
corr = similarity_with_user.loc[user, index]
fin = pd.concat([item_ratings, corr], axis=1)
fin.columns = ['adg_score','correlation']
fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
final_score = avg_user + fin['score'].sum() / fin['correlation'].sum()
score.append(final_score)
data = pd.DataFrame({'Category': check_final.columns, 'Score': score})
return data.sort_values(by='Score', ascending=False).head(top_n)


st.subheader("Personalized recommendation (User–Item)")
usuarios = check_final.index.tolist()
usuario_sel = st.selectbox("Usuario:", usuarios, key="user_select")

if st.button("Generate recommendation"):
recomendaciones_user_df = User_item_score1(usuario_sel)
st.success(f"Top 5 categorías recomendadas para {usuario_sel}:")
for cat in recomendaciones_user_df['Category']:
st.write(f"- {cat}")

```
fig = px.bar(
    recomendaciones_user_df,
    x="Category",
    y="Score",
    text="Category",
    color="Score
```
