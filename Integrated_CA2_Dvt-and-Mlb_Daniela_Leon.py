import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
plt.style.use("default")

st.title("Integrated-ca2-dvt-and-mlb")
st.title("E-Commerce Dashboard")
st.write(
"""This dashboard displays the most important business information clearly and simply.
It is designed so that anyone can easily understand sales, products, and customer behavior."""
)

# Load data

df1 = pd.read_csv("Online-eCommerce.csv")

# Data cleaning

df1['Product'] = df1['Product'].astype(str).str.strip().str.lower()
df1['Category'] = df1['Category'].astype(str).str.strip().str.lower()
df1['Brand'] = df1['Brand'].astype(str).str.strip().str.lower()
df1['Customer_Name'] = df1['Customer_Name'].astype(str).str.strip()

# KPI metrics

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${df1['Total_Sales'].sum():,.0f}")
col2.metric("Unique Products", df1['Product'].nunique())
col3.metric("Categories", df1['Category'].nunique())



cat_sales = df1.groupby("Category")["Quantity"].sum().reset_index()
st.subheader("Top Selling Categories (Interactive)")
fig = px.bar(
    cat_sales,
    x="Category",
    y="Quantity",
    color="Quantity",
    hover_data=["Quantity"],
    labels={"Quantity": "Quantity Sold", "Category": "Category"},
    title="Sales by Category"
)
st.plotly_chart(fig)


st.subheader("Sales per Customer (Interactive)")
customer_sales = df1.groupby('Customer_Name')['Total_Sales'].sum().reset_index()
fig = px.scatter(
    customer_sales,
    x="Customer_Name",
    y="Total_Sales",
    size="Total_Sales",
    color="Total_Sales",
    hover_data=["Total_Sales"],
    title="Total Sales per Customer"
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)


df1 = df1[['Customer_Name', 'Product', 'Category', 'Brand', 'Quantity', 'Total_Sales']].copy()
df1.dropna(inplace=True)


check_final = df1.pivot_table(index='Customer_Name', columns='Category', values='Quantity')
check_final = check_final.apply(lambda row: row.fillna(row.mean()), axis=1)

similarity_with_user = pd.DataFrame(
    cosine_similarity(check_final),
    index=check_final.index,
    columns=check_final.index
)

def find_n_neighbours_user_ids(similarity_with_user, n):
    top_users = similarity_with_user.apply(
        lambda row: pd.Series(
            row.sort_values(ascending=False).iloc[1:n+1].index,
            index=['top{}'.format(i) for i in range(1, n+1)]
        ),
        axis=1
    )
    return top_users

top_n = 30
sim_user_30_u = find_n_neighbours_user_ids(similarity_with_user, top_n)

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
            fin.columns = ['adg_score', 'correlation']
            fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
            final_score = avg_user + fin['score'].sum() / fin['correlation'].sum()
        score.append(final_score)
    data = pd.DataFrame({'Category': check_final.columns, 'Score': score})
    return data.sort_values(by='Score', ascending=False).head(top_n)


st.subheader("Personalized Recommendation (User–Item)")
users = check_final.index.tolist()
selected_user = st.selectbox("Select User:", users, key="user_select")

if st.button("Generate Recommendation"):
    user_recommendations = User_item_score1(selected_user)
    st.success(f"Top 5 recommended categories for **{selected_user}**:")
    
    for cat in user_recommendations['Category']:
        st.write(f"- {cat}")
    
    fig = px.bar(
        user_recommendations,
        x="Category",
        y="Score",
        color="Score",
        labels={"Score": "Predicted Score", "Category": "Category"},
        title=f"Top 5 recommended categories for '{selected_user}'",
        color_continuous_scale="Viridis"
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)



item_sim = cosine_similarity(check_final.T)
similarity_with_item = pd.DataFrame(item_sim, index=check_final.columns, columns=check_final.columns)

def find_n_neighbours_item(similarity_with_item, n):
    top_items = similarity_with_item.apply(
        lambda row: pd.Series(
            row.sort_values(ascending=False).iloc[1:n+1].index,
            index=[f'top_item_{i}' for i in range(1, n+1)]
        ),
        axis=1
    )
    return top_items

top_n_items = 12
sim_items_12_u = find_n_neighbours_item(similarity_with_item, top_n_items)

def Item_item_score1(top_n=5):
    score = []
    for item_actual in check_final.columns:
        neighbors = sim_items_12_u.loc[item_actual].dropna().values.squeeze().tolist()
        if len(neighbors) == 0:
            final_score = 0
        else:
            corr = similarity_with_item.loc[item_actual, neighbors]
            adg_score = pd.Series(1, index=neighbors)
            fin = pd.concat([adg_score, corr], axis=1)
            fin.columns = ['adg_score', 'correlation']
            fin['score'] = fin['adg_score'] * fin['correlation']
            final_score = fin['score'].sum() / fin['correlation'].sum()
        score.append(final_score)
    data = pd.DataFrame({'Category': check_final.columns, 'score': score})
    top_items = data.sort_values(by='score', ascending=False).head(top_n)
    return top_items['Category'].tolist()

st.subheader("Item-Based Recommendation (Item-Item)")
items = sorted(check_final.columns.tolist())
selected_item = st.selectbox("Select a Category:", items, key="item_select")

if st.button("Show Similar Items"):
    item_recommendations = Item_item_score1(top_n=5)
    st.success(f"Top 5 categories similar to **{selected_item}**:")
    for cat in item_recommendations:
        st.write(f"- {cat}")



df2 = pd.read_csv("products.csv")
df_sample = df2.sample(n=3000, random_state=42)
df_sample['Products'] = df_sample['Products'].apply(
lambda x: [p.strip() for p in x.split(',')] if isinstance(x, str) else []
)

all_products = [p for sublist in df_sample['Products'] for p in sublist if p]

product_counts = Counter(all_products)
df_products = pd.DataFrame(product_counts.items(), columns=['Product', 'Count'])

top_n = st.slider("Select the number of products to display:", 5, 20, 10)
order = st.radio("Sort products by:", ['Count descending', 'Alphabetical'], index=0)

if order == 'Count descending':
    df_plot = df_products.sort_values(by='Count', ascending=False).head(top_n)
else:
    df_plot = df_products.sort_values(by='Product').head(top_n)

fig = px.bar(
    df_plot,
    x='Product',
    y='Count',
    text='Count',
    color='Count',
    color_continuous_scale='Viridis',
    title=f"Top {top_n} Best-Selling Products",
    hover_data={'Product': True, 'Count': True}
)

fig.update_layout(
    font=dict(size=18),
    title_font_size=24,
    xaxis_tickangle=-45,
    xaxis_title="Product",
    yaxis_title="Quantity Sold",
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig.update_traces(
    textposition='outside',
    cliponaxis=False,
    textfont_size=16
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "Each bar shows how many times each product was sold. Darker colors indicate higher sales.",
    unsafe_allow_html=True
)






