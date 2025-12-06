

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import plotly.express as px

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

import warnings
warnings.filterwarnings('ignore')

plt.style.use("default")

st.title("Integrated-ca2-dvt-and-mlb")


st.title("e-Commerce Dashboard")
st.write("""This dashboard displays the most important business information clearly and simply.

It's designed so that anyone can easily understand sales, products, and customer behavior.""")


# In[3]:


df1 = pd.read_csv("Online-eCommerce.csv")
df1.head()

# In[12]:


df1['Product'] = df1['Product'].astype(str).str.strip().str.lower()
df1['Category'] = df1['Category'].astype(str).str.strip().str.lower()
df1['Brand'] = df1['Brand'].astype(str).str.strip().str.lower()
df1['Customer_Name'] = df1['Customer_Name'].astype(str).str.strip()


# In[13]:


st.subheader("Top Selling Categories (Total Amount)")

col1, col2, col3 = st.columns(3)

col1.metric("Total de Ventas", f"${df1['Total_Sales'].sum():,.0f}")
col2.metric("Productos únicos", df1['Product'].nunique())
col3.metric("Categorías", df1['Category'].nunique())

cat_sales = df1.groupby("Category")["Quantity"].sum().reset_index()



st.subheader("Top Selling Categories (Interactive)")

fig = px.bar(cat_sales, 
             x="Category", 
             y="Quantity", 
             color="Quantity",
             hover_data=["Quantity"],
             labels={"Quantity":"Cantidad vendida","Category":"Categoría"},
             title="Ventas por Categoría")
st.plotly_chart(fig)


st.subheader("Sales per Customer (Interactive)")

customer_sales = df1.groupby('Customer_Name')['Total_Sales'].sum().reset_index()
fig = px.scatter(customer_sales, 
                 x="Customer_Name", 
                 y="Total_Sales",
                 size="Total_Sales",
                 color="Total_Sales",
                 hover_data=["Total_Sales"],
                 title="Ventas Totales por Cliente")
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)


df1 = df1[['Customer_Name', 'Product', 'Category', 'Brand', 'Quantity', 'Total_Sales']].copy()

df1.dropna(inplace=True)

df1['Customer_Name'] = df1['Customer_Name'].astype(str)
df1['Product'] = df1['Product'].astype(str)
df1['Category'] = df1['Category'].astype(str)
df1['Brand'] = df1['Brand'].astype(str)


# Collaborative Filtering
# 
# User-User

# In[15]:


Check = df1.pivot_table( index='Customer_Name',
                        columns='Category',
                        values='Quantity',
                        )

Check.head()


# In[16]:


check_final = Check.fillna(Check.mean(axis=0))
check_final = Check.apply(lambda row: row.fillna(row.mean()), axis=1)
check_final.head()


# In[17]:


user= cosine_similarity(check_final)
similarity_with_user= pd.DataFrame(user,
                                  index=check_final.index,
                                  columns=check_final.index)

similarity_with_user.head()


# In[18]:


def find_n_neighbours_user_ids(similarity_with_user, n):
    top_users = similarity_with_user.apply(
        lambda row: pd.Series(
            row.sort_values(ascending=False).iloc[1:n+1].index,
            index=['top{}'.format(i) for i in range(1, n+1)]
        ),
        axis=1
    )
    return top_users


# In[19]:


top_n = 30
sim_user_30_u = find_n_neighbours_user_ids(similarity_with_user, top_n)
sim_user_30_u.head()


# In[20]:


def get_user_common_items(user1, user2):
    common_items = df1[df1['Customer_Name'] == user1].merge(
        df1[df1['Customer_Name'] == user2],
        on="Category",
        how="inner"
    )

    return common_items


# In[21]:


a = get_user_common_items("Adhir Samal", "Ajay Mehta")

a = a.loc[:, ['Quantity_x', 'Quantity_y', 'Category']]

a.head()


# In[22]:


def User_item_score(user, item):
  
    a = sim_user_30_u[sim_user_30_u.index == user].values
    b = a.squeeze().tolist()
    c = check_final.loc[:, item]
    d = c[c.index.isin(b)]
    f = d[d.notnull()]
    
    avg_user = check_final.loc[user].mean()
    index = f.index.values.squeeze().tolist()
    corr = similarity_with_user.loc[user, index]
    fin = pd.concat([f, corr], axis=1)
    fin.columns = ['adg_score', 'correlation']
    fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
    nume = fin['score'].sum()
    deno = fin['correlation'].sum()
    final_score = avg_user + (nume / deno)
    
    return final_score


# In[23]:


score = User_item_score("Adhir Samal", "cpu")
print("score (u,i) is", score)


# In[24]:


check_final_str = check_final.astype(str)


# In[25]:


Customer_items = check_final_str.apply(lambda row: ','.join(row[row.notnull()].index), axis=1)
Customer_items.head()


# In[26]:


def User_item_score1(user):
    score = []
    for item in check_final.columns:
        
        a = sim_user_30_u.loc[user].values
        neighbors = a.squeeze().tolist()
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

    data = pd.DataFrame({'Category': check_final.columns, 'score': score})
    top_5_recommendation = data.sort_values(by='score', ascending=False).head(5)
    top_categories = top_5_recommendation['Category'].tolist()
    return top_categories

st.subheader("Personalized recommendation (User–Item)")
st.write("Select a user to view recommended categories.")

usuarios = check_final.index.tolist()
usuario_sel = st.selectbox("Usuario:", usuarios, key="user_select")

if st.button("Generate recommendation"):
    try:

        score = []
        for item in check_final.columns:
            a = sim_user_30_u.loc[usuario_sel].values
            neighbors = a.squeeze().tolist()
            item_ratings = check_final.loc[neighbors, item]
            item_ratings = item_ratings[item_ratings.notnull()]

            avg_user = check_final.loc[usuario_sel].mean()

            if len(item_ratings) == 0:
                final_score = avg_user
            else:
                index = item_ratings.index.values.squeeze().tolist() if len(item_ratings) > 1 else [item_ratings.index[0]]
                corr = similarity_with_user.loc[usuario_sel, index]
                fin = pd.concat([item_ratings, corr], axis=1)
                fin.columns = ['adg_score','correlation']
                fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
                final_score = avg_user + fin['score'].sum() / fin['correlation'].sum()

            score.append(final_score)

        recomendaciones_user_df = pd.DataFrame({
            'Category': check_final.columns,
            'Score': score
        }).sort_values(by='Score', ascending=False).head(5)

        st.success(f"Top 5 categorías recomendadas para **{usuario_sel}**:")
        for cat in recomendaciones_user_df['Category']:
            st.write(f"- {cat}")

        fig = px.bar(
            recomendaciones_user_df,
            x="Category",
            y="Score",
            text="Category",
            color="Score",
            labels={"Score":"Predicted Score","Category":"Category"},
            title=f"Top 5 categorías recomendadas para '{usuario_sel}'",
            color_continuous_scale="Viridis"
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"No se pudo generar la recomendación para este usuario: {e}")


def User_item_score_selected(user, top_n=5):
    score = []
    for item in check_final.columns:
        a = sim_user_30_u.loc[user].values
        neighbors = a.squeeze().tolist()
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
    top_recommendations = data.sort_values(by='Score', ascending=False).head(top_n)
    return top_recommendations


# In[27]:

st.subheader("User-User Similarity Heatmap")

fig = px.imshow(similarity_with_user.values,
                labels=dict(x="Usuario", y="Usuario", color="Similitud"),
                x=similarity_with_user.columns,
                y=similarity_with_user.index,
                color_continuous_scale="Viridis")
st.plotly_chart(fig)

# Item-Item

# In[29]:


item_sim = cosine_similarity(check_final.T) 
similarity_with_item = pd.DataFrame(item_sim,
                                   index=check_final.columns,
                                   columns=check_final.columns)
similarity_with_item.head()


# In[30]:


def find_n_neighbours_item(similarity_with_item, n):

    top_items = similarity_with_item.apply(
        lambda row: pd.Series(
            row.sort_values(ascending=False).iloc[1:n+1].index,
            index=[f'top_item_{i}' for i in range(1, n+1)]
        ),
        axis=1
    )
    
    return top_items


# In[31]:


top_n_items = 12
sim_items_12_u = find_n_neighbours_item(similarity_with_item, top_n_items)
sim_items_12_u


# In[32]:


def Item_item_score(user, item):
    a = sim_items_12_u[sim_items_12_u.index == item].values
    b = a.squeeze().tolist()  
    c = check_final.loc[user, b] 
    f = c[c.notnull()]  
    
    avg_user = check_final.loc[user].mean()  
    index = f.index.values.squeeze().tolist() if len(f) > 1 else [f.index[0]]
    corr = similarity_with_item.loc[item, index]
    fin = pd.concat([f, corr], axis=1)
    fin.columns = ['adg_score', 'correlation']
    fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
    final_score = avg_user + (fin['score'].sum() / fin['correlation'].sum())
    
    return final_score


# In[33]:


score_item = Item_item_score("Adhir Samal", "cpu")
print("score (item-item) for user 'Adhir Samal' and item 'cpu' is", score_item)


# In[34]:


Item_neighbors_str = sim_items_12_u.apply(lambda row: ','.join(row[row.notnull()].astype(str)), axis=1)
Item_neighbors_str.head()


# In[35]:


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
            fin.columns = ['adg_score','correlation']
            fin['score'] = fin['adg_score'] * fin['correlation']
            final_score = fin['score'].sum() / fin['correlation'].sum()

        score.append(final_score)
    
    data = pd.DataFrame({'Category': check_final.columns, 'score': score}) 
    top_items = data.sort_values(by='score', ascending=False).head(top_n)
    
    return top_items['Category'].tolist()


# In[36]:


top_recommendations_items = Item_item_score1(top_n=5)
print("Top 5 recommended items (item-item):", top_recommendations_items)



# Model evaluation

# In[38]:


from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df1, test_size=0.2, random_state=42)

train_matrix = train_df.pivot_table(index='Customer_Name', columns='Category', values='Quantity')
test_matrix = test_df.pivot_table(index='Customer_Name', columns='Category', values='Quantity')


# RMSE/MAE

# In[40]:


pred_matrix_user = check_final.copy()

for user in check_final.index:
    neighbors = sim_user_30_u.loc[user].values.squeeze().tolist()
    
    for item in check_final.columns:
        neighbor_ratings = check_final.loc[neighbors, item].dropna()
        
        if len(neighbor_ratings) > 0:
            corr = similarity_with_user.loc[user, neighbor_ratings.index]
            pred_matrix_user.loc[user, item] = np.sum(neighbor_ratings * corr) / np.sum(corr)
        else:
            pred_matrix_user.loc[user, item] = check_final.loc[user].mean()


# In[41]:


true_values = []
pred_values = []

for user in test_matrix.index:
    if user in pred_matrix_user.index:
        for item in test_matrix.columns:
            if not np.isnan(test_matrix.loc[user, item]):
                true_values.append(test_matrix.loc[user, item])
                pred_values.append(pred_matrix_user.loc[user, item])

rmse_user = np.sqrt(mean_squared_error(true_values, pred_values))
mae_user = mean_absolute_error(true_values, pred_values)

print("User-User RMSE:", rmse_user)
print("User-User MAE:", mae_user)


# In[42]:


pred_matrix_item = check_final.copy()

for user in check_final.index:
    for item in check_final.columns:
        neighbors = sim_items_12_u.loc[item].dropna().values.squeeze().tolist()
        neighbor_ratings = check_final.loc[user, neighbors].dropna()
        if len(neighbor_ratings) > 0:
            corr = similarity_with_item.loc[item, neighbor_ratings.index]
            pred_matrix_item.loc[user, item] = (neighbor_ratings * corr).sum() / corr.sum()
        else:
            pred_matrix_item.loc[user, item] = check_final.loc[user].mean()


# In[43]:


true_values = []
pred_values = []

for user in test_matrix.index:
    if user in pred_matrix_item.index:
        for item in test_matrix.columns:
            if not np.isnan(test_matrix.loc[user, item]):
                true_values.append(test_matrix.loc[user, item])
                pred_values.append(pred_matrix_item.loc[user, item])

rmse_item = np.sqrt(mean_squared_error(true_values, pred_values))
mae_item = mean_absolute_error(true_values, pred_values)

print("RMSE:", rmse_item)
print("MAE:", mae_item)


# In[44]:


rmse_df = pd.DataFrame({'RMSE': [rmse_user, rmse_item]}, index=['User-User CF', 'Item-Item CF'])
rmse_df


# In[163]:


def precision_at_k(recommended, actual, k=5):
    recommended_k = recommended[:k]
    relevant = set(actual)
    recommended_set = set(recommended_k)
    return len(recommended_set & relevant) / len(recommended_set) if recommended_set else 0

def recall_at_k(recommended, actual, k=5):
    recommended_k = recommended[:k]
    relevant = set(actual)
    recommended_set = set(recommended_k)
    return len(recommended_set & relevant) / len(relevant) if relevant else 0



# In[170]:


user_test = "Ajay Sharma"
actual_products = check_final.loc[user_test][check_final.loc[user_test] > 0].index.tolist()

recommended_user = User_item_score1(user_test)
recommended_item = Item_item_score1(top_n=5)

metrics = {
    "User-User Precision@K": precision_at_k(recommended_user, actual_products),
    "User-User Recall@K": recall_at_k(recommended_user, actual_products),
    "Item-Item Precision@K": precision_at_k(recommended_item, actual_products),
    "Item-Item Recall@K": recall_at_k(recommended_item, actual_products)
}
metrics_df = pd.DataFrame(metrics, index=[0])
print(metrics_df)

st.subheader("Item-Based Recommendation (Item-Item)")
st.write("Relationship with other items.")


items = sorted(check_final.columns.tolist())
item_sel = st.selectbox("Select a category:", items, key="item_select")


if st.button("Show similar items"):
    try:

        recomendaciones_items = Item_item_score1(top_n=5) 

        st.success(f"Top 5 categories similar to **{item_sel}**:")


        for cat in recomendaciones_items:
            st.write(f"- {cat}")


        rec_items_df = pd.DataFrame({
            "Category": recomendaciones_items,
            "Score": range(len(recomendaciones_items), 0, -1)
        })

        fig = px.bar(
            rec_items_df,
            x="Category",
            y="Score",
            text="Category",
            color="Score",
            labels={"Score": "Similarity", "Category": "Category"},
            title=f"Top 5 categories similar to '{item_sel}'",
            color_continuous_scale="Viridis"
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"The item-item recommendation could not be generated: {e}")
# Part 2

# In[168]:


df2 = pd.read_csv("products.csv")
df2.head()


# In[48]:


df2.shape 


# In[49]:


duplicate_rows_df = df2[df2.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[50]:


df2.describe()


# In[51]:


df2.info()


# In[52]:


df2.dtypes


# Data clean

# In[54]:


df2.dtypes.value_counts()


# In[55]:


df2.isnull().sum()


# In[56]:


Missing_df=df2.isnull().sum()
Missing_df =Missing_df[Missing_df > 0]
Missing_df


# Part 1

# Key columns for analysis:
# 
# • TransactionID
# 
# • Product
# 
# • CustomerID 

# In[59]:


df_ana=df2.drop(columns=["Timestamp"])
df_ana.head()


# In[60]:


df_ana.dtypes


# In[61]:


df_ana['TransactionID'] = df_ana['TransactionID'].astype(str)
df_ana['CustomerID'] = df_ana['CustomerID'].astype(str)
df_ana['Products'] = df_ana['Products'].astype(str)

#df_ana['Products_List'] = df_ana['Products'].apply(lambda x: [p.strip() for p in x.split(',')])


# In[62]:


df_ana.dtypes


# In[63]:


df_ana.head()


# In[64]:


df_sample = df_ana.sample(n=3000, random_state=42)


# In[65]:


df_sample


# In[66]:


df_sample['Products'] = df_sample['Products'].apply(
    lambda x: [p.strip() for p in x.split(',')] if isinstance(x, str) else [])


# In[67]:


all_products = [p for sublist in df_sample['Products'] for p in sublist]
top_products = [p for p, _ in Counter(all_products).most_common(10)]

df_sample['Products'] = df_sample['Products'].apply(
    lambda x: [p for p in x if p in top_products])


# Apriori Algorithm

# In[69]:


basket = df_sample['Products'].tolist()

te = TransactionEncoder()
te_array = te.fit(basket).transform(basket)
df_both = pd.DataFrame(te_array, columns = te.columns_)
df_both


# In[70]:


frequent_itemsets_ap = apriori(df_both, min_support=0.01, use_colnames=True)


# In[71]:


print(frequent_itemsets_ap)


# In[72]:


frequent_itemsets_ap.shape


# In[73]:


rules= association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.1)


# In[74]:


rules = association_rules(frequent_itemsets_ap, metric="lift", min_threshold=0.1)


# In[75]:


rules[['lift', 'confidence']].describe()


# In[76]:


filtered_rules = rules[
    (rules['lift'] >= 1.2) &
    (rules['confidence'] >= 0.25)]
filtered_rules


# FP Growth Algorithm

# In[78]:


frequent_itemsets_fp = fpgrowth(df_both, min_support = 0.01, use_colnames = True)
print(frequent_itemsets_fp)


# In[79]:


rules_fp = association_rules(frequent_itemsets_fp, metric = "confidence", min_threshold = 0.1)


# In[80]:


rules_fp = association_rules(frequent_itemsets_fp, metric = "lift", min_threshold = 0.1)


# In[81]:


rules[['lift', 'confidence']].describe()


# In[82]:


filtered_rules = rules[
    (rules['lift'] >= 1.2) &
    (rules['confidence'] >= 0.25)]
filtered_rules


# In[83]:


start = time.time()
frequent_itemsets_ap = apriori(df_both, min_support=0.01, use_colnames=True)
apriori_time = time.time() - start

start = time.time()
frequent_itemsets_fp = fpgrowth(df_both, min_support=0.01, use_colnames=True)
fpgrowth_time = time.time() - start


# In[84]:


print("=== Apriori ===")
print(f"Number of frequent itemsets: {len(frequent_itemsets_ap)}")
print(f"Number of rules generated: {len(rules)}")
print(f"Execution time: {apriori_time:.4f} s\n")

print("=== FP-Growth ===")
print(f"Number of frequent itemsets: {len(frequent_itemsets_fp)}")
print(f"Number of rules generated: {len(rules_fp)}")
print(f"Execution time: {fpgrowth_time:.4f} s\n")


# In[85]:


print("\nTop 5 rules Apriori by lift:")
print(rules.sort_values(by='lift', ascending=False).head(5)[['antecedents','consequents','support','confidence','lift']])

print("\nTop 5 rules FP-Growth by lift:")
print(rules_fp.sort_values(by='lift', ascending=False).head(5)[['antecedents','consequents','support','confidence','lift']])


# In[ ]:



st.set_page_config(page_title="Market Basket", layout="wide")

st.markdown("<h1 style='font-size:36px; color:navy;'>Product Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:20px;'>Information panel.</p>", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)
col1.metric("Transactions", len(df2), delta=None)
col2.metric("Unique clients", df2['CustomerID'].nunique())
col3.metric("Unique products", df2['Products'].nunique())


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