#!/usr/bin/env python
# coding: utf-8

# Integrated CA2 DVT and MLB Daniela Leon

# In[2]:


import streamlit as st

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")


# In[3]:


df1 = pd.read_csv("Online-eCommerce.csv")
df1.head()


# In[4]:


df1.shape 


# In[5]:


duplicate_rows_df = df1[df1.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[6]:


df1.describe()


# In[7]:


df1.info()


# In[8]:


df1.dtypes


# In[9]:


df1.dtypes.value_counts()


# In[10]:


df1.isnull().sum()


# In[11]:


Missing_df=df1.isnull().sum()
Missing_df =Missing_df[Missing_df > 0]
Missing_df


# In[12]:


df1['Product'] = df1['Product'].astype(str).str.strip().str.lower()
df1['Category'] = df1['Category'].astype(str).str.strip().str.lower()
df1['Brand'] = df1['Brand'].astype(str).str.strip().str.lower()
df1['Customer_Name'] = df1['Customer_Name'].astype(str).str.strip()


# In[13]:


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
                        aggfunc='sum',
                        fill_value=0)

Check


# In[16]:


from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

user= cosine_similarity(Check)
similarity_with_user= pd.DataFrame(user,
                                  index=Check.index,
                                  columns=Check.index)

similarity_with_user.head()


# In[17]:


def find_n_neighbours_user(similarity_with_user, n):

    top_users = similarity_with_user.apply(
        lambda row: pd.Series(
            row.sort_values(ascending=False).iloc[1:n+1].index,
            index=[f'top_user_{i}' for i in range(1, n+1)]),
        axis=1)

    return top_users


# In[18]:


top_n = 30
sim_user_30_u = find_n_neighbours_user(similarity_with_user, top_n)
sim_user_30_u.head()


# In[19]:


def get_common_products(user1, user2):
    common_products_df = Check.loc[[user1, user2]].T
    
    common_products_df = common_products_df[(common_products_df[user1] > 0) & (common_products_df[user2] > 0)]
    
    return common_products_df.index.tolist()


# In[20]:


user1 = 'Ajay Sharma'

top_neighbor = sim_user_30_u.loc[user1, 'top_user_1']

common_products = get_common_products(user1, top_neighbor)

print(f"User: {user1}")
print(f"Nearest neighbor: {top_neighbor}")
print(f"Common products bought by both: {common_products}")


# Item-Item

# In[22]:


item= cosine_similarity(Check.T)
item_similarity= pd.DataFrame(item,
                                  index=Check.columns,
                                  columns=Check.columns)

item_similarity.head()


# In[23]:


def find_n_neighbours_item(item_similarity, n):

    top_items = item_similarity.apply(
        lambda row: pd.Series(
            row.sort_values(ascending=False).iloc[1:n+1].index,
            index=[f'top_item_{i}' for i in range(1, n+1)]
        ),
        axis=1
    )
    
    return top_items

top_n_items = 10
sim_item_10 = find_n_neighbours_item(item_similarity, top_n_items)

sim_item_10.head()


# In[24]:


def get_users_who_bought(product):
    return Check[Check[product] > 0].index.tolist()


# In[25]:


product_example = "cpu"   

similar_items = sim_item_10.loc[product_example]

print(f"Producto base: {product_example}")
print("Most similar products (item-item):")
print(similar_items.tolist())


# User–User Collaborative Filtering Recommendation

# In[27]:


def predict_user_product_score(user, product, Check, similarity_with_user, top_n=30):
    similar_users = similarity_with_user.loc[user].sort_values(ascending=False)[1:top_n+1]
    product_vector = Check.loc[similar_users.index, product]
    product_vector = product_vector[product_vector > 0]
    if product_vector.empty:
        return 0
    correlation = similar_users[product_vector.index]
    score = (product_vector * correlation).sum() / correlation.sum()
    return score


# In[28]:


def recommend_products_user_user(user, Check, similarity_with_user, top_n_neighbors=30, n_recommendations=5):
    bought_products = Check.loc[user][Check.loc[user] > 0].index.tolist()
    candidate_products = [p for p in Check.columns if p not in bought_products]
    scores = {p: predict_user_product_score(user, p, Check, similarity_with_user, top_n_neighbors) 
              for p in candidate_products}
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [p for p, s in recommended[:n_recommendations]]


# In[29]:


recommendations_user = recommend_products_user_user("Ajay Sharma", Check, similarity_with_user)
print("User-User Recomendaciones:", recommendations_user)


# Item–Item Collaborative Filtering Recommendation

# In[31]:


def recommend_products_item_item(user, item_similarity, Check, top_n_items=10, n_recommendations=5):
    bought_products = Check.loc[user][Check.loc[user] > 0].index.tolist()
    candidate_products = [p for p in Check.columns if p not in bought_products]
    scores = {}
    for product in candidate_products:
        similar_products = item_similarity.loc[product].sort_values(ascending=False)[1:top_n_items+1]
        bought_scores = Check.loc[user, similar_products.index]
        scores[product] = (bought_scores * similar_products).sum() / similar_products.sum()
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [p for p, s in recommended[:n_recommendations]]


# In[32]:


recommendations_item = recommend_products_item_item("Ajay Sharma", item_similarity, Check)
print("Item-Item Recomendaciones:", recommendations_item)


# In[33]:


plt.figure(figsize=(12,8))
sns.heatmap(similarity_with_user, cmap='viridis')
plt.title("User-User Similarity Heatmap")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(item_similarity, cmap='magma')
plt.title("Item-Item Similarity Heatmap")
plt.show()


# In[34]:


#all_products = [p for sublist in df1['Category'] for p in sublist]
#product_counts = Counter(all_products)

#product_counts_series = pd.Series(product_counts)

#print("Number of products:", product_counts_series.shape[0])
#print("Median counts per product:", product_counts_series.median())

#plt.figure(figsize=(6,4))
#plt.hist(product_counts_series, bins=50, color='skyblue', edgecolor='black')
#plt.xlabel("Number of occurrences per product")
#plt.ylabel("Number of products")
#plt.title("Distribution of Product Occurrences (log scale)")
#plt.yscale('log')
#plt.show()


# In[35]:


import numpy as np

user_vals = similarity_with_user.values[np.triu_indices_from(similarity_with_user.values, k=1)]
item_vals = item_similarity.values[np.triu_indices_from(item_similarity.values, k=1)]

plt.hist(user_vals, bins=25, edgecolor='black')
plt.title("Distribución User-User Cosine Similarity")
plt.show()

plt.hist(item_vals, bins=25, edgecolor='black', color='orange')
plt.title("Distribución Item-Item Cosine Similarity")
plt.show()


# In[36]:


def precision_at_k(recommended, relevant):
    recommended_set = set(recommended)
    relevant_set = set(relevant)
    return len(recommended_set & relevant_set) / len(recommended_set) if recommended_set else 0


# In[37]:


def recall_at_k(recommended, relevant):
    recommended_set = set(recommended)
    relevant_set = set(relevant)
    return len(recommended_set & relevant_set) / len(relevant_set) if relevant_set else 0


# In[38]:


user_test = "Ajay Sharma"
actual_products = Check.loc[user_test][Check.loc[user_test] > 0].index.tolist()

recommended_user = recommend_products_user_user(user_test, Check, similarity_with_user)
recommended_item = recommend_products_item_item(user_test, item_similarity, Check)

metrics = {
    "User-User Precision@K": precision_at_k(recommended_user, actual_products),
    "User-User Recall@K": recall_at_k(recommended_user, actual_products),
    "Item-Item Precision@K": precision_at_k(recommended_item, actual_products),
    "Item-Item Recall@K": recall_at_k(recommended_item, actual_products)
}

print(metrics)


# In[ ]:





# Part 2

# In[40]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


import time

import warnings
warnings.filterwarnings('ignore')


# In[41]:


#pip install mlxtend


# In[42]:


df2 = pd.read_csv("products.csv")
df2.head()


# In[43]:


df2.shape 


# In[44]:


duplicate_rows_df = df2[df2.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[45]:


df2.describe()


# In[46]:


df2.info()


# In[47]:


df2.dtypes


# Data clean

# In[49]:


df2.dtypes.value_counts()


# In[50]:


df2.isnull().sum()


# In[51]:


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

# In[54]:


df_ana=df2.drop(columns=["Timestamp"])
df_ana.head()


# In[55]:


df_ana.dtypes


# In[56]:


df_ana['TransactionID'] = df_ana['TransactionID'].astype(str)
df_ana['CustomerID'] = df_ana['CustomerID'].astype(str)
df_ana['Products'] = df_ana['Products'].astype(str)

#df_ana['Products_List'] = df_ana['Products'].apply(lambda x: [p.strip() for p in x.split(',')])


# In[57]:


df_ana.dtypes


# In[58]:


df_ana.head()


# In[59]:


df_sample = df_ana.sample(n=3000, random_state=42)


# In[60]:


df_sample


# In[61]:


df_sample['Products'] = df_sample['Products'].apply(
    lambda lst: [p.strip() for p in lst if isinstance(p, str) and p.strip() not in ['', ',', ';']])


# In[62]:


all_products = [p for sublist in df_ana['Products'] for p in sublist]
top_products = [p for p, _ in Counter(all_products).most_common(10)]

df_sample['Products'] = df_sample['Products'].apply(
    lambda x: [p for p in x if p in top_products])


# Apriori Algorithm

# In[64]:


basket = df_sample['Products'].tolist()

te = TransactionEncoder()
te_array = te.fit(basket).transform(basket)
df_both = pd.DataFrame(te_array, columns = te.columns_)


# In[65]:


#basket = df_sample['Products'].tolist()

#te = TransactionEncoder()
#te_array = te.fit(basket).transform(basket, sparse=True)
#df_both = pd.DataFrame.sparse.from_spmatrix(te_array, columns=te.columns_)


# In[66]:


#df_apriori = df_both.astype(bool)


# In[67]:


frequent_itemsets_ap = apriori(df_both, min_support=0.01, use_colnames=True)


# In[68]:


print(frequent_itemsets_ap)


# In[69]:


frequent_itemsets_ap.shape


# In[70]:


rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.1)
rules_ap


# FP Growth Algorithm

# In[72]:


basket = df_sample['Products'].tolist()

te = TransactionEncoder()
te_array = te.fit(basket).transform(basket)
df_both = pd.DataFrame(te_array, columns = te.columns_)


# In[73]:


#basket = df_sample['Products'].tolist()

#te = TransactionEncoder()
#te_array = te.fit(basket).transform(basket, sparse=True)
#df_both = pd.DataFrame.sparse.from_spmatrix(te_array, columns=te.columns_)


# In[74]:


frequent_itemsets_fp = fpgrowth(df_both, min_support = 0.01, use_colnames = True)
print(frequent_itemsets_fp)


# In[75]:


rules_fp = association_rules(frequent_itemsets_fp, metric = "confidence", min_threshold = 0.8)

print(rules_fp)


# In[76]:


start = time.time()
frequent_itemsets_ap = apriori(df_both, min_support=0.01, use_colnames=True)
print("Apriori:", time.time() - start)

start = time.time()
frequent_itemsets_fp = fpgrowth(df_both, min_support=0.01, use_colnames=True)
print("FP-Growth:", time.time() - start)


# In[77]:


print("=== Apriori ===")
print(f"Número de itemsets frecuentes: {len(frequent_itemsets_ap)}")
print(f"Número de reglas generadas: {len(rules_ap)}")
print(f"Tiempo de ejecución: {start:.4f} s\n")

print("=== FP-Growth ===")
print(f"Número de itemsets frecuentes: {len(frequent_itemsets_fp)}")
print(f"Número de reglas generadas: {len(rules_fp)}")
print(f"Tiempo de ejecución: {start:.4f} s\n")


# In[78]:


print("\nTop 5 reglas Apriori por lift:")
print(rules_ap.sort_values(by='lift', ascending=False).head(5)[['antecedents','consequents','support','confidence','lift']])

print("\nTop 5 reglas FP-Growth por lift:")
print(rules_fp.sort_values(by='lift', ascending=False).head(5)[['antecedents','consequents','support','confidence','lift']])


# In[79]:


#pip install voila


# In[80]:


#pip install streamlit pandas plotly mlxtend

import streamlit as st
import pandas as pd

st.title("Product Analysis")

# Cargar datos directamente desde GitHub
url = "https://github.com/CCT-Dublin/integrated-ca2-dvt-and-mlb-Daniela2729/blob/main/products.csv"
df2 = pd.read_excel(url)

df_sample = df2.sample(n=3000, random_state=42)

st.subheader("Raw Data")
st.write(df_sample)
