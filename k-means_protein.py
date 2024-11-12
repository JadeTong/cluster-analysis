'''
聚类分析————蛋白质消费特征分析
1.数据及分析对象
'protein.csv'，主要记录了25个国家的9个属性，主要属性如下：
（1）ID：国家的ID；
（2）Country（国家类别）：该数据集涉及25个欧洲国家肉类和其他食品之间的关系；
（3）关于肉类和其他食品的9个数据包括RedMeat（红肉），WhiteMeat（白肉），Eggs（蛋类），Milk（牛奶），Fish（鱼类），Cereals（谷类），Starch（淀粉类），Nuts（坚果类），Fr&Veg（水果和蔬菜）。

2.目的及分析任务
理解机器学习方法在数据分析中的应用——采用k-means方法进行聚类分析：
（1）将数据集导入后，在初始化阶段随机选择k个类簇进行聚类，确定初始聚类中心；
（2）以初始化后的分类模型为基础，通过计算每一簇的中心点 重新确定聚类中心；
（3）迭代重复“计算距离—确定聚类中心—聚类”的过程；
（4）通过检验特定的指标来验证k-means模型聚类的正确性和合理性。

'''
#%%                       1.业务理解
'''
为不同国家的蛋白质消费结构分析；从数据集中选取不同国家蛋白质饰品的消费数据，通过k-means算法模型对其进行迭代求解的聚类分析，最后评价聚类效果的优度。
'''
#%%                       2.数据读取
import pandas as pd
import numpy as np
data = pd.read_table('D:/desktop/ML/聚类分析/protein.txt', sep='\t') #'sep='\t'表示分隔符为tab制表符
data.shape #(25, 10)

#%%                       3.数据准备
#%%% 提取数据集中与蛋白质相关的列，drop掉‘Country’
protein = data.drop('Country', axis = 1)
protein

#%%% 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
protein_scaled = scaler.fit_transform(protein)

#%%                       4.模型训练
# 在使用k-means之前，需要在初始阶段产生一个随机的K值作为类簌的个数。sklearn用决定系数作为性能评估的分数，判断模型对数据的拟合能力。sklearn.cluster模块的k-means.fit().score()
# 选择k值，使用肘部法则确定最优簇数 K
from sklearn.cluster import KMeans

sse = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=30)
    kmeans.fit(protein_scaled)
    sse.append(kmeans.inertia_)  # SSE (Sum of Squared Errors)

# 绘制肘部图
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(K_range, sse, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()
# 肘部图看得出来，第一次变缓的是k=3，进一步话轮廓图

#%%% 轮廓系数
from sklearn.metrics import silhouette_score

silhouette_scores = []
K_range = range(2, 10)  # 轮廓系数要求 K >= 2
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=30)
    kmeans.fit(protein_scaled)
    score = silhouette_score(protein_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# 绘制轮廓系数图形
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.xticks(K_range)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method for Optimal K')
plt.grid(True)
plt.show()

# 轮廓图中k=3时，轮廓系数最大，选3

#%%%  根据k值，随机选择K个数据点作为初始簇中心（Centroids）
kmeans = KMeans(n_clusters=3, init='k-means++', n_init='auto', random_state=30)
data['Cluster'] = kmeans.fit_predict(protein_scaled)
data['Cluster']

#%%                       5.模型评价
# Step 8: 聚类结果评估 (轮廓系数)
silhouette_avg = silhouette_score(protein_scaled, data['Cluster'])
print(f"轮廓系数 (Silhouette Score): {silhouette_avg:.2f}")   
# 轮廓系数 (Silhouette Score): 0.35

#%%           可视化聚类结果
# 为了可视化，我们使用 PCA 将数据降维至 2 维
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_pca = pca.fit_transform(protein_scaled)
data['PCA1'] = data_pca[:, 0]
data['PCA2'] = data_pca[:, 1]

# 将簇中心点也降维
centers=pca.transform(kmeans.cluster_centers_)

#%%% 绘制聚类散点图
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster',data=data, palette='Set2', s=100)
plt.scatter(centers[:,0],centers[:,1],marker='x',s=100,c='red')  # 标注簇中心点
for i, country in enumerate(data['Country']):
    plt.text(data['PCA1'][i], data['PCA2'][i], country)   #标注国家名称
plt.title('K-means Clustering of Countries Based on Protein Consumption')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)

#%%    查看每个簇的特征均值
# 分析不同簇的食品消费特征
numeric_columns = data.select_dtypes(include=[np.number]).columns

# 对数值列按 Cluster 进行分组求平均
features = protein.columns
cluster_analysis = data.groupby('Cluster', observed=False)[features].mean()
print("\n各簇的平均特征值：")
print(cluster_analysis)

#%%%   分析与解读
'''
Cluster 0
鱼类（Fish）和水果蔬菜（Fr&Veg） 的消费量显著高于其他簇。
谷类（Cereals）和淀粉（Starch） 的消费量居中。
可能代表了一组鱼类和蔬菜消费较高的国家，且红肉和牛奶的消费偏低。

Cluster 1
红肉（RedMeat）和白肉（WhiteMeat） 的平均消费量最高。
蛋类（Eggs）、牛奶（Milk）和鱼类（Fish） 的消费量也较高。
这类国家可能偏向高蛋白饮食，尤其是肉类和奶制品的消费量较高。

Cluster 2
谷类（Cereals） 的消费量显著高于其他簇，达到了 46.16。
淀粉（Starch） 的消费量最低，同时 鱼类（Fish） 也是最低的。
这种饮食模式可能偏向高碳水化合物（主要来自谷类）而低蛋白的结构，代表传统的谷类为主食的国家。
'''

# Step 11: 按簇分组查看各国家
grouped = data[['Country', 'Cluster']].groupby('Cluster')
for name, group in grouped:
    print(f"\nCluster {name} 包含的国家：\n", group['Country'].values)
 









