
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # 项目根目录
sys.path.append(str(ROOT))

import numpy as np
from MKGLMFA.MKGLMFA import MKGLMFA
from MKGLMFA.graph import m_locaglob
from MKGLMFA.kernel import construct_kernel
from sklearn.neighbors import KNeighborsClassifier
from MKGLMFA.knn_utils import knnclassification

N = 4000

import numpy as np

def load_data(path, rows=90, cols=4000):
    data = np.loadtxt(path)

    # 如果文件是 (4000, 90) 或 (4000, >90)
    if data.shape[0] == cols:
        data = data[:, :rows].T
    else:
        data = data[:rows, :cols]

    return data


normaldata = load_data(r'C:\Users\Clavi\Desktop\coding\DANN-zcy\MKGLMFA\MFPTdata/base_1_overlap.txt')
outer100data = load_data(r'C:\Users\Clavi\Desktop\coding\DANN-zcy\MKGLMFA\MFPTdata/outer_100_overlap.txt')
outer200data = load_data(r'C:\Users\Clavi\Desktop\coding\DANN-zcy\MKGLMFA\MFPTdata/outer_200_overlap.txt')
inner0data = load_data(r'C:\Users\Clavi\Desktop\coding\DANN-zcy\MKGLMFA\MFPTdata/inner_0_overlap.txt')
inner300data = load_data(r'C:\Users\Clavi\Desktop\coding\DANN-zcy\MKGLMFA\MFPTdata/inner_300_overlap.txt')

# print(normaldata.shape)
# print(outer100data)

# training (前60)  共300个点
trfea = np.vstack([
    normaldata[:60],
    outer100data[:60],
    outer200data[:60],
    inner0data[:60],
    inner300data[:60]
])

# testing (后30)
tefea = np.vstack([
    normaldata[60:90],
    outer100data[60:90],
    outer200data[60:90],
    inner0data[60:90],
    inner300data[60:90]
])

gnd = np.concatenate([
    np.ones(60) * 1,
    np.ones(60) * 2,
    np.ones(60) * 3,
    np.ones(60) * 4,
    np.ones(60) * 5
]).astype(int)


options = {
    'intraK': 10,
    'interK': 20,
    'Regu': 1,
    'ReguAlpha': 0.5,
    'ReguBeta': 0.5,
    'ReducedDim': 10,
    'KernelType': 'Gaussian',
    't': 5,
    'Kernel': 1
}

L, Ln = m_locaglob(trfea, TYPE='nn', PARAM=5)

# -----------MKGLMFA------------------
eigvector, eigvalue, elapse = MKGLMFA(
    gnd=gnd,
    data=trfea,
    Ln=Ln,
    L=L,
    options=options
)

eigvector = np.nan_to_num(eigvector, nan=0.0)
print("Eigenvalues:", eigvalue)

# kernel for training # 训练集降维：
Ktrain, _ = construct_kernel(trfea, None, options)
Ktrain = np.nan_to_num(Ktrain, nan=0.0)
trY = Ktrain @ eigvector

# kernel for testing # 测试集降维：用同一投影矩阵把测试核矩阵也压到同一低维空间
Ktest, _ = construct_kernel(tefea, trfea, options)
Ktest = np.nan_to_num(Ktest, nan=0.0)
teY = Ktest @ eigvector


# --------------------1-NN分类-----------------------------
result = knnclassification(teY, trY, gnd, k=1, norm='2norm')

print(result)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(trY, gnd)

pred = knn.predict(teY)

true_labels = np.repeat([1,2,3,4,5], 30)

accuracy = np.mean(pred == true_labels)
print("Recognition rate:", accuracy)

from sklearn.metrics import classification_report
# 计算每一类的分类准确率
report = classification_report(true_labels, pred, digits=4)
print(report)

import matplotlib.pyplot as plt
import numpy as np

# 假设 trY, teY 已经是 numpy array
# gnd 是训练标签

# 只取前2维进行可视化
X_vis = trY[:, :2]

# 类别数
labels = np.unique(gnd)
colors = ['r', 'g', 'b', 'c', 'm']  # 可按需要增加颜色

plt.figure(figsize=(8,6))
for i, label in enumerate(labels):
    idx = np.where(gnd == label)
    plt.scatter(X_vis[idx, 0], X_vis[idx, 1], color=colors[i], label=f'Class {label}', alpha=0.7)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('MKGLMFA 2D Projection (Training Data)')
plt.legend()
plt.grid(True)
plt.show()


