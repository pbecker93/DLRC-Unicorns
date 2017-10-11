import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from color_cluster.siamese_model import Model
import color_cluster.load_data
import numpy as np

img_size = (64, 64)
DATA_PATH = 'data'
data = load_data.SiameseData(img_size=img_size, path=DATA_PATH)

m = Model(img_size, data=data)
#m.train(batch_size=100, num_batches=1001, verbose_interval=50)

#m.save('model/')
m.load_model('model/')
fig = plt.figure()
ax = Axes3D(fig)
means = np.ones((5, 3))
for i, c in enumerate(data.colors):
    col = c if c is not data.COL_WHITE else "black"
    current = m.get_latent(data.train_imgs[c])
    means[i] = np.mean(current, 0)
    ax.scatter(current[:, 0], current[:, 1], current[:, 2], c=col)

print(means)
for i in range(5):
    for j in range(5):
        print(np.linalg.norm(means[i] - means[j]))
