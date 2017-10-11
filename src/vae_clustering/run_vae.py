from vae_clustering.vae_data import VAEData
from vae_clustering.vae import VAE
import tensorflow as tf
import numpy as np
import copy
import cv2
import matplotlib.colors as mplCol
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as img_filter
import sklearn.decomposition as d


DATA_PATH = '/home/dlrc/crop_64x64'
IMG_SIZE = (64, 64)

def clahe(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20,20))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def preprocess(img):
    return img_filter.gaussian_filter(img, [0, 0, 5])


data = VAEData(DATA_PATH, img_size=IMG_SIZE)

#tf_session = tf.Session()

train_data_dict = data.train_imgs
train_data = np.concatenate([train_data_dict[k] for k in train_data_dict.keys()], 0)
train_smooth = np.stack([preprocess(train_data[i]) for i in range(len(train_data))])
train_data_dict_smooth = dict()
k = np.ones((3, 3))
for c in data.colors:
    cur_list = train_data_dict[c]
    cur_smooth = np.zeros((len(cur_list), 64, 64, 3))
    for i in range(len(cur_list)):
        cur_smooth[i] = preprocess(cur_list[i]) #cv2.dilate(cur_list[i], k, iterations=10)
    train_data_dict_smooth[c] = cur_smooth
        #train_data_dict[c] #img_filter.gaussian_filter(train_data_dict[c], [0, 20, 20, 0])
    print(c)
#    cv2.imshow(c, cv2.cvtColor(test_data_dict[c][0], cv2

print(train_data.shape)

#for c in data.colors:
#    plt.imshow(train_data_dict_smooth[c][0])
#    plt.show()

#train_data =  #np.array([equilize_light(imgs[i]) for i in range(len(imgs))])[:, 12:36, 12:36, :]

#train_smooth = img_filter.gaussian_filter(train_data, [0, 20, 20, 0])

#test_data_dict = data.test_imgs

#test_data = np.concatenate([test_data_dict[k] for k in test_data_dict.keys()], 0)
#test_smooth = test_data #img_filter.gaussian_filter(test_data, [0, 20, 20, 0])
#test_data_dict_smooth = dict()
#for c in data.colors:
#    test_data_dict_smooth[c] = test_data_dict[c] #img_filter.gaussian_filter(test_data_dict[c], [0, 20, 20, 0])
#    print(c)
#    cv2.imshow(c, cv2.cvtColor(test_data_dict[c][0], cv2.COLOR_HSV2RGB))
#    cv2.waitKey(0)

#print(test_data.shape)
#real_data_smooth_dict = dict()
#for k in real_data_dict.keys():
#    real_data_smooth_dict[k] = img_filter.gaussian_filter(real_data_dict[k], [0, 13, 13, 0])
#real_data_smooth = np.concatenate([real_data_smooth_dict[k] for k in real_data_smooth_dict.keys()], 0)

vae = VAE((64, 64), 4, 'nll')
vae.train(train_smooth, 1000, 100)
#fig =plt.figure()
#ax = Axes3D(fig)
mean_dict = dict()
for i, c in enumerate(data.colors):

    mean_dict[c] = vae.get_latent_distribution(train_data_dict_smooth[c])

all_latent = np.concatenate([mean_dict[c] for c in data.colors], 0)

total_std = np.std(all_latent, 0)
print(total_std.shape)

pca = d.PCA()
pca.fit(all_latent)


pca_dict = dict()
for c in data.colors:
    pca_dict[c] = pca.transform(mean_dict[c])


for i in range(4):
    fig = plt.figure()
    for c in data.colors:
        col = "black" if c == data.COL_WHITE else c
        plt.plot(pca_dict[c][:, i])#, c=col)


#for cur_key in data.colors:
#    o_col = copy.deepcopy(data.colors)
#    o_col.remove(cur_key)
#    current = mean_dict[cur_key]
#    others = np.concatenate([mean_dict[c] for c in o_col], 0)
#    print(current.shape)
#    print(others.shape)
#    dist = 0
#    for i in range(len(current)):
#        for j in range(len(others)):
#            dist += (current[i] - others[j])**2
#    avg_dist = dist / (len(current) * len(others))
#    norm_avg_dist = avg_dist / total_std
#    max_dist = np.max(norm_avg_dist)
#    max_dist_dim = np.argmax(norm_avg_dist)
#    print(max_dist, max_dist_dim, norm_avg_dist)
#    plt.figure()
#    for c in data.colors:
#        col = 'black' if c == VAEData.COL_WHITE else c
#        plt.plot(mean_dict[c][:, max_dist_dim], c=col)




            #ax.scatter(latent_train_means[:, 0], latent_train_means[:, 1], latent_train_means[:, 2], c=col)
#vae.save('')

#real_data_dict = data.real_imgs




recon_train = vae.predict(train_smooth[:10])

plt.figure()
plt.imshow(train_smooth[1])
plt.figure()
plt.imshow(recon_train[1])
#fig = plt.figure()
plt.show()


#recon_test = vae.predict(test_smooth)
#plt.figure()
#plt.imshow(test_smooth[1])
#plt.figure()
#plt.imshow(recon_test[1])

#means = np.zeros((7,7))

#plt.figure()
#ax = Axes3D(fig)

#for i, c in enumerate(data.colors):
#    col = "black" if c == VAEData.COL_WHITE else c
#    latent_test_means = vae.get_latent_distribution(test_data_dict_smooth[c])
#    ax.scatter(latent_test_means[:, 0], latent_test_means[:, 1], latent_test_means[:, 2], c=col)

#plt.figure()        # ax = Axes3D(fig)
#for i, c in enumerate(data.colors):
#    col = "black" if c == VAEData.COL_WHITE else c
#    latent_real_mean, _ = vae.get_latent_distribution(real_data_smooth_dict[c])
#    plt.scatter(latent_real_mean[:, 0], latent_real_mean[:, 1], c=col, marker='+')

    #plt.figure()
    #for j in range(20):

     #   plt.subplot(20,1,j+1)
      #  if j == 0:
    #        plt.title('color: ' + c)
    #    plt.plot(latent_test_means[j, :])
    #    means[i, :] = np.mean(latent_test_means, 0)

#plt.figure()
#plt.title("means")
#for i, c in enumerate(data.colors):
#    col = c if c != VAEData.COL_WHITE else 'black'
#    plt.plot(means[i, :], c=col)


#ax = Axes3D(fig)

#    latent_test_means, _ = vae.get_latent_distribution(test_data_dict[c])
#    ax.scatter(latent_test_means[:, 0], latent_test_means[:, 1], latent_test_means[:, 2], c=c)


