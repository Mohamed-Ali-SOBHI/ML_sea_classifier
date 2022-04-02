import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale, resize
from sklearn.metrics import accuracy_score
from joblib import load
import os
from tqdm import tqdm
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

histogram_clf = load('histogram.joblib')
hog_clf = load('hog.joblib')
horizontality_clf = load('horizontality.joblib')

X_test = []
y_test = []
directory = "./Data/Evaluate"  # Changer dossier d'évaluation ici
for file in tqdm(os.listdir(directory)):
    X_test.append(directory + "/" + file)
    y_test.append(int(file[0]))  # Changer reconnaissance de la classe ici


histogram_representation = {}
for file in tqdm(X_test):
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((500,750))
    image = image.crop((0, 250, 500, 750))
    r, g, b = image.split()
    histogram_representation[file] = b.histogram()

hog_representation = {}
for file in tqdm(X_test):
    image = imread(file, as_gray=True)
    image = resize(image, (1200, 1200))
    image = rescale(image, 1/3, mode='reflect')
    img_hog = hog(image, orientations=16, pixels_per_cell=(16,16),
                    cells_per_block=(1,1), feature_vector=True, block_norm='L2')
    hog_representation[file] = img_hog


# bleues : 150 -> 240; verts : 75 -> 150; jaunes : 30 -> 65
def color_perc(imarray_rgb, lt, ht):
    # calcul du pourcentage de bleu à l'aide d'un masque
    imarray_rgb = imarray_rgb[:, :, :3]
    imarray_hue = rgb2hsv(imarray_rgb)[:, :, 0] * 360
    nb_px = imarray_hue.shape[0] * imarray_hue.shape[1]
    return np.sum(((imarray_hue > lt) & (imarray_hue < ht)).astype(int)) / nb_px


# rouges: 300 -> 360, 0 -> 30;
def red_perc(imarray_rgb):
    imarray_rgb = imarray_rgb[:, :, :3]
    imarray_hue = rgb2hsv(imarray_rgb)[:, :, 0] * 360
    nb_px = imarray_hue.shape[0] * imarray_hue.shape[1]
    return np.sum(((imarray_hue > 310) | (imarray_hue < 35)).astype(int)) / nb_px


def angle_to_hori(p1, p2):
    # source : https://stackoverflow.com/questions/7586063/how-to-calculate-the-angle-between-a-line-and-the-horizontal-axis
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    theta = np.arctan2(dy, dx)
    return np.degrees(theta)


def compute_hvt_props(imarray):
    """
    Nécessite une image en mode gris
    """
    # source : https://scikit-image.org/docs/stable/auto_examples/edges/plot_line_hough_transform.html
    canny_imarray = canny(imarray, 5, 1, 19)  # filtre pour retrouver les contours
    lines = probabilistic_hough_line(canny_imarray, threshold=10, line_length=25, line_gap=10)
    h, v, t = 0, 0, 0
    for p0, p1 in lines:
        angle = np.abs(angle_to_hori(p0, p1))
        if 60 <= angle <= 120 or 240 <= angle <= 300:
            v += 1
        elif 0 <= angle <= 20 or 340 <= angle <= 360 or 160 <= angle <= 200:
            h += 1
        else:
            t += 1
    tot = len(lines) + 1
    return h / tot, v / tot, t / tot, canny_imarray


horizontality_representation = {}
for file in tqdm(X_test):
    imarray = resize(imread(file), (800, 800), anti_aliasing=True)
    b_perc = color_perc(imarray[250:, :], 175, 280)
    g_perc = color_perc(imarray[:, :], 70, 170)
    r_per = red_perc(imarray)
    imarray_l = rgb2gray(imarray[:, :, :3]) * 255
    # im_blurred = nd.median_filter(imarray_l, size=13)
    h, v, t, _ = compute_hvt_props(imarray_l[150:,:])
    horizontality_representation[file] = [g_perc, b_perc, r_per, h, v, t]

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

clf_hist = RandomForestClassifier(max_depth=5, min_samples_split=5)
clf_hog = SVC(gamma='auto')
clf_horizontality = RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_split=5)

classifieurs = {histogram_clf: histogram_representation,
                hog_clf : hog_representation,
                horizontality_clf : horizontality_representation}

def majority_voting(predicted_ys):
    y_majority = []
    for i in range(len(y_test)):
        nb_0 = 0
        nb_1 = 0
        for pred in predicted_ys:
            if pred[i] == 0:
                nb_0 += 1
            else:
                nb_1 += 1
        if nb_0 > nb_1:
            y_majority.append(0)
        else:
            y_majority.append(1)
    return y_majority

def plot_results(clf_scores):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
    fig.suptitle('Résultats')
    ax1.hist(clf_scores)
    ax2.boxplot(clf_scores)

    print("Moyenne : " + str(np.mean(clf_scores))+  ", Variance : " + str(np.var(clf_scores)) + ", Écart-type : " + str(np.std(clf_scores)))

predicted_ys = []

for clf in classifieurs.keys():

    test_representation = []
    for file in X_test:
        test_representation.append(classifieurs[clf][file])

    predicted_ys.append(clf.predict(test_representation))

y_majority = majority_voting(predicted_ys)
print("Score : " + str(accuracy_score(y_test, y_majority)))