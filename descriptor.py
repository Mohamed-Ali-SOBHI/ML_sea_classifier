import math
import os
from abc import ABCMeta, abstractmethod

import numpy as np
from PIL import Image
from skimage.feature import greycomatrix, greycoprops, hog
from skimage.io import imread
from skimage.measure import shannon_entropy
from skimage.transform import resize, rescale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tqdm import tqdm

_path_mer = "./Data/Mer/"
_path_ailleurs = "./Data/Ailleurs/"


class BaseDescriptor(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        """
        """
        self.X = []
        self.y = []
        pass

    @abstractmethod
    def data_collect(self, image_path):
        """
        Méthode à définir par les sub-objets pour récupérer les données du modèle pour une image.
        """
        raise NotImplementedError("Cette fonction doit être implémentée")

    @abstractmethod
    def to_fit(self):
        """
        Méthode à définir pour transformer la séquence créée à l'aide de la méthode data_collect de manière à ce qu'elle
        soit acceptée par les méthodes comme fit(X, y) des classifieurs.
        Retourne par défaut X et y
        """
        return self.X, self.y

    @abstractmethod
    def my_train_test_split(self, X, y, test_size):
        """
        Méthode à définir pour effectuer (si nécessaire) soit même le shuffle de l'opération train_test_split de
        sklearn.
        Renvoie par défaut le train_test_split de sklearn
        """
        return train_test_split(X, y, test_size=test_size)

    @abstractmethod
    def real_predict(self, y_predicts, predictor=None):
        """
        Méthode pour transformer la séquence de predicts créée par classifieur.predict en une séquence qui
        correspond bien aux images_denoise.
        Renvoie par défaut le y déjà calculé.

        predictor : fonction qui prend en entrée les predictions pour un ensemble de fenêtre correspondant à une image
                    et qui renvoie la classe prédite pour l'image (par exemple on pourra faire un vote majoritaire)
        """
        return y_predicts

    def reset(self):
        self.X = []
        self.y = []

    def construct(self):
        """
        Méthode pour charger les données depuis les échantillons fournis en faisant appel à la fonction data_collect
        """
        print("constructing descriptor..")
        self.X += [self.data_collect(_path_mer + file) for file in tqdm(os.listdir(_path_mer))] + \
                  [self.data_collect(_path_ailleurs + file) for file in tqdm(os.listdir(_path_ailleurs))]
        self.y += [1] * len(os.listdir(_path_mer)) + [0] * len(os.listdir(_path_ailleurs))
        print("done")

    def construct_from_images(self, images_paths, class_extractor=None):
        """
        Méthode pour charger les données à partir d'un chemin donné et d'une méthode d'extraction de classe (qui prend
        en entrée le chemin du fichier). Si la méthode d'extraction n'est pas définie, il faudra la définir
        manuellement.
        """
        for img_path in tqdm(images_paths):
            self.X.append(self.data_collect(images_paths + img_path))
            if class_extractor:
                self.y.append(class_extractor(img_path))

    def train_test_evaluate(self, classifier, predictor=None, n_iter=10, test_size=0.20):
        print("evaluating..")
        average_accuracy = 0

        for _ in tqdm(range(n_iter)):
            X_train, X_test, y_train, y_test = self.my_train_test_split(self.X, self.y, test_size)
            classifier.fit(X_train, y_train)
            y_predicts = classifier.predict(X_test)
            y_predicts = self.real_predict(y_predicts, predictor)
            average_accuracy += accuracy_score(y_test, y_predicts)

        print("done")

        return average_accuracy / n_iter

    def to_fitted_classifier(self, classifier):
        X, y = self.to_fit()
        print("fitting..")
        classifier.fit(X, y)
        return classifier


class OneInstanceBaseDescriptor(BaseDescriptor, metaclass=ABCMeta):

    def __init__(self):
        # rajouter attributs si necessaire
        # self.att = att
        super().__init__()

    def data_collect(self, image_path):
        # à implémenter, c'est tout
        pass

    def to_fit(self):
        return super().to_fit()

    def my_train_test_split(self, X, y, test_size):
        return super().my_train_test_split(X, y, test_size)

    def real_predict(self, y_predicts, predictor=None):
        return super().real_predict(y_predicts, predictor)


class MultiInstanceBaseDescriptor(BaseDescriptor, metaclass=ABCMeta):

    def __init__(self, n_target, n_measures):
        """
        n_target : le nombre attendu de fenêtres de mesures pour chaque image
        n_measures : le nombre attendu de mesures pour chaque une des fenêtres d'une image
        """
        self.n_target = n_target
        self.n_measures = n_measures
        super().__init__()

    def to_fit(self):
        """
        unpacking des valeurs
        """
        return zip(*[(self.X[index_img][index_patch], self.y[index_img][index_patch])
                     for index_patch in range(self.n_target)
                     for index_img in range(len(self.X))])

    def my_train_test_split(self, X, y, test_size):
        """"
        Méthode personnalisée fonctionnant exactement comme l'équivalant de sklearn
        """
        nb_tot = len(X)
        nb_test = int(nb_tot * test_size)
        while nb_test % self.n_target != 0:
            nb_test += 1

        X, y = shuffle(np.array(X), np.array(y))
        X_train, y_train = zip(*[(X[i_img][i_patch], y[i_img][i_patch])
                                 for i_patch in range(self.n_target)
                                 for i_img in range(nb_tot - nb_test)])
        X_test = [X[i_img][i_patch]
                  for i_patch in range(self.n_target)
                  for i_img in range(nb_test, nb_tot)]
        y_test = [y[i_img][0] for i_img in range(nb_test, nb_tot)]
        return X_train, X_test, y_train, y_test

    def real_predict(self, y_predicts, predictor=None):
        """
        Méthode pour transformer la séquence de predicts créée par classifieur.predict en une séquence qui
        correspond bien aux images_denoise.

        predicter : fonction qui prend en entrée les predictions pour un ensemble de fenêtre correspondant à une image
                    et qui renvoie la classe prédite pour l'image (par exemple on pourra faire un vote majoritaire)
        """
        if not predictor:
            return super().real_predict(y_predicts)
        return [predictor(y_predicts[i:i + self.n_target]) for i in range(0, len(y_predicts), self.n_target)]


def rotationally_invariant_measure(image, distances, levels, measure):
    DIRECTIONS = [0, np.pi / 4, np.pi / 2, (3 * np.pi) / 4]
    GLCMs = greycomatrix(image, distances=distances, angles=DIRECTIONS, levels=levels, symmetric=True, normed=True)
    return np.sum(greycoprops(GLCMs, measure))


########################################################################################################################
####################################################  CLASSES REELS ####################################################
########################################################################################################################


class TextureDescriptor(MultiInstanceBaseDescriptor):

    def __init__(self, measures, p_size, formatter, patcher, n_target=9, q_factor=4):
        """
        measures : les mesures à calculer dans les GLCMs
        p_size : la taille des patchs à appliquer sur chaque image
        patcher : callable chargé de calculer les coordonnées d'application des patchs. Il prend en entrée la largeur,
                  la hauteur de l'image, le nombre cible de patchs à appliquer et la taille que doit avoir chaque patch.
                  Il retourne un tuple de deux séquences, une pour les x, une pour les y.
        n_target : le nombre de patchs cible
        q_factor : le facteur de quantification à appliquer comme dernière transformation à l'image
        formatter : un callable qui transforme l'image avant d'appliquer les patchs (par exemple si on souhaite faire
                    un zoom. Prend et renvoie une image.
        """
        self.measures = measures
        self.p_size = p_size
        self.q_factor = q_factor
        self.formatter = formatter
        self.patcher = patcher
        super().__init__(n_target, len(measures) + 1)

    def data_collect(self, image_path):
        im = Image.open(image_path)
        if self.formatter is not None:
            image = self.formatter(im).quantize(self.q_factor)
        else:
            image = im.quantize(self.q_factor)
        xs, ys = self.patcher(image.width, image.height, self.n_target, self.p_size)
        n_patch = len(xs)
        patchs_data = []
        for k in range(n_patch):
            data = []
            cx = xs[k]
            cy = ys[k]
            patch = image.crop((cx, cy, cx + self.p_size, cy + self.p_size))
            for measure in self.measures:
                data.append(rotationally_invariant_measure(patch, [1], self.q_factor, measure))
            data.append(shannon_entropy(patch))
            patchs_data.append(data)
        return patchs_data

    def construct_y(self):
        self.y = [[y] * self.n_target for y in self.y]


class HistogramDescriptor(OneInstanceBaseDescriptor):

    def __init__(self, q_factor):
        self.q_factor = q_factor
        super().__init__()

    def data_collect(self, image_path):
        img = Image.open(image_path)
        img = img.convert('RGB')
        proportions = [0] * self.q_factor
        r_, g_, b_ = img.split()
        img_data = r_.histogram(), g_.histogram(), b_.histogram()
        pixel_values = img_data[2]
        step = 255 / self.q_factor
        index = 0
        while index < 255:
            proportions[math.floor(index / step)] += pixel_values[index]
            index += 1
        p_number_tot = img.width * img.height
        return [p / p_number_tot for p in proportions]

    def to_fit(self):
        return super().to_fit()

    def my_train_test_split(self, X, y, test_size):
        return super().my_train_test_split(X, y, test_size)

    def real_predict(self, y_predicts, predictor=None):
        return super().real_predict(y_predicts, predictor)


class HogDescriptor(OneInstanceBaseDescriptor):

    def __init__(self):
        self.w = 1200
        self.scale = 1 / 3
        self.ppc = (16, 16)  # pixels_per_cell
        self.cpb = (1, 1)  # cells_per_block
        self.ori = 16  # orientations
        self.bln = 'L2'  # {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
        super().__init__()
        pass

    def data_collect(self, image_path):
        image = imread(image_path, as_gray=True)
        image = resize(image, (self.w, self.w))
        image = rescale(image, self.scale, mode='reflect')
        img_hog = hog(image, orientations=self.ori, pixels_per_cell=self.ppc,
                      cells_per_block=self.cpb, feature_vector=True, block_norm=self.bln)
        return img_hog

    def to_fit(self):
        return super().to_fit()

    def my_train_test_split(self, X, y, test_size):
        return super().my_train_test_split(X, y, test_size)

    def real_predict(self, y_predicts, predictor=None):
        return super().real_predict(y_predicts, predictor)


def extract_class(path):
    return int(path[len("Data/Evaluation/"):][0])


def get_patch_coordinates(w, h, n_target, p_size):
    '''
    Calcul les coordonnées d'application de chaque patch pour obtenir une répartition la plus équitable possible
    '''
    xs = []
    ys = []
    max_patch = (w / p_size) * (h / p_size)
    step = int(max_patch / n_target)
    current_step = 0
    for i in range(int(w / p_size)):
        for j in range(int(h / p_size)):
            if current_step == step:
                xs.append(i * p_size)
                ys.append(j * p_size)
                current_step = -1
            current_step += 1
    return xs, ys


def img_format(image):
    return image.resize((700, 700))


def predictor(y_predicts):
    sum1 = sum(y_predicts)
    sum0 = len(y_predicts - sum1)
    return 1 if sum1 >= sum0 else 0


if __name__ == "__main__":
    print("COLOR")
    classifier = GaussianNB()
    color_descr = HistogramDescriptor(q_factor=8)
    color_descr.construct()
    print(color_descr.train_test_evaluate(classifier))

    print("TEXTURE")
    classifier = KNeighborsClassifier(n_neighbors=5, weights="distance")
    descr = TextureDescriptor(["contrast", "dissimilarity", "homogeneity"], 175, img_format, get_patch_coordinates)
    descr.construct()
    descr.construct_y()
    descr.n_target = len(descr.X[0])  # car problème avec patch coordinates
    print(descr.train_test_evaluate(classifier, predictor))

    print("FORME")
    classifier = SVC()
    descr = HogDescriptor()
    descr.construct()
    print(descr.train_test_evaluate(classifier))
