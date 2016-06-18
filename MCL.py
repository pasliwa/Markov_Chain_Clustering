import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio


def load_matrix(file_name, matrix_type, matrix_name="matrix"):
    if matrix_type == "txt":
        return np.loadtxt(file_name).astype(float)
    elif matrix_type == "npy":
        return np.load(file_name).astype(float)
    else: # matrix_type == "mat"
        return sio.loadmat(file_name)[matrix_name]


def save_matrix(matrix, matrix_type, save_file="MCL_output"):
    if matrix_type == "txt":
        np.savetxt(save_file, matrix)
    elif matrix_type == "npy":
        np.save(save_file, matrix)
    else: # matrix_type == "mat"
        sio.savemat(save_file, {'matrix': matrix})


def expand(matrix):
    return matrix.dot(matrix)


def normalize(matrix):
    if sps.issparse(matrix):
        spsCopy = matrix.copy()
        for column_i in range(spsCopy.shape[1]):
            spsCopy[:, column_i] = spsCopy[:, column_i] / float(sum(spsCopy[:, column_i]).data)
        return spsCopy
    else:
        return matrix / sum(matrix)


def inflate(matrix, r):
    if sps.issparse(matrix):
        return matrix.power(r)
    else:
        return matrix ** r


def norm(matrix):
    if sps.issparse(matrix):
        return sum(sum(np.absolute(matrix)).data)
    else:
        return sum(sum(np.absolute(matrix)))


def mcl(matrix, n, r, e):
    differences = []
    matrix = normalize(matrix)
    last = matrix.copy()
    i = 0
    while i < n:
        matrix = expand(matrix)
        matrix = inflate(matrix, r)
        matrix = normalize(matrix)
        difference = norm(matrix - last)
        if difference <= e:
            break
        differences.append(difference)
        last = matrix.copy()
        i += 1
    return (last, differences)



# parsing
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, help="number of steps", default=8)
parser.add_argument("-r", type=float, help="exponent for inflating", default=2)
parser.add_argument("-e", type=float, help="epsilon", default=0.1)
parser.add_argument("--file_name", type=str, help="name of the matrix file", default="matrix.txt")
parser.add_argument("--matrix_type", type=str, help="type of the matrix file [txt/npy/mat]", default="txt")
parser.add_argument("--matrix_name", type=str, help="name of the matrix in mat dictionary", default="matrix")
parser.add_argument("--plot", type=bool, help="plot |M(i+2) - M(i)| [True/False]", default=False)
parser.add_argument("--save_matrix_file", type=str, help="name of the file in which matrix will be saved", default="MCL_outcome")
parser.add_argument("--save_matrix_type", type=str, help="type of the file in which matrix will be saved [txt/npy/mat]", default="None")

args = parser.parse_args()
n = args.n
r = args.r
e = args.e
file_name = args.file_name
matrix_type = args.matrix_type
matrix_name = args.matrix_name
plot = args.plot
save_matrix_file = args.save_matrix_file
save_matrix_type = args.save_matrix_type

matrix = load_matrix(file_name, matrix_type, matrix_name)

outcome, differences = mcl(matrix, n, r, e)

if save_matrix_type == "None":
    save_matrix_type = matrix_type

save_matrix(outcome, save_matrix_type, save_matrix_file)

plt.imshow(outcome, interpolation="None")
plt.show()

if plot:
    x = list(range(len(differences)))
    plt.plot(x, differences)
    plt.show()

