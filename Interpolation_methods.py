'''from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

mmr = [0, 0.2224793, 0.222491719, 0.222720536, 0.397235793, 0.397445672, 0.397456002, 0.399006475]
G_1 = [0.24, 0.715379093, 0.681833214, 0.709578417, 0.64093123, 0.689117348, 0.702169357, 0.792330023]
G_2 = [0.594, 0.204698139, 0.195113348, 0.203321575, 0.422388763, 0.454542761, 0.463171861, 0.52603696]
G_c = np.add(G_1, G_2)
G_c5 = [0.83, 0.920077232, 0.876946563, 0.912899992, 1.06332, 1.143660109, 1.165341218, 1.318366987]

mmr_new = np.arange(0, 0.4, 0.001)

f1 = interp1d(mmr, G_1)
f2 = interp1d(mmr, G_2)
f3 = interp1d(mmr, G_c)
import pandas as pd
df = pd.DataFrame({
    'mmr': mmr_new, 
    'G_1 interpolated': f1(mmr_new),
    'G_2 interpolated': f2(mmr_new),
    'G_3 interpolated': f3(mmr_new),
})

df.to_excel('linearinterpolation.xlsx', index=False, engine='openpyxl')

print("mmr:", mmr_new)
print("G_1 interpolated:", f1(mmr_new))
print("G_2 interpolated:", f2(mmr_new))
print("G_3 interpolated:", f3(mmr_new))

fig = plt.figure()
plt.plot(mmr_new, f1(mmr_new), 'b', label='G_1')
plt.plot(mmr_new, f2(mmr_new), 'r', label="G_2")
plt.plot(mmr_new, f3(mmr_new), 'g', label="G_C")
plt.title('Linear Interpolation')
plt.grid()
plt.xlabel('mmr')
plt.ylabel('G')
plt.legend()
plt.show()'''

#################CUBIC SPLINE INTERPOLATION############################
'''
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

mmr = [0, 0.2224793, 0.222491719, 0.222720536, 0.397235793, 0.397445672, 0.397456002, 0.399006475]
G_1 = [0.24, 0.715379093, 0.681833214, 0.709578417, 0.64093123, 0.689117348, 0.702169357, 0.792330023]
G_2 = [0.594, 0.204698139, 0.195113348, 0.203321575, 0.422388763, 0.454542761, 0.463171861, 0.52603696]
G_c = np.add(G_1, G_2)
G_c5 = [0.83, 0.920077232, 0.876946563, 0.912899992, 1.06332, 1.143660109, 1.165341218, 1.318366987]

mmr_new = np.arange(0, 0.4, 0.01)

f1 = CubicSpline(mmr, G_1)
f2 = CubicSpline(mmr, G_2)
f3 = CubicSpline(mmr, G_c)
import pandas as pd
df = pd.DataFrame({
    'mmr': mmr_new, 
    'G_1 interpolated': f1(mmr_new),
    'G_2 interpolated': f2(mmr_new),
    'G_3 interpolated': f3(mmr_new),
})

df.to_excel('cubicsplineinterpolation.xlsx', index=False, engine='openpyxl')

fig = plt.figure()
plt.plot(mmr_new, f1(mmr_new), 'b', label='G_1')
plt.plot(mmr_new, f2(mmr_new), 'r', label="G_2")
plt.plot(mmr_new, f3(mmr_new), 'g', label="G_C")
plt.title('Cubic Spline Interpolation')
plt.grid()
plt.xlabel('mmr')
plt.ylabel('G')
plt.legend()
plt.show()'''

#####################LAGRANGE INTERPOLATION###########################
'''from scipy.interpolate import lagrange
import numpy as np
import matplotlib.pyplot as plt

mmr = [0, 0.2224793, 0.222491719, 0.222720536, 0.397235793, 0.397445672, 0.397456002, 0.399006475]
G_1 = [0.24, 0.715379093, 0.681833214, 0.709578417, 0.64093123, 0.689117348, 0.702169357, 0.792330023]
G_2 = [0.594, 0.204698139, 0.195113348, 0.203321575, 0.422388763, 0.454542761, 0.463171861, 0.52603696]
G_c = np.add(G_1, G_2)
G_c5 = [0.83, 0.920077232, 0.876946563, 0.912899992, 1.06332, 1.143660109, 1.165341218, 1.318366987]

mmr_new = np.arange(0, 0.4, 0.002)

f1 = lagrange(mmr, G_1)
f2 = lagrange(mmr, G_2)
f3 = lagrange(mmr, G_c)
import pandas as pd
df = pd.DataFrame({
    'mmr': mmr_new, 
    'G_1 interpolated': f1(mmr_new),
    'G_2 interpolated': f2(mmr_new),
    'G_3 interpolated': f3(mmr_new),
})

df.to_excel('lagrangeinterpolation.xlsx', index=False, engine='openpyxl')

fig = plt.figure()
plt.plot(mmr_new, f1(mmr_new), 'b', label='G_1')
plt.plot(mmr_new, f2(mmr_new), 'r', label="G_2")
plt.plot(mmr_new, f3(mmr_new), 'g', label="G_C")
plt.title('Lagrange Interpolation')
plt.grid()
plt.xlabel('mmr')
plt.ylabel('G')
plt.legend()
plt.show()'''
##########NEWTON"S DIVIDED DIFFERENCE##############################
'''
import numpy as np
import matplotlib.pyplot as plt

def divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    # the first column is y
    coef[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = \
           (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
            
    return coef

def newton_poly(coef, x_data, x):
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1,n+1):
        p = coef[n-k] + (x -x_data[n-k])*p
    return p


mmr = np.array([0, 0.2224793, 0.222491719, 0.222720536, 0.397235793, 0.397445672, 0.397456002, 0.399006475])
G_1 = np.array([0.24, 0.715379093, 0.681833214, 0.709578417, 0.64093123, 0.689117348, 0.702169357, 0.792330023])
G_2 = np.array([0.594, 0.204698139, 0.195113348, 0.203321575, 0.422388763, 0.454542761, 0.463171861, 0.52603696])
G_c = np.add(G_1, G_2)
G_c5 = np.array([0.83, 0.920077232, 0.876946563, 0.912899992, 1.06332, 1.143660109, 1.165341218, 1.318366987])


# get the divided difference coef
G1_s = divided_diff(mmr, G_1)[0, :]
G2_s = divided_diff(mmr, G_2)[0, :]
Gc_s = divided_diff(mmr, G_c)[0, :]
# evaluate on new data points
mmr_new = np.arange(0, 0.4, 0.002)

G_1_new = newton_poly(G1_s, mmr, mmr_new)
G_2_new = newton_poly(G2_s, mmr, mmr_new)
G_c_new = newton_poly(Gc_s, mmr, mmr_new)
import pandas as pd
df = pd.DataFrame({
    'mmr': mmr_new, 
    'G_1 interpolated': G_1_new,
    'G_2 interpolated': G_2_new,
    'G_c interpolated': G_c_new
})

df.to_excel('divided_difference_interpolation.xlsx', index=False, engine='openpyxl')

fig = plt.figure()
plt.plot(mmr_new, G_1_new, 'b', label='G_1')
plt.plot(mmr_new, G_2_new, 'r', label="G_2")
plt.plot(mmr_new, G_c_new, 'g', label="G_C")
plt.title("Newton's divided difference Interpolation")
plt.grid()
plt.xlabel('mmr')
plt.ylabel('G')
plt.legend()
plt.show()'''

###########BARYCENTRIC INTERPOLATION###############################

from scipy.interpolate import barycentric_interpolate
import numpy as np
import matplotlib.pyplot as plt

mmr = [0, 0.2224793, 0.222491719, 0.222720536, 0.397235793, 0.397445672, 0.397456002, 0.399006475]
G_1 = [0.24, 0.715379093, 0.681833214, 0.709578417, 0.64093123, 0.689117348, 0.702169357, 0.792330023]
G_2 = [0.594, 0.204698139, 0.195113348, 0.203321575, 0.422388763, 0.454542761, 0.463171861, 0.52603696]
G_c = np.add(G_1, G_2)
G_c5 = [0.83, 0.920077232, 0.876946563, 0.912899992, 1.06332, 1.143660109, 1.165341218, 1.318366987]

mmr_new = np.arange(0, 0.4, 0.002)
import pandas as pd
df = pd.DataFrame({
    'mmr': mmr_new, 
    'G_1 interpolated': barycentric_interpolate(mmr, G_1, mmr_new),
    'G_2 interpolated': barycentric_interpolate(mmr, G_2, mmr_new),
    'G_3 interpolated': barycentric_interpolate(mmr, G_c, mmr_new)
})

df.to_excel('barycentricinterpolation.xlsx', index=False, engine='openpyxl')

fig = plt.figure()
plt.plot(mmr_new, barycentric_interpolate(mmr, G_1, mmr_new), 'b', label='G_1')
plt.plot(mmr_new, barycentric_interpolate(mmr, G_2, mmr_new), 'r', label="G_2")
plt.plot(mmr_new, barycentric_interpolate(mmr, G_c, mmr_new), 'g', label="G_C")
plt.title('Barycentric Interpolation')
plt.grid()
plt.xlabel('mmr')
plt.ylabel('G')
plt.legend()
plt.show()


########################GETTING THE VALUES#####################
