from scipy.interpolate import CubicSpline,PchipInterpolator, Akima1DInterpolator, interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path= 'C:/Users/PMLS/Desktop/Hassaan Digitrain/Hassaan Digitain/Claculated-Gc and MMR.xlsx'


df = pd.read_excel(file_path)

# Extract the columns MMR, G_1, and G_2
mmr_list = df['MMR'].tolist()
g1_list = df['G_I'].tolist()
g2_list = df['G_II'].tolist()


mmr_list=[0.00,0.397445672096082, 0.397235792741896, 0.397456001539942, 0.222491719368064, 0.222479300424964, 0.222720535682272, 0.399006474721271,1.00]
sorted_mmr=sorted(mmr_list)

orignal_mmr=[0.00,0.397445672096082, 0.397235792741896, 0.397456001539942, 0.222491719368064, 0.222479300424964, 0.222720535682272, 0.399006474721271,1.00]
sorted_mmr=[0.0, 0.222479300424964, 0.222491719368064, 0.222720535682272, 0.397235792741896, 0.397445672096082, 0.397456001539942, 0.399006474721271, 1.0]


orignal_g2=[0,0.45454276068680965, 0.4223887632202626, 0.4631718610826947, 0.19511334848932155, 0.20469813899811715, 0.20332157534125056, 0.5260369637076783,0.594]
sorted_g2=[0, 0.20469813899811715, 0.19511334848932155, 0.20332157534125056, 0.4223887632202626, 0.45454276068680965, 0.4631718610826947,0.5260369637076783,0.594]

orignal_g1= [0.24,0.6891173483530103, 0.6409312369860374, 0.7021693572864753, 0.6818332140321766, 0.7153790933876258, 0.7095784171017654, 0.7923300228810617,0]
sorted_g1=[0.24, 0.7153790933876258,0.6818332140321766, 0.7095784171017654, 0.6409312369860374,0.6891173483530103, 0.7021693572864753,0.7923300228810617,0]


MMR=np.array(sorted_mmr)
G1=np.array(sorted_g1)
G2=np.array(sorted_g2)

#overshooting
#G1fn=CubicSpline(MMR,G1)
#G2fn=CubicSpline(MMR,G2)
#plt.title('CubicSpline')

#better results
G1fn=PchipInterpolator(MMR,G1)
G2fn=PchipInterpolator(MMR,G2)
plt.title('PchipInterpolator')

#not good results
#G1fn=Akima1DInterpolator(MMR,G1)
#G2fn=Akima1DInterpolator(MMR,G2)
#plt.title('Akima1DInterpolator')

#G1fn=interp1d(MMR,G1,kind='linear')
#G2fn=interp1d(MMR,G2,kind='linear')
#plt.title('Linear Interpolation')

mmr_range=np.linspace(0,1,num=100)
G1_val=G1fn(mmr_range)
G2_val=G2fn(mmr_range)

plt.scatter(MMR,G1,label='GI_orignal', color='blue')
plt.scatter(MMR,G2,label='GII_orignal',color="green")
plt.plot(mmr_range,G1_val,label='spline G1', color='red')
plt.plot(mmr_range,G2_val,label='spline G2', color='yellow')
plt.xlabel('MMR')
plt.ylabel('G1 and G2')

plt.legend()
plt.show()