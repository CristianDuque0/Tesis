import pandas as pd
import seaborn as sns
import matplotlib . pyplot as plt
data_pf = pd.read_csv('pruebas.csv')  
print(data_pf.describe())
sns.set ( style = "ticks" )
sns.pairplot(data_pf, hue="genero")
plt.savefig ('pixeles.pdf', dpi =400, bbox_inches =  'tight' , pad_inches=0.1)