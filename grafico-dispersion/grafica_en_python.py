import seaborn as sns
import matplotlib as mpl
import matplotlib . pyplot as plt
import pandas as pd
data = pd.read_csv ('union.csv')
sns.set ( style = "ticks" )
sns.pairplot ( data, hue="genero" )
plt.savefig ( 'pixeles.pdf', dpi =400, bbox_inches =  'tight' , pad_inches=0.1)
