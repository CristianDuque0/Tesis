import pandas as pd
df=pd.read_csv('vector.csv')
df.columns = ['mCt','mAe','mFt','mZt','mMFCC1','mMFCC2','mMFCC3','mMFCC4','mMFCC5','stdCt','stdAe','stdFt','stdZt','stdMFCC1','stdMFCC2','stdMFCC3','stdMFCC4','stdMFCC5']
genero = ['bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','bambuco','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','carranga','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo','pasillo']
df['genero']=genero
df.to_csv('vgraficar.csv', index=False)