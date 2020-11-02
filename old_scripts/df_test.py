import pandas as pd
import numpy as np
import time
#import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
if 0:
    N = 10000
    df = pd.DataFrame({'x':[np.random.randn(4) for i in range(N)],'inds':[i for i in range(N)],'w':[i for i in range(N,0,-1)],'z':[4 for i in range(N)]})
    df2 = pd.DataFrame({'x':[np.random.randn(4) for i in range(N)],'y':[2*i for i in range(N)]})

    df['x'] = df['x'].apply(lambda x: x.astype(np.float32))
    #print(np.stack(df['x'].values))
    #print(tf.concat(df['x'].values,axis=0))

    #df3 = df[['x','z'.combine(
    #print(df.loc[[1,1,1,2,2,2]])
    #df3 = pd.concat([df[['x','z']],df2['y']],axis=1)
    #print(df3)
    #print(pd.Series(df2.index,name='e_inds'))
    #df4 = df3.join(pd.Series(df2.index[:2],name='e_inds',index=[2,3]))
    #print(df4.merge(pd.Series(df2.index[:2],name='e_inds',index=[0,1]),'outer'))
    #print(pd.concat([df,df3],ignore_index=True))
    #e_inds = df.index[df['y']>2]
    #print(*df[['w','z']].loc[e_inds[0]])

    #df5 = pd.DataFrame(columns = ['x','y'])
    #df5.loc[0] = pd.Series({'x':1,'y':2,'z':3})
    #print(df5)

    df6 = pd.concat([df],ignore_index=True)
    #print(df6)
    #print(df2)
    #print(df2['y'].loc[df6['inds']])
    start = time.time()
    for i in range(10):
        df_train = pd.merge(df6,df2['y'].loc[df6['inds']].reset_index(drop=True),left_index=True,right_index=True)
    print(time.time()-start)
    def test_mutate(df_L):
        pass
    #print(df_train['y'][4532])
    #df_train = df6.copy(deep=True)
    #start = time.time()
    #for i in range(10):
        
    #    df_train['y'] = df_train.apply(lambda row : df2['y'].loc[row['inds']],axis=1)
    #print(time.time()-start)
    #print(df_train['y'][4532])

    df = df_train[:-len(df_train)]
    print(df)
    df['action'] = np.random.randint(6,size=(len(df),1))   #Sample action
    #df[['a','b','c']]=1,2,3
    print(df)

    print('action2' in df)

    print(df_train)
    df_train[['x','xx']] = df_train.apply(lambda row : (row['x'][:2] + row['x'][1:3],row['x'][:2] + row['x'][1:3]) ,axis=1,result_type='expand')
    print(df_train)


