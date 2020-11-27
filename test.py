import ALICE
import portalocker
import time
import pandas

start = time.time()
for i in range(1000):
    my_df = pandas.DataFrame({'name':['JS'],'time':[time.time()]})
    df_agg = ALICE.load_agg_save_safe('test_csv.csv',my_df)
print(time.time()-start,len(df_agg))
