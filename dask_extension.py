from dask.diagnostics import ProgressBar

def pd_reset_index(ddf):
    ddf = ddf.assign(idx=1)
    ddf = ddf.set_index(ddf.idx.cumsum() - 1, sorted=True)
    ddf = ddf.map_partitions(lambda df: df.rename(index = {'idx': None}))
    ddf = ddf.drop('idx', axis=1)
    return ddf