import numpy as np

class Feature_Scaling(object):
    '''
    Build a class to do feature scaling and store the constants for validation and test sets
    '''
    def __init__(self):
        '''
        method_id: defines a unique scaling mathod for each feature
        consts: keeps a dictionary of the scaling methods
        '''
        self.method_id = 0
        self.consts = {}
        self.methods = {}
        self.method_dict = {}
    def map_scaling(self, df):
        '''
        Map the column name with the scaling methods used in training sets
        '''
        for fname in df.columns:
            for method_id in self.method_dict[fname]:
                try:
                    df[fname] = self.methods[method_id](df[fname], method_id = method_id)
                except:
                    raise NotImplementedError('Method %d for feature %s not found' %(method_id, fname))
        return df
    
    def add_method(self, fname):
        '''
        Add method id to the dictionary using the key of feature name
        '''
        self.method_id += 1
        if fname in self.method_dict:
            self.method_dict[fname].append(self.method_id)
        else:
            self.method_dict[fname] = [self.method_id]
            
    def linear_scaling(self, series, method_id = None):
        if method_id:
            return series.apply(lambda x: (x-self.consts[method_id][0])/self.consts[method_id][1]-1)
        self.add_method(series.name)
        my_min = series.min()
        my_max = series.max()
        scale = (my_max-my_min)/2
        self.methods[self.method_id] = self.linear_scaling
        self.consts[self.method_id] = (my_min, scale)
        return series.apply(lambda x: (x-my_min)/scale-1)
    
    def log_scaling(self, series, method_id = None):
        if method_id:
            return series.apply(lambda x: np.log(x+1))
        self.add_method(series.name)
        self.methods[self.method_id] = self.log_scaling
        return series.apply(lambda x: np.log(x+1))
    
    def clip(self, series, clip_min = None, clip_max = None, method_id = None):
        if method_id:
            return series.apply(lambda x: min(max(x, self.consts[method_id][0]), 
                                             self.consts[method_id][1]))
        self.add_method(series.name)
        self.methods[self.method_id] = self.clip
        self.consts[self.method_id] = (clip_min, clip_max)
        return series.apply(lambda x: min(max(x, clip_min), clip_max))
    
    def z_score_scaling(self, series, method_id = None):
        if method_id:
            return series.apply(lambda x: (x-self.consts[method_id][0])/self.consts[method_id][1])
        self.add_method(series.name)
        self.methods[self.method_id] =  self.z_score_scaling
        mu = series.mean()
        std = series.std()
        self.consts[self.method_id] = (mu, std)
        return series.apply(lambda x: (x-mu)/std)
    
    def binary_threshold(self, series, threshold = None, method_id = None):
        if method_id:
            return series.apply(lambda x: (1 if x > self.consts[method_id] else 0))
        self.add_method(series.name)
        self.methods[self.method_id] = self.binary_threshold
        self.consts[self.method_id] = threshold
        return series.apply(lambda x: (1 if x > threshold else 0))
