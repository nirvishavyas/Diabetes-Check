# -*- coding: utf-8 -*-



import pickle
import numpy as np
import pandas as pd
loaded_model = pickle.load(open(r"C:\Users\3511\Documents\data analist study material\Projrcts\diabetic or not\trained_model.sav", 'rb'))
user_input=(4,110,92,0,0,37.6,0.191,30)
input_array=np.asarray(user_input)
reshaped_array=input_array.reshape(1,-1)

#standardization
#x_standaredization_df=sc.fit_transform(reshaped_array)

# prediction
preds=loaded_model.predict(reshaped_array)
preds

if preds[0]==0:
    print("person is not diabetic")
else:
    print("person is diabetic")
