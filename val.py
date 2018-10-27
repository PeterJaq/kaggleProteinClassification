import numpy as np
import pandas as pd
from utils.DataLoad_K import DataSet
from model.InceptionV3 import Inception_V3
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image

BATCH_SIZE = 16
InceptionV3 = Inception_V3()
Data_input = DataSet()
model = InceptionV3.create_model()
model.summary()


submit = pd.read_csv('../data/sample_submission.csv')
predicted = []
draw_predict = []
model.load_weights('./checkpoint/InceptionV3_Focal_Loss.h5')




for name in submit['Id']:
    path = os.path.join('../data/perprocess/test/', name)
    image = Data_input.data_load(path, (299, 299, 3))/255.
    #score_predict, img_path_batch = model.predict_generator(Data_input.submit_generator())
    score_predict = model.predict(image[np.newaxis])[0]
    draw_predict.append(score_predict)
    label_predict = np.arange(28)[score_predict >= float(0.5)]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
#np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv('submit_InceptionV3_%f.csv' % 0.1, index=False)