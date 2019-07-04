import numpy as np

import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from daily_sport import generate_model
from keras import backend as K


from utils.generic_utils import load_dataset_at

def inference(model:Model, dataset_id, dataset_prefix, dataset_fold_id=None, batch_size=128, test_data_subset=None,
              normalize_timeseries=False):
    _, _, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                          fold_index=dataset_fold_id,
                                                          normalize_timeseries=normalize_timeseries)
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    X_single_batch = X_test[0][np.newaxis, :, :]

    prediction = model.predict(X_single_batch)

    return np.around(prediction)[0]



if __name__ == "__main__":
    model = generate_model()
    model.load_weights('./weights/daily_sport_no_attention_weights_0.996.h5')
    prediction = inference(model, 15, dataset_prefix='daily_sport_no_attention', batch_size=1)
    print(prediction)
