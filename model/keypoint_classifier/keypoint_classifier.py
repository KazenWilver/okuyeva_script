import numpy as np
import pickle


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.pkl',
    ):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def __call__(self, landmark_list):
        probs = self.model.predict_proba([landmark_list])[0]
        idx = np.argmax(probs)
        result_index = int(self.model.classes_[idx])
        confidence = float(probs[idx])
        return result_index, confidence
