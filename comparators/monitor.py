import numpy as np
class PairedEarlyStopMonitor:

    def __init__(self, window_length):
        self.window_length = window_length
        self.improvement = []

    def __call__(self, current_round, regressor, local):
        oob = ~local["sample_mask"]
        after = local["raw_predictions"][oob].ravel()
        stage = regressor.learning_rate * regressor.estimators_[current_round, 0].predict(local["X"][oob])
        y = local["y_oob_masked"]
        before = after - stage
        self.improvement.append(float(np.mean((y - before) ** 2) - np.mean((y - after) ** 2)))
        if current_round >= self.window_length - 1:
            return float(np.mean(self.improvement[-self.window_length:])) < 0
        return False
