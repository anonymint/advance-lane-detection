import numpy as np

class Line():

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        # self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        # self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # radius of curvature of the line in some units
        # self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        # self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        # self.allx = None
        # y values for detected line pixels
        # self.ally = None
        self.number_frames_kept = 10

    def update(self, fit):
        if fit is not None:
            if self.best_fit is not None:
                self.diffs = abs(fit - self.best_fit)

            if (self.diffs[0] > 0.001 or self.diffs[1] > 1.0 or self.diffs[2] > 100.) and len(self.current_fit) > 0:
                self.detected = False
            else:
                self.detected = True
                self.current_fit.append(fit)
                if len(self.current_fit) > self.number_frames_kept:
                    self.current_fit = self.current_fit[len(self.current_fit) - self.number_frames_kept:]
                self.best_fit = np.average(self.current_fit, axis=0)

        # # or remove one from the history, if not found
        # else:
        #     self.detected = False
        #     if len(self.current_fit) > 0:
        #         self.current_fit = self.current_fit[1:len(self.current_fit)]
        #     if len(self.current_fit) > 0:
        #         # if there are still any fits in the queue, best_fit is their average
        #         self.best_fit = np.average(self.current_fit, axis=0)
