from keras.callbacks import Callback
import os
import logging


# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s : %(name)s : %(message)s", level=logging.INFO)


class ManualStopCallback(Callback):

    def __init__(self, marker_file):
        self.marker_file = marker_file

    def on_epoch_end(self, epoch, logs={}):
        if os.path.isfile(self.marker_file):
            logger.info("Stopping training manually")
            # Delete file
            try:
                os.remove(self.marker_file)
            except Exception as e:
                logger.warn("Could not delete marker file: %s, error: %r" % (self.marker_file, e))
            # Stop training
            self.model.stop_training = True