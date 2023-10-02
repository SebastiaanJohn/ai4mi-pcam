from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger


class LogPredictionSamplesCallback(Callback):
    """Callback to log prediction samples to W&B."""

    def __init__(self, wandb_logger: WandbLogger) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 5 sample image predictions from the first batch
        if batch_idx == 0:
            n = 5
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])]


            # Option 1: log images with `WandbLogger.log_image`
            self.wandb_logger.log_image(
                key="sample_images",
                images=images,
                caption=captions)

