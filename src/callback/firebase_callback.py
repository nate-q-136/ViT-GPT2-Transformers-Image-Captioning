from firebase.utils import Firebase
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers import TrainerCallback


class FirebaseCallback(TrainerCallback):
    def __init__(
        self, firebase_credentials_path, storage_collection, storage_url, **kwargs
    ):
        self.firebase = Firebase(firebase_credentials_path, storage_url)
        self.storage_collection = storage_collection

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        epoch = state.epoch
        print(f"Uploading model to Firebase Storage at epoch {epoch}")

        local_folder_path = args.output_dir

        self.firebase.upload_files_to_storage_collection(
            local_folder_path, self.storage_collection
        )

        print(f"Model uploaded to Firebase Storage at epoch {epoch}")

        pass
