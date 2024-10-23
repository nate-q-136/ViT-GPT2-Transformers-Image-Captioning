import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import firebase_admin.firestore
import os


def get_firebase_storage_client(firebase_credentials_path, storage_bucket_url):
    cred = credentials.Certificate(firebase_credentials_path)
    firebase_admin.initialize_app(cred, {"storageBucket": storage_bucket_url})
    return storage.bucket()


class Firebase:
    def __init__(self, firebase_credentials_path, storage_bucket_url):
        self.bucket = get_firebase_storage_client(
            firebase_credentials_path, storage_bucket_url
        )

    def upload_files_to_storage_collection(self, local_folder_path, storage_collection):
        try:
            for root, _, files in os.walk(local_folder_path):
                for file_name in files:
                    local_file_path = os.path.join(root, file_name)
                    # Create a storage blob (file reference) path in the specified storage collection
                    storage_blob_path = f"{storage_collection}/{file_name}"

                    blob = self.bucket.blob(storage_blob_path)
                    blob.upload_from_filename(local_file_path)
                    print(f"File {local_file_path} uploaded to {storage_blob_path}")

        except Exception as e:
            import traceback

            traceback.print_exc()

    def download_file(self, blob_name, local_path):
        """
        Downloads a single file from Firebase Storage.
        :param blob_name: Path of the file in Firebase Storage.
        :param local_path: Local path where the file should be saved.
        """
        blob = self.bucket.blob(blob_name)
        os.makedirs(
            os.path.dirname(local_path), exist_ok=True
        )  # Create directories if not exist
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob_name} to {local_path}")

    def download_files_from_storage_collection(
        self, local_folder_path, storage_collection
    ):
        try:
            # Ensure the local folder exists
            if not os.path.exists(local_folder_path):
                os.makedirs(local_folder_path)

            blobs = self.bucket.list_blobs(prefix=storage_collection)

            for blob in blobs:
                # blob: storage_collection/file
                file_bucket = blob.name.replace(storage_collection + "/", "")
                if file_bucket:
                    local_file_path = os.path.join(local_folder_path, file_bucket)
                    self.download_file(blob_name=blob.name, local_path=local_file_path)

        except Exception as e:
            import traceback

            traceback.print_exc()

        pass


if __name__ == "__main__":
    firebase = Firebase(
        firebase_credentials_path="/Volumes/Untitled 2 1/7-ViT-GPT2-Transformer/ViT-GPT2-Transformers-Image-Captioning/key.json",
        storage_bucket_url="chat-app-react-cc93a.appspot.com",
    )
    # firebase.upload_files_to_storage_collection(local_folder_path="/Volumes/Untitled 2 1/7-ViT-GPT2-Transformer/checkpoint-vit-gpt2", storage_collection="data")
    firebase.download_files_from_storage_collection(
        local_folder_path="/Volumes/Untitled 2 1/7-ViT-GPT2-Transformer/checkpoint-vit-gpt3",
        storage_collection="data",
    )
