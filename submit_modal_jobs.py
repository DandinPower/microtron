import pathlib
import modal

from microtron import playground

app = modal.App("microtron")

image = modal.Image.debian_slim(python_version="3.12") \
    .uv_pip_install(
        "numpy==2.3.2",
        "tiktoken==0.11.0",
        "torch==2.8.0",
        "transformers==4.56.1"
    ).env({
        "HF_HOME":"/model_cache"
    }).add_local_python_source(
        "microtron"
    )

volume = modal.Volume.from_name("microtron-caches", create_if_missing=True)

VOL_MOUNT_PATH = pathlib.Path("/model_cache")

@app.function(gpu="A100-80GB", image=image, volumes={VOL_MOUNT_PATH: volume}, secrets=[modal.Secret.from_name("huggingface-access-token")],)
def modal_function_wrapper():
    playground.main()    

@app.local_entrypoint()
def main():
    modal_function_wrapper.remote()