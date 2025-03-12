from modelscope import snapshot_download

model_dir = snapshot_download('unsloth/llama-3-8b-bnb-4bit',cache_dir="./model")

