from huggingface_hub import hf_hub_download

ruta = hf_hub_download(
    repo_id="Adit-jain/Soccana_Keypoint",
    filename="Model/weights/best.pt",
    local_dir="./modelos"
)
print(ruta)

