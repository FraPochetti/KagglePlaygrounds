import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def model_fn(model_dir):
    print('Loading model')
    model = torch.jit.load('model.pt').eval().to(device)
    print(f'Is model on CUDA? {next(model.parameters()).is_cuda}')
    print('Model loaded successfully')
    return model

def predict_fn(image, model):
    print("Predicting...")
    with torch.inference_mode():
        mask = model(image.to(device))
    return mask