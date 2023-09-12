import numpy as np
import pandas as pd
import cupy as cp
from tqdm import tqdm
from test_config import CFG
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from additional_models import ResU_Net
from segmentation_models_pytorch.encoders.mix_transformer import MixVisionTransformerEncoder
xp = cp

IS_DEBUG = False
mode = 'train' if IS_DEBUG else 'test'
TH=0.5

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    # pixels = (pixels >= thr).astype(int)
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

delta_lookup = {
    "xx": xp.array([[1, -2, 1]], dtype=float),
    "yy": xp.array([[1], [-2], [1]], dtype=float),
    "xy": xp.array([[1, -1], [-1, 1]], dtype=float),
}

def operate_derivative(img_shape, pair):
    assert len(img_shape) == 2
    delta = delta_lookup[pair]
    fft = xp.fft.fftn(delta, img_shape)
    return fft * xp.conj(fft)

def soft_threshold(vector, threshold):
    return xp.sign(vector) * xp.maximum(xp.abs(vector) - threshold, 0)

def back_diff(input_image, dim):
    assert dim in (0, 1)
    r, n = xp.shape(input_image)
    size = xp.array((r, n))
    position = xp.zeros(2, dtype=int)
    temp1 = xp.zeros((r+1, n+1), dtype=float)
    temp2 = xp.zeros((r+1, n+1), dtype=float)
    
    temp1[position[0]:size[0], position[1]:size[1]] = input_image
    temp2[position[0]:size[0], position[1]:size[1]] = input_image
    
    size[dim] += 1
    position[dim] += 1
    temp2[position[0]:size[0], position[1]:size[1]] = input_image
    temp1 -= temp2
    size[dim] -= 1
    return temp1[0:size[0], 0:size[1]]

def forward_diff(input_image, dim):
    assert dim in (0, 1)
    r, n = xp.shape(input_image)
    size = xp.array((r, n))
    position = xp.zeros(2, dtype=int)
    temp1 = xp.zeros((r+1, n+1), dtype=float)
    temp2 = xp.zeros((r+1, n+1), dtype=float)
        
    size[dim] += 1
    position[dim] += 1

    temp1[position[0]:size[0], position[1]:size[1]] = input_image
    temp2[position[0]:size[0], position[1]:size[1]] = input_image
    
    size[dim] -= 1
    temp2[0:size[0], 0:size[1]] = input_image
    temp1 -= temp2
    size[dim] += 1
    return -temp1[position[0]:size[0], position[1]:size[1]]

def iter_deriv(input_image, b, scale, mu, dim1, dim2):
    g = back_diff(forward_diff(input_image, dim1), dim2)
    d = soft_threshold(g + b, 1 / mu)
    b = b + (g - d)
    L = scale * back_diff(forward_diff(d - b, dim2), dim1)
    return L, b

def iter_xx(*args):
    return iter_deriv(*args, dim1=1, dim2=1)

def iter_yy(*args):
    return iter_deriv(*args, dim1=0, dim2=0)

def iter_xy(*args):
    return iter_deriv(*args, dim1=0, dim2=1)

def iter_sparse(input_image, bsparse, scale, mu):
    d = soft_threshold(input_image + bsparse, 1 / mu)
    bsparse = bsparse + (input_image - d)
    Lsparse = scale * (d - bsparse)
    return Lsparse, bsparse

def denoise_image(input_image, iter_num=100, fidelity=150, sparsity_scale=10, continuity_scale=0.5, mu=1):
    image_size = xp.shape(input_image)
    #print("Initialize denoising")
    norm_array = (
        operate_derivative(image_size, "xx") + 
        operate_derivative(image_size, "yy") + 
        2 * operate_derivative(image_size, "xy")
    )
    norm_array += (fidelity / mu) + sparsity_scale ** 2
    b_arrays = {
        "xx": xp.zeros(image_size, dtype=float),
        "yy": xp.zeros(image_size, dtype=float),
        "xy": xp.zeros(image_size, dtype=float),
        "L1": xp.zeros(image_size, dtype=float),
    }
    g_update = xp.multiply(fidelity / mu, input_image)
    for i in tqdm(range(iter_num), total=iter_num):
        #print(f"Starting iteration {i+1}")
        g_update = xp.fft.fftn(g_update)
        if i == 0:
            g = xp.fft.ifftn(g_update / (fidelity / mu)).real
        else:
            g = xp.fft.ifftn(xp.divide(g_update, norm_array)).real
        g_update = xp.multiply((fidelity / mu), input_image)
        
        #print("XX update")
        L, b_arrays["xx"] = iter_xx(g, b_arrays["xx"], continuity_scale, mu)
        g_update += L
        
        #print("YY update")
        L, b_arrays["yy"] = iter_yy(g, b_arrays["yy"], continuity_scale, mu)
        g_update += L
        
        #print("XY update")
        L, b_arrays["xy"] = iter_xy(g, b_arrays["xy"], 2 * continuity_scale, mu)
        g_update += L
        
        #print("L1 update")
        L, b_arrays["L1"] = iter_sparse(g, b_arrays["L1"], sparsity_scale, mu)
        g_update += L
        
    g_update = xp.fft.fftn(g_update)
    g = xp.fft.ifftn(xp.divide(g_update, norm_array)).real
    
    g[g < 0] = 0
    g -= g.min()
    g /= g.max()
    return g

def read_image(fragment_id):
    images = []

    mid = 30 # 65 // 2 , 28, 30
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    
    if CFG.in_chans%2 != 0:
        end+=1

    idxs = range(start, end)

    for i in tqdm(idxs):
        
        image = cv2.imread(CFG.comp_dataset_path + f"{mode}/{fragment_id}/surface_volume/{i:02}.tif", 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)
    
    return images

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug

class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # return len(self.xyxys)
        return len(self.images)

    def __getitem__(self, idx):
        # x1, y1, x2, y2 = self.xyxys[idx]
        image = self.images[idx]
        data = self.transform(image=image)
        image = data['image']
        return image

def make_test_dataset(fragment_id):
    test_images = read_image(fragment_id)
    
    x1_list = list(range(0, test_images.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, test_images.shape[0]-CFG.tile_size+1, CFG.stride))
    
    test_images_list = []
    xyxys = []
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            
            test_images_list.append(test_images[y1:y2, x1:x2])
            xyxys.append((x1, y1, x2, y2))
    xyxys = np.stack(xyxys)
            
    test_dataset = CustomDataset(test_images_list, CFG, transform=get_transforms(data='valid', cfg=CFG))
    
    test_loader = DataLoader(test_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    return test_loader, xyxys

class CustomModel(nn.Module):
    def __init__(self, cfg, backb, weight=None):
        super().__init__()
        self.cfg = cfg
        
        if backb in ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5']:
            in_chans = 3
        else:
            in_chans = 3
        
        if backb in ['mit_b2_s', 'mit_b3_s', 'mit_b4_s']:
            x = backb[:6]
            self.encoder = torch.load(f"/kaggle/input/dad-mit-models/Unet___{x}_chans_27_30_fold_3_best.pth")['encoder']
        
        elif backb in ['tu-seresnext26d_32x4d', 'tu-skresnext50_32x4d', 'tu-res2net50_26w_6s']:
            self.encoder = smp.Unet(
                encoder_name=backb, 
                encoder_weights=weight,
                decoder_attention_type=None,
                in_channels=in_chans,
                classes=cfg.target_size,
                activation=None,
            )
        else:
            self.encoder = smp.Unet(
                encoder_name=backb, 
                encoder_weights=weight,
                in_channels=in_chans,
                classes=cfg.target_size,
                activation=None,
            )

    def forward(self, image):
        output = self.encoder(image)
        output = output.squeeze(-1)
        return output

def build_model(cfg, backb, weight="None"):
    print('model_name', cfg.model_name)
    print('backbone', backb)
    
    if backb == 'ResUNet':
        model = ResU_Net(img_ch=6,output_ch=1)
    else:
        model = CustomModel(cfg, backb, weight)
    
    return model

class EnsembleModel:
    def __init__(self, use_tta=False):
        self.models = []
        self.use_tta = use_tta

    def __call__(self, x):
        outputs=[]
        THR = 0.3
        for model in self.models:
            if type(model) == type(ResU_Net(img_ch=6,output_ch=1)):
                outputs.append(torch.sigmoid(model(x)).to('cpu').numpy())
            elif type(model.encoder.encoder) == MixVisionTransformerEncoder:
                count=1
                x_ = torch.sigmoid(model(x[:, 0:3, :, :]))
                for i in range(4, CFG.in_chans):
                    x_ += torch.sigmoid(model(x[:, i-3:i, :, :]))
                    count += 1
                
                outputs.append((x_/count).to('cpu').numpy())
            else:
                count=1
                x_ = torch.sigmoid(model(x[:, 0:3, :, :]))
                for i in range(4, CFG.in_chans):
                    x_ += torch.sigmoid(model(x[:, i-3:i, :, :]))
                    count += 1
                
                outputs.append((x_/count).to('cpu').numpy())
        avg_preds = np.mean(outputs, axis=0)
        return avg_preds

    def add_model(self, model):
        self.models.append(model)

def build_ensemble_model(device):
    model = EnsembleModel()
    for backb in CFG.backbone:
        ffolds = []
        if backb in ['mit_b2', 'mit_b3', 'ResUNet', 'mit_b4', 'mit_b5']:
            ffolds = [1, 2, 3]
        else:
            ffolds = [1, 2, 3]
        
        for fold in ffolds:
            _model = build_model(CFG, backb, weight=None)
            _model.to(device)
            flag = False
            
            if backb == 'ResUNet':
                model_path = f'/kaggle/input/vesuvius-models-6fold/ResUNet/Unet_fold{fold}_best.pth'
            elif backb in ['tu-regnety_064', 'tu-resnest50d_4s2x40d', 'resnet50', 'resnet34']:
                model_path = f'/kaggle/input/vesuvius-models-3/{backb}/Unet_fold{fold}_best.pth'
            elif backb in ['mit_b2_s', 'mit_b3_s', 'mit_b4_s']:
                x = backb[:6]
                model_path = f'/kaggle/input/dad-mit-models/Unet___{x}_chans_27_30_fold_{fold}_best.pth'
                flag = True
            elif backb=='tu-seresnextaa101d_32x8d':
                model_path = f'/kaggle/input/vesuvius-models/seresnextaa101d_32x8d/{CFG.model_name}_fold{fold}_best.pth'
            elif backb.startswith("mit_b") or backb=='se_resnext50_32x4d':
                if backb in ['se_resnext50_32x4d', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5']:
                    model_path = f'/kaggle/input/vesuvius-models/{backb}_30/{CFG.model_name}_fold{fold}_best.pth'
                else:
                    model_path = f'/kaggle/input/vesuvius-models-6fold/{backb}/Unet_fold{fold}_best.pth'
            elif backb == 'tu-seresnext26d_32x4d':
                model_path = f'/kaggle/input/vesuvius-models/{backb}/{CFG.model_name}_fold{fold}_best.pth'
            else:
                model_path = f'/kaggle/input/vesuvius-models/{backb}/{CFG.model_name}_fold{fold}_best.pth'
            
            if flag:
                state = torch.load(model_path)['weights']
            else:
                state = torch.load(model_path)['model']
            
            _model.load_state_dict(state)
            _model.eval()

            model.add_model(_model)
    
    return model

def TTA(x:torch.Tensor,model:nn.Module, device):
    # x.shape=(batch,c,h,w)
    shape=x.shape
    rot = [1, 3] # How much to rotate the fragments for TTA
    x=[torch.rot90(x,k=i,dims=(-2,-1)) for i in rot]
    x=torch.cat(x,dim=0)
    x=model(x)
    x = torch.from_numpy(x).to(device)
    x=x.reshape(len(rot),shape[0],1,*shape[-2:])
    x=[torch.rot90(x[count],k=-i,dims=(-2,-1)) for count, i in enumerate(rot)]
    x=torch.stack(x,dim=0)
    return x.mean(0)