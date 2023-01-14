import torch
from torchvision import io
from torchvision import transforms as T
from PIL import Image
from semseg.models import *
from semseg.datasets import *


def create_model(model_path):
    model = eval('SegFormer')(
        backbone='MiT-B3',
        num_classes=150
    )
    model.load_state_dict(torch.load('./segformer.b3.ade.pth', map_location='cpu'))
    model.eval()
    return model


def detect_road(model, image_path = None, image = None):
    """
        image_path: String
        image     : Numpy array (H, W, 3)
        Give either input

        Returns:
        Numpy array (H, W, 3) with everything other than road masked in white
    """
    if image_path is not None:
        image = io.read_image(image_path)
    if image is not None:
        image = torch.Tensor(image).permute(2, 0, 1)

    # resize
    image = T.Resize((512, 512))(image)
    # scale to [0.0, 1.0]
    image = image.float() / 255
    # normalize
    image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
    # add batch size
    image = image.unsqueeze(0)


    palette = eval('ADE20K').PALETTE
    with torch.inference_mode():
        seg = model(image)
    seg = seg.softmax(1).argmax(1).to(int)
    seg.unique()
    seg_map = palette[seg].squeeze().to(torch.uint8)

    seg_map = seg_map.numpy()
    image = torch.permute(image[0], (1,2,0)).numpy()
    h, w, c = image.shape 
    print(h,w,c)
    for i in range(h):
        for j in range(w):
            if i==100 and j == 0:
                print(seg_map[i, j])
            if (list(seg_map[i,j]) != [120,120,70]):
                image[i, j] = [255,255,255]

    return image
