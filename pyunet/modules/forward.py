import sys
import os
import torch
import cv2
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.unet import UNet

class Forward:
    def __init__(self, params={}):
        self.params = params

        self.device     = params.get('device')
        self.gpu_index  = params.get('gpu_index')
        self.input_img  = params.get('input_img')
        self.model_file = params.get('model_file')

    def execute(self):
        print("Forwarding for {}...".format(self.input_img))

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        print("Loading model {}...".format(self.model_file))
        state = torch.load(self.model_file)

        saved_params = state['params']

        img_width   = saved_params.get('img_width')
        img_height  = saved_params.get('img_height')
        dim         = (img_width, img_height)

        in_channels     = saved_params.get('in_channels') or 3
        out_channels    = saved_params.get('out_channels') or 3
        features        = saved_params.get('features') or [64, 128, 256, 512]

        model   = UNet(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    features=features
                  ).to(self.device)

        model.load_state_dict(state['state_dict'])

        # Load image
        img = cv2.imread(self.input_img)
        rows, cols, _ = img.shape
        original_dim = (cols, rows)

        image = (cv2.resize(img, dim) / 255).transpose((2, 0, 1))
        tensor_image = torch.Tensor(image)

        x = torch.tensor([tensor_image.numpy()]).to(self.device)

        result = cv2.resize(model.forward(x).detach().cpu().numpy()[0].transpose(1, 2, 0), original_dim)

        cv2.imshow("result", result)
        cv2.imshow("Original", img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
