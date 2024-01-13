import cv2
import timm
import torch
from torch import nn
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class AttractiveModel():
    def __init__(self, model_a_path, model_b_path=None, device="cpu"):
        self.features = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
       'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        self.device = device
        
        self.model_a = timm.create_model(
        'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        pretrained=False, num_classes=40
        )
        self.model_a.load_state_dict(torch.load(model_a_path))
        self.model_a.to(self.device)
        self.model_a.eval()

        data_config = timm.data.resolve_model_data_config(self.model_a)
        self.transforms_a = timm.data.create_transform(**data_config, is_training=False)

        self.model_b = timm.create_model(
        'vit_small_patch16_224.augreg_in21k',
        pretrained=False, num_classes=1
        )
        st = torch.load(model_b_path)
        self.model_b.load_state_dict(st)
        self.model_b.to(self.device)
        self.model_b.eval()

        data_config = timm.data.resolve_model_data_config(self.model_b)
        self.transforms_b = timm.data.create_transform(**data_config, is_training=False)
        
    def get_image_for_model(self, image):
        image = Image.fromarray(image)
        image_a = self.transforms_a(image).unsqueeze(0)
        image_b = self.transforms_a(image).unsqueeze(0)
        
        return image_a.to(self.device), image_b.to(self.device)

    def get_score_from_image(self, image):
        image_a, image_b = self.get_image_for_model(image)
        scores = nn.Sigmoid()(self.model_a(image_a)).detach().cpu().numpy()[0]
        score_b = float(self.model_b(image_b).detach().cpu().numpy()[0])
        
        res = {k:v for k,v in zip(self.features, scores)}
        res["Attractive_2"] = score_b
        res["Attractive_Mean"] = (res["Attractive"] + score_b/10)/2
    
        return res


if __name__ == "__main__":
    # attractive_model = AttractiveModel(model_a_path="models/model_a.pt",
    #                                    model_b_path="models/model_b.pt")
    
    # img_path = "bad.jpg"
    # image = cv2.imread(img_path)

    # print(attractive_model.get_score_from_image(image))
    try:
        attractive_model = AttractiveModel(model_a_path="models/model_a.pt",
                                           model_b_path="models/model_b.pt")
        print("success!")
    except Exception as e:
        print("something goes wrong: ", str(e))