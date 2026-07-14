import cv2
import torch
import torchvision.models as models
# import clip
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F

# Base Wrapper Class
class VisionTransformerWrapper:
    def __init__(self, model_name, device, smaller_edge_size=224, half_precision=False):
        self.device = device
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError("This method should be overridden in a subclass")
    
    def extract_features(self, img_tensor):
        raise NotImplementedError("This method should be overridden in a subclass")

# DINOv2 Wrapper
class DINOv2Wrapper(VisionTransformerWrapper):
    def load_model(self):

        model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        model.eval()

        # print(f"Loaded model: {self.model_name}")
        # print("Resizing images to", self.smaller_edge_size)
        self.transform = transforms.Compose([
            transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
            ])
        
        return model.to(self.device)
    
    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor = self.transform(img)
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size
    
    def extract_features(self, image_tensor, feature_list=None, cls=False):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)
            if cls == False:
                if feature_list == None:
                    tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
                    return tokens.cpu().numpy()
                else:
                    tokens = self.model.get_intermediate_layers(image_batch, n=feature_list)
                    tokens = list(tokens)
                    return tokens
            else:
                if feature_list == None:
                    outputs = self.model.get_intermediate_layers(image_batch, return_class_token=True)[0]
                    tokens, class_token = outputs[0], outputs[1]
                    return tokens.cpu().squeeze().numpy(), class_token.squeeze().cpu().numpy()
                else:
                    outputs = self.model.get_intermediate_layers(image_batch, n=feature_list, return_class_token=True)
                    tokens = []
                    class_token = []
                    for output in outputs:
                        tokens.append(output[0])
                        class_token.append(output[1].squeeze().cpu().numpy())
                    tokens = list(tokens)
                    class_token = class_token[-1]
                    return tokens, class_token

                
    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens
        
    def MLMP(self, features_list, grid_size, scales=[1, 5]):

        H, W = grid_size
        fused_features = []

        for feat in features_list:
            B, N, C = feat.shape
            assert N == H * W, f"Mismatch: N={N}, but H×W={H}×{W}={H*W}"
            feat = feat.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

            multi_scale_feats = [feat]
            for s in scales:
                pooled = F.adaptive_avg_pool2d(feat, (s, s))
                upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
                multi_scale_feats.append(upsampled)
            
            fused = sum(multi_scale_feats)
            fused_features.append(fused)

        final_feature = sum(fused_features) / len(fused_features)  
        return final_feature.view(B, C, -1).permute(0, 2, 1).squeeze().cpu().numpy()
      
    def compute_background_mask_from_image(self, image, threshold = 10, masking_type = None):
        image_tensor, grid_size = self.prepare_image(image)
        tokens = self.extract_features(image_tensor)
        return self.compute_background_mask(tokens, grid_size, threshold, masking_type)
    
    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False, kernel_size = 3, border = 0.2):
        # Kernel size for morphological operations should be odd
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))
        if masking_type == True:
            mask = first_pc > threshold
            m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
            if m.sum() <=  m.size * 0.35:
                mask = - first_pc > threshold
            # postprocess mask, fill small holes in the mask, enlarge slightly
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        elif masking_type == False:
            mask = np.ones_like(first_pc, dtype=bool)
        return mask.squeeze()
    def Kmeans_forge(self, img_features, grid_size, masking_type=True, kernel_size=3, border=0.2):

        if masking_type is False:
            return np.ones((img_features.shape[0],), dtype=bool)

        H, W = grid_size
        features_np = img_features.astype(np.float32)
        

        kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42).fit(features_np)
        labels_flat = kmeans.labels_  # shape: (H*W,)
        labels = labels_flat.reshape(H, W)
        
        
        
        crop_labels = labels[int(H*border):int(H*(1-border)), int(W*border):int(W*(1-border))]
        
        fg_label = -1
        
        if crop_labels.size > 0:
            count_0 = np.sum(crop_labels == 0)
            count_1 = np.sum(crop_labels == 1)
            
        
            fg_label = 0 if count_0 > count_1 else 1
            
        else:

            count_0_global = np.sum(labels_flat == 0)
            count_1_global = np.sum(labels_flat == 1)
            fg_label = 0 if count_0_global > count_1_global else 1

        mask_kmeans = (labels == fg_label).astype(np.uint8)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_kmeans = cv2.morphologyEx(mask_kmeans, cv2.MORPH_CLOSE, kernel)

        return mask_kmeans.astype(bool).flatten()

def get_model(model_name, device, smaller_edge_size=672):
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")


    if model_name.startswith("dinov2"):
        return DINOv2Wrapper(model_name, device, smaller_edge_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")