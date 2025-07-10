import os
import torch
import numpy as np
import faiss
from PIL import Image
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, img_path, self.labels[idx], True
        except Exception as e:
            # Return dummy data for failed images
            dummy_tensor = torch.zeros(3, 224, 224)  # Adjust size as needed
            return dummy_tensor, self.image_paths[idx], self.labels[idx], False
        
        
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.convnext = model.convnext
        self.vit = model.vit
        self.fusion = model.fusion
        self.projector = model.embedding_projector
        
    def forward(self, x):
        # ConvNeXt feature extraction
        conv_features = self.convnext(x)[-1]
        
        # Vision Transformer feature extraction
        vit_features = self.vit(pixel_values=x).last_hidden_state
        B, N, D = vit_features.shape
        
        # Remove the [CLS] token and reshape
        vit_features = vit_features[:, 1:, :]
        H, W = conv_features.shape[2], conv_features.shape[3]
        
        # Reshape ViT output to match ConvNeXt spatial dimensions
        vit_features = vit_features.permute(0, 2, 1)
        vit_features = vit_features.reshape(B, D, int((N-1)**0.5), int((N-1)**0.5))
        
        # Resize ViT features to match ConvNeXt spatial dimensions
        vit_features = F.interpolate(vit_features, size=(H, W), mode='bilinear', align_corners=False)
        
        # Extract fused features
        fused_features = self.fusion(conv_features, vit_features)
        fused_features = self.projector(fused_features)
        return fused_features
    
class ViTFeatureExtractor(torch.nn.Module):
    """
    Feature extractor for pure ViT embedding model
    """
    def __init__(self, model):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = model.vit
        self.projector = model.embedding_projector
        
    def forward(self, x):
        # Vision Transformer feature extraction
        vit_outputs = self.vit(pixel_values=x)
        
        # Extract the [CLS] token representation
        cls_token = vit_outputs.last_hidden_state[:, 0, :]
        
        # Project to embedding space
        embeddings = self.projector(cls_token)
        
        return embeddings
class ConvNeXtV2FeatureExtractor(torch.nn.Module):
    """
    Feature extractor for pure ConvNeXtV2 embedding model
    """
    def __init__(self, model):
        super(ConvNeXtV2FeatureExtractor, self).__init__()
        self.convnext = model.convnext
        self.projector = model.embedding_projector
        
    def forward(self, x):
        # ConvNeXtV2 feature extraction
        # The model with num_classes=0 returns features directly
        features = self.convnext(x)  # Shape: (B, feature_dim)
        
        # Project to embedding space
        embeddings = self.projector(features)
        
        return embeddings
    
class ImageRetrievalSystem:
    def __init__(self, model_path, image_folder, index_file="plant_index_2.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load trained model
        self.model = self.load_model(model_path)
        self.feature_extractor = FeatureExtractor(self.model).to(self.device)
        self.feature_extractor.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize FAISS index and image database
        self.image_folder = image_folder
        self.index_file = index_file
        self.image_paths = []
        self.image_labels = []
        self.index = None
        
        # Build or load the image database and FAISS index
        if os.path.exists(self.index_file):
            self.load_index()
        else:
            self.build_index_optimized()
    
    def load_model(self, model_path):
        # Determine the number of classes from the model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        # This assumes ImageEmbeddingModel is imported from elsewhere
        from embedder import ImageEmbeddingModel
        model = ImageEmbeddingModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def extract_features(self, img):
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()
    
    def build_index(self):
        print("Building FAISS index...")
        features_list = []
        for root, _, files in os.walk(self.image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    label = os.path.basename(root)
                    self.image_paths.append(full_path)
                    self.image_labels.append(label)
        
        for img_path in tqdm(self.image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                features = self.extract_features(img)
                features_list.append(features.flatten())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                idx = self.image_paths.index(img_path)
                self.image_paths.pop(idx)
                self.image_labels.pop(idx)
        
        if features_list:
            all_features = np.vstack(features_list)
            d = all_features.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(all_features)
            self.save_index()
            print(f"FAISS index built with {len(self.image_paths)} images.")
        else:
            print("No valid images found.")
    def build_index_optimized(self, batch_size=64, num_workers=8, prefetch_factor=4):
        """
        Optimized version with proper CPU-GPU pipeline overlap
        """
        print("Building FAISS index with optimized batch processing...")
        
        # Gather all image paths and labels
        all_image_paths = []
        all_image_labels = []
        for root, _, files in os.walk(self.image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    label = os.path.basename(root)
                    all_image_paths.append(full_path)
                    all_image_labels.append(label)
        
        print(f"Found {len(all_image_paths)} images to process")
        
        # Create dataset and dataloader with multiprocessing
        dataset = ImageDataset(all_image_paths, all_image_labels, self.transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            drop_last=False
        )
        
        features_list = []
        valid_image_paths = []
        valid_image_labels = []
        
        # Pre-allocate CUDA stream for async operations
        stream = torch.cuda.Stream()
        
        with torch.cuda.stream(stream):
            for batch_data in tqdm(dataloader, desc="Extracting features"):
                batch_tensors, batch_paths, batch_labels, valid_flags = batch_data
                
                # Filter out invalid images
                valid_mask = valid_flags.bool()
                if not valid_mask.any():
                    continue
                    
                valid_tensors = batch_tensors[valid_mask]
                valid_paths = [path for i, path in enumerate(batch_paths) if valid_flags[i]]
                valid_labels = [label for i, label in enumerate(batch_labels) if valid_flags[i]]
                
                # Move to GPU asynchronously
                batch_tensor = valid_tensors.to(self.device, non_blocking=True)
                
                # Extract features
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        features = self.feature_extractor(batch_tensor)
                        
                # Normalize and move to CPU
                features = F.normalize(features.float(), p=2, dim=1)
                features_cpu = features.cpu().numpy()
                
                # Store results
                for i, feature in enumerate(features_cpu):
                    features_list.append(feature.flatten())
                    valid_image_paths.append(valid_paths[i])
                    valid_image_labels.append(valid_labels[i])
        
        # Build FAISS index
        if features_list:
            all_features = np.vstack(features_list)
            d = all_features.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(all_features)
            self.image_paths = valid_image_paths
            self.image_labels = valid_image_labels
            self.save_index()
            print(f"FAISS index built with {len(self.image_paths)} images.")
        else:
            print("No valid images found for indexing.")
    def save_index(self):
        print(f"Saving index to {self.index_file}")
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'image_paths': self.image_paths,
                'image_labels': self.image_labels,
                'index_data': faiss.serialize_index(self.index)
            }, f)
    
    def load_index(self):
        print(f"Loading index from {self.index_file}")
        with open(self.index_file, 'rb') as f:
            data = pickle.load(f)
        self.image_paths = data['image_paths']
        self.image_labels = data['image_labels']
        self.index = faiss.deserialize_index(data['index_data'])
        print(f"FAISS index loaded with {len(self.image_paths)} images.")
    
    def retrieve_similar_images(self, query_img, top_k=5):
        query_features = self.extract_features(query_img)
        D, I = self.index.search(query_features, top_k)
        similar_paths = [self.image_paths[i] for i in I[0]]
        similar_labels = [self.image_labels[i] for i in I[0]]
        return similar_paths, similar_labels, D[0]

    def is_out_of_distribution_class_based(self, query_img, k=100, min_unique_classes=10, threshold=0.25):
        """
        Determine if an image is OOD using class-based distance averaging
        
        Args:
            query_img: PIL Image
            k: Initial number of neighbors to retrieve
            min_unique_classes: Minimum number of unique classes to find
            threshold: OOD threshold for minimum class average
            
        Returns:
            (is_ood, confidence, details)
        """
        # Extract features
        query_features = self.extract_features(query_img)
        
        # Start with k neighbors
        current_k = k
        max_k = min(1000, self.index.ntotal)  # Set a reasonable upper limit
        unique_classes = set()
        
        # Increase k until we find min_unique_classes or hit the maximum
        while len(unique_classes) < min_unique_classes and current_k <= max_k:
            D, I = self.index.search(query_features, current_k)
            
            # Extract classes from results
            for idx in I[0]:
                if idx < len(self.image_labels):  # Safety check
                    unique_classes.add(self.image_labels[idx])
            
            # If we still need more classes, double k and search again
            if len(unique_classes) < min_unique_classes:
                current_k = min(current_k * 2, max_k)
            else:
                break
        
        # Group distances by class
        class_distances = {}
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.image_labels):  # Safety check
                label = self.image_labels[idx]
                if label not in class_distances:
                    class_distances[label] = []
                class_distances[label].append(float(dist))
        
        # If no valid classes found, it's probably OOD
        if not class_distances:
            return True, 0.95, {
                "error": "No valid classes found in results",
                "k_used": current_k
            }
        
        # Calculate average distance for each class
        class_avg_distances = {}
        for label, distances in class_distances.items():
            class_avg_distances[label] = sum(distances) / len(distances)
        
        # Find the class with minimum average distance
        min_class = min(class_avg_distances.items(), key=lambda x: x[1])
        min_class_label, min_avg_distance = min_class
        
        # Determine if OOD based on minimum class average
        is_ood = min_avg_distance > threshold
        
        # Calculate confidence
        if is_ood:
            # Confidence increases as we move further from threshold
            excess = min_avg_distance - threshold
            confidence = min(0.95, 0.7 + 0.25 * min(excess, threshold) / threshold)
        else:
            # Confidence increases as we move closer to zero distance
            confidence = 0.7 + 0.25 * (threshold - min_avg_distance) / threshold
            confidence = min(0.95, confidence)
        
        # Gather detailed results for debugging and analysis
        details = {
            "min_class": min_class_label,
            "min_avg_distance": min_avg_distance,
            "threshold": threshold,
            "class_averages": {k: round(v, 4) for k, v in sorted(class_avg_distances.items(), key=lambda x: x[1])[:5]},
            "num_classes_found": len(class_distances),
            "total_results": current_k,
            "classes_required": min_unique_classes,
            "is_ood": is_ood
        }
        
        return is_ood, confidence, details
    
    def validate_images_class_based(self, image_paths, threshold=0.25, k=100, min_unique_classes=10):
        """
        Validate a batch of images using the class-based OOD detection
        
        Args:
            image_paths: List of paths to images
            threshold: OOD threshold
            k: Number of neighbors to retrieve
            min_unique_classes: Minimum number of unique classes to consider
            
        Returns:
            valid_paths, ood_results
        """
        valid_paths = []
        ood_results = []
        
        for path in image_paths:
            # try:
            img = Image.open(path).convert("RGB")
            is_ood, confidence, details = self.is_out_of_distribution_class_based(
                img, k=k, min_unique_classes=min_unique_classes, threshold=threshold
            )
            
            if not is_ood:
                valid_paths.append(path)
                print(f"Image {path} accepted with min avg distance {details['min_avg_distance']:.4f}")
            else:
                ood_results.append({
                    "path": path,
                    "confidence": confidence,
                    "details": details
                })
                print(f"Image {path} rejected with min avg distance {details['min_avg_distance']:.4f}")
            # except Exception as e:
            #     print(f"Error processing {path}: {e}")
            #     ood_results.append({
            #         "path": path,
            #         "error": str(e)
            #     })
        
        return valid_paths, ood_results
    
    def cleanup(self):
        print("Cleaning up resources...")
        del self.model
        del self.feature_extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("Cleanup complete.")