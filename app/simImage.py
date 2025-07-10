from PIL import Image
import torch
from torchvision import transforms
import json
import gc
from classifier import HybridClassifier
from torchvision import datasets
import os
import numpy as np
from collections import defaultdict

class SimpleClassificationPipeline:
    def __init__(self, classifier_path=None, label_path=None, token=None):
        print("Loading classifier...")
        self.classifier = HybridClassifier(token=token)
        if classifier_path:
            checkpoint = torch.load(classifier_path, map_location='cuda' if torch.cuda.is_available() else "cpu")
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        print("Loaded classifier")
        
        # Define the image transformation.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.labels_path = label_path
        # Load labels if provided.
        if self.labels_path:
            # self.labels = json.load(open(label_path, 'r'))
            self.labels = datasets.ImageFolder(root=label_path).classes
        else:
            self.labels = None
            
    def get_representative_image(self, label):
        """
        Returns a representative image path for a given label.
        Assumes that your `labels_path` directory contains subfolders named after labels.
        """
        label_folder = os.path.join(self.labels_path, label)
        if os.path.isdir(label_folder):
            for f in os.listdir(label_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    return os.path.join(label_folder, f)
        # Fallback: if no image is found, return None or a default image.
        return None

    def process_image(self, image_path):
        # Load the image and apply transformations.
        image = Image.open(image_path).convert("RGB")
        tensor_image = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor_image = tensor_image.to(device)
        self.classifier = self.classifier.to(device)
        

        
        # Run the classifier.
        with torch.no_grad():
            logits = self.classifier(tensor_image)
            probs = torch.softmax(logits, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
        
        # Get the predicted label.
        if self.labels and pred_idx.item() < len(self.labels):
            pred_label = self.labels[pred_idx.item()]
        else:
            pred_label = f"Class {pred_idx.item()}"
        
        print(f"Predicted label: {pred_label}, Confidence: {max_prob.item():.4f}")
        return pred_label, max_prob.item()
    
    def process_image_topk(self, image_path, top_k=5):
        # Load the image and apply transformations.
        image = Image.open(image_path).convert("RGB")
        tensor_image = self.transform(image).unsqueeze(0)  # Add batch dimension

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor_image = tensor_image.to(device)
        self.classifier = self.classifier.to(device)

        # Run the classifier.
        with torch.no_grad():
            logits = self.classifier(tensor_image)
            probs = torch.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probs, top_k)

        results = []
        for idx, prob in zip(topk_indices[0].tolist(), topk_probs[0].tolist()):
            # Get the predicted label.
            if self.labels and idx < len(self.labels):
                label = self.labels[idx]
            else:
                label = f"Class {idx}"
            # Get a representative image for this label.
            rep_img = self.get_representative_image(label)
            results.append((label, prob, rep_img))
        return results
    
    def process_multiple_images_topk(self, image_paths, top_k=5):
        """
        Process multiple images and return top-k predictions using ensemble voting.
        
        Args:
            image_paths (list): List of paths to image files
            top_k (int): Number of top predictions to return
            
        Returns:
            List of (label, confidence, image_path) tuples for the top-k classes
        """
        if not image_paths:
            return []
        
        # Process each image and collect predictions
        all_predictions = []
        print(f"Processing {len(image_paths)} images...")
        
        for path in image_paths:
            try:
                predictions = self.process_image_topk(path, top_k=10)  # Get more predictions per image for better aggregation
                all_predictions.append(predictions)
                print(f"Processed {path}, found {len(predictions)} predictions")
            except Exception as e:
                print(f"Error processing image {path}: {(e)}")
                # Continue with other images even if one fails
        
        # If no valid predictions, return empty list
        if not all_predictions:
            return []
        
        # Combine predictions using confidence-weighted voting
        return self.combine_predictions(all_predictions, top_k=top_k)
    
    def combine_predictions(self, image_predictions, top_k=5):
        """
        Combine predictions from multiple images using confidence-weighted voting.
        
        Args:
            image_predictions: List of lists, where each inner list contains 
                             (label, confidence, image_path) tuples for one image
            top_k: Number of top predictions to return
            
        Returns:
            List of (label, confidence, image_path) tuples for the top-k classes
        """
        print(f"Combining predictions from {len(image_predictions)} images...")
        
        # Aggregate votes by label
        label_scores = defaultdict(float)
        label_images = {}
        label_counts = defaultdict(int)
        
        # Step 1: Sum all confidence scores per label
        for image_results in image_predictions:
            for label, confidence, image_path in image_results:
                label_scores[label] += confidence
                label_counts[label] += 1
                
                # Keep the image path with the highest confidence for each label
                if label not in label_images or confidence > label_scores[label] / label_counts[label]:
                    label_images[label] = image_path
        
        # Step 2: Calculate weighted average confidence score
        for label in label_scores:
            label_scores[label] /= len(image_predictions)  # Normalize by number of images
        
        # Step 3: Sort by final confidence score and take top-k
        sorted_results = sorted(
            [(label, score, label_images[label]) for label, score in label_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        print(f"Final top {len(sorted_results)} predictions after ensemble:")
        for label, score, _ in sorted_results:
            print(f"  - {label}: {score:.4f}")
            
        return sorted_results

    def cleanup(self):
        print("Cleaning up resources...")
        del self.classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Cleanup complete.")