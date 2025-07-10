
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig, AutoModel
from timm import create_model
class EnhancedProxyNCALoss(nn.Module):
    """
    Enhanced Proxy-NCA loss with focal loss and hard negative mining
    for better handling of imbalanced datasets
    """
    def __init__(self, num_classes, embedding_dim, scale=10.0, 
                 focal_alpha=0.25, focal_gamma=2.0, hard_negative_ratio=0.3):
        super(EnhancedProxyNCALoss, self).__init__()
        
        # Initialize one proxy per class
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        self.scale = scale
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.hard_negative_ratio = hard_negative_ratio
        self.num_classes = num_classes
        
        # Moving average for proxy updates (helps with stability)
        self.momentum = 0.99
        self.register_buffer('proxy_momentum', torch.zeros_like(self.proxies))
        
    def forward(self, embeddings, labels, class_weights=None):
        batch_size = embeddings.size(0)
        
        # Normalize embeddings and proxies
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        proxies_norm = F.normalize(self.proxies, p=2, dim=1)
        
        # Compute cosine similarities
        similarities = torch.mm(embeddings_norm, proxies_norm.t()) * self.scale
        
        # Default weights if none provided
        if class_weights is None:
            class_weights = torch.ones(self.num_classes, device=embeddings.device)
        
        total_loss = 0.0
        
        for i, label in enumerate(labels):
            label_idx = label.item()
            
            # Get similarities for current sample
            sample_similarities = similarities[i]  # Shape: [num_classes]
            
            # Positive similarity (to correct proxy)
            pos_sim = sample_similarities[label_idx]
            
            # --- Hard Negative Mining ---
            # Get similarities to all negative proxies
            neg_mask = torch.ones(self.num_classes, dtype=torch.bool, device=embeddings.device)
            neg_mask[label_idx] = False
            neg_similarities = sample_similarities[neg_mask]
            
            # Select hard negatives (top-k most similar negatives)
            num_hard_negatives = max(1, int(len(neg_similarities) * self.hard_negative_ratio))
            hard_neg_similarities, _ = torch.topk(neg_similarities, num_hard_negatives)
            
            # Combine positive with hard negatives for loss computation
            combined_similarities = torch.cat([pos_sim.unsqueeze(0), hard_neg_similarities])
            
            # Compute softmax probabilities
            probs = F.softmax(combined_similarities, dim=0)
            pos_prob = probs[0]  # Probability of correct class
            
            # --- Focal Loss Component ---
            # Standard cross-entropy loss
            ce_loss = -torch.log(pos_prob + 1e-8)
            
            # Focal loss weighting
            focal_weight = self.focal_alpha * (1 - pos_prob) ** self.focal_gamma
            focal_loss = focal_weight * ce_loss
            
            # Apply class weight
            weighted_loss = focal_loss * class_weights[label_idx]
            
            total_loss += weighted_loss
        
        # Update proxy momentum (for stability)
        with torch.no_grad():
            # Compute gradients w.r.t. proxies for momentum update
            proxy_gradients = torch.zeros_like(self.proxies)
            for i, label in enumerate(labels):
                # Simple gradient approximation for momentum
                embedding_norm = embeddings_norm[i]
                proxy_norm = proxies_norm[label.item()]
                grad_approx = embedding_norm - proxy_norm
                proxy_gradients[label.item()] += grad_approx
            
            # Update momentum
            self.proxy_momentum = self.momentum * self.proxy_momentum + (1 - self.momentum) * proxy_gradients
        
        return total_loss / batch_size
class ProxyNCALoss(nn.Module):
    """
    Proxy-NCA loss for efficient metric learning on large, imbalanced datasets
    """
    def __init__(self, num_classes, embedding_dim, scale=10.0):
        super(ProxyNCALoss, self).__init__()
        # Initialize one proxy per class
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        # Initialize with Kaiming normalization
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.scale = scale  # Temperature scaling factor
        
    def forward(self, embeddings, labels, class_weights=None):
        # Normalize proxies to unit length
        proxies = F.normalize(self.proxies, p=2, dim=1)
        
        # Default weights if none provided
        if class_weights is None:
            class_weights = torch.ones(proxies.shape[0], device=embeddings.device)
        
        # Calculate distance to all proxies
        dist = torch.cdist(embeddings, proxies)
        
        # Turn distances into similarities with temperature scaling
        # Negating distances since we want lower distances to have higher probs
        similarity = -dist * self.scale
        
        # For each embedding, compute probability of it being assigned to the right proxy
        loss = 0
        for i, label in enumerate(labels):
            # Cross-entropy-like loss
            pos_sim = similarity[i, label]
            exp_pos_sim = torch.exp(pos_sim)
            
            # Sum of all similarities
            exp_all_sim = torch.exp(similarity[i]).sum()
            
            # Negative log probability with class weight
            sample_loss = -torch.log(exp_pos_sim / exp_all_sim)
            weighted_loss = sample_loss * class_weights[label]
            
            loss += weighted_loss
            
        return loss / len(embeddings)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class DualStreamFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualStreamFusionBlock, self).__init__()

        # Initial 1x1 convolutions after concatenation
        self.initial_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1_3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1_4 = nn.Conv2d(3*in_channels, in_channels, kernel_size=1)

        
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Element-wise multiplication and final channel attention
        self.channel_attention = ChannelAttention(in_channels)
        self.final_conv = nn.Conv2d(3 * in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(768, eps=1e-6)
    def forward(self, F_C, F_V):
        # Concatenate inputs
        x = torch.cat((F_C, F_V), dim=1)

        # Initial convolution
        x = self.initial_conv(x)

        x_1 = self.conv1_1(x)
        x_1_1 = self.conv1_2(x_1)
        x_1_1 = F.sigmoid(x_1_1)
        
        x_2 = self.conv3_1(x)
        x_2 = F.relu(x_2)
        x_2_1 = self.conv1_3(x_2)
        x_2_1 = F.sigmoid(x_2_1)
        
        x_3 = self.conv3_2(x)
        x_3 = F.relu(x_3)
        x_3 = self.conv3_3(x_3)

        x_2 = x_1_1 * x_2
        x_3 = x_3 * x_2_1
        # Channel attention
        combined = torch.cat((x_1, x_2, x_3), dim=1)
        combined = self.conv1_4(combined)
        attention = self.channel_attention(combined)
        top = x_1 * attention
        middle = x_2 * attention
        bottom = x_3 * attention
        final = torch.cat((top, middle, bottom), dim=1)

        # Final 1x1 convolution
        F_fusion = self.final_conv(final)
        return self.norm(F_fusion.mean([-2, -1]))
class ImageEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=512, l2_normalize=True):
        super(ImageEmbeddingModel, self).__init__()
        # Load pretrained ConvNeXt V2
        self.convnext = create_model('convnextv2_tiny', pretrained=True, features_only=True)

        # Load pretrained Vision Transformer
        self.vit = AutoModel.from_pretrained('facebook/dinov2-base')

        # Fusion module
        self.fusion = DualStreamFusionBlock(768, 768)

        # Embedding projection - replace classifier with embedding projector
        self.embedding_projector = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Linear(768, embedding_dim)
        )
        
        self.l2_normalize = l2_normalize

    def forward(self, x):
        # ConvNeXt feature extraction
        conv_features = self.convnext(x)[-1]  # Shape: (B, C, H, W)

        # Vision Transformer feature extraction
        vit_features = self.vit(pixel_values=x).last_hidden_state  # Shape: (B, N, D)
        B, N, D = vit_features.shape

        # Remove the [CLS] token and reshape
        vit_features = vit_features[:, 1:, :]  # Shape: (B, N-1, D)
        H, W = conv_features.shape[2], conv_features.shape[3]  # Match ConvNeXt spatial size

        # Reshape ViT output to match ConvNeXt spatial dimensions
        vit_features = vit_features.permute(0, 2, 1)  # Shape: (B, D, N-1)
        vit_features = vit_features.reshape(B, D, int((N-1)**0.5), int((N-1)**0.5))  # Shape: (B, D, H', W')

        # Resize ViT features to match ConvNeXt spatial dimensions
        vit_features = nn.functional.interpolate(vit_features, size=(H, W), mode='bilinear', align_corners=False)

        # Fusion module
        fused_features = self.fusion(conv_features, vit_features)

        # Project to embedding space
        embeddings = self.embedding_projector(fused_features)
        
        # L2 normalize if specified (important for cosine similarity retrieval)
        if self.l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings
    
