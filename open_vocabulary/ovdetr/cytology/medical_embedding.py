"""
Medical Dataset Class
- Integrates medical descriptions
- Handles CLIP text/image embeddings
- Combines with proposals
"""

import json
import pickle
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, split_name):
        # Load metadata
        with open(f"dataset/splits/{split_name}.json") as f:
            self.data = json.load(f)
        
        # Load medical descriptions
        with open("medical_embeddings.json") as f:
            self.medical_emb = json.load(f)
        
        # Load proposals
        with open(f"dataset/proposals/{split_name}_proposals.pkl", 'rb') as f:
            self.proposals = pickle.load(f)
        
        # CLIP initialization
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def __len__(self):
        return len(self.data['images'])
    
    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        
        # Load image
        image = Image.open(f"dataset/images/{img_info['file_name']}")
        
        # Get proposals
        img_proposals = self.proposals[img_info['id']]
        
        # Get text embeddings for all classes
        text_descriptions = [
            self.medical_emb[c['name']] 
            for c in self.data['categories']
        ]
        text_inputs = self.processor(
            text=text_descriptions,
            return_tensors="pt",
            padding=True
        )
        
        # Get image embeddings for proposals
        proposal_embeddings = []
        for box in img_proposals['boxes']:
            x1, y1, x2, y2 = box
            crop = image.crop((x1, y1, x2, y2))
            image_input = self.processor(
                images=crop,
                return_tensors="pt"
            )
            embeddings = self.clip_model.get_image_features(**image_input)
            proposal_embeddings.append(embeddings)
        
        return {
            'image': image,
            'text_embeddings': text_inputs,
            'proposal_embeddings': torch.stack(proposal_embeddings),
            'proposals': img_proposals
        }

if __name__ == "__main__":
    # Test dataset
    dataset = MedicalDataset("base_train")
    sample = dataset[0]
    print("Text embeddings shape:", sample['text_embeddings']['input_ids'].shape)
    print("Proposal embeddings shape:", sample['proposal_embeddings'].shape)