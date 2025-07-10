from pymongo import MongoClient
import json
import os
from datetime import datetime

# Connect to MongoDB
client = MongoClient('mongodb+srv://hoquocthienanh:11102003@cluster0.ezt6v0r.mongodb.net/')
db = client['plant_database']  # Your database name

# Create collections
plants_collection = db['plants']
relationships_collection = db['relationships']
images_collection = db['images']

# Create indexes for faster queries
plants_collection.create_index('scientificName', unique=True)
plants_collection.create_index('vietnameseName')
relationships_collection.create_index([
    ('plant_from_scientific', 1),
    ('plant_to_scientific', 1)
])
images_collection.create_index('scientific_name', unique=True)

# Migration function for plant metadata
def migrate_plants(metadata_path):
    print(f"Starting migration of plants from {metadata_path}")
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Transform data if needed
        plants_data = []
        for scientific_name, plant_info in metadata.items():
            plant_doc = plant_info.copy()
            plant_doc['scientificName'] = scientific_name
            
            # Extract Vietnamese name for easier querying
            vn_name = plant_info.get("Tên tiếng Việt", "")
            if ";" in vn_name:
                vn_name = vn_name.split(";")[0].strip()
            plant_doc['vietnameseName'] = vn_name
            
            # Add migration timestamp
            plant_doc['created_at'] = datetime.now()
            plant_doc['updated_at'] = datetime.now()
            
            plants_data.append(plant_doc)
        
        # Insert into MongoDB
        if plants_data:
            # Use bulk operations for better performance
            result = plants_collection.insert_many(plants_data)
            print(f"Migrated {len(result.inserted_ids)} plants successfully")
        else:
            print("No plant data found to migrate")
            
    except Exception as e:
        print(f"Error migrating plants: {str(e)}")

# Migration function for relationships
def migrate_relationships(relationships_path):
    print(f"Starting migration of relationships from {relationships_path}")
    try:
        with open(relationships_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        relationships = data.get('relationships', [])
        
        # Transform data if needed - adding timestamps
        for rel in relationships:
            rel['created_at'] = datetime.now()
            rel['updated_at'] = datetime.now()
        
        # Insert into MongoDB
        if relationships:
            result = relationships_collection.insert_many(relationships)
            print(f"Migrated {len(result.inserted_ids)} relationships successfully")
        else:
            print("No relationship data found to migrate")
            
    except Exception as e:
        print(f"Error migrating relationships: {str(e)}")

# Migration function for images
def migrate_images(images_db_path):
    print(f"Starting migration of images from {images_db_path}")
    try:
        with open(images_db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        plants_images = data.get('plants', {})
        
        # Transform and insert each plant's images
        image_docs = []
        for scientific_name, plant_data in plants_images.items():
            # Format images for database
            # Use original structure but fix paths for web access
            image_list = []
            for img in plant_data.get('images', []):
                # Create web-accessible path from filesystem path
                web_path = f"/plant-images/{scientific_name}/{img['filename']}"
                
                image_list.append({
                    "filename": img['filename'],
                    "path": web_path,  # Web accessible path
                    "original_path": img['path'],  # Keep original for reference
                    "is_primary": img.get('is_primary', False),
                    "order": img.get('order', 0)
                })
            
            # Create complete document
            image_doc = {
                "scientific_name": scientific_name,
                "directory_name": plant_data.get('directory_name', scientific_name),
                "total_images": plant_data.get('total_images', len(image_list)),
                "images": image_list,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            image_docs.append(image_doc)
        
        # Insert into MongoDB
        if image_docs:
            result = images_collection.insert_many(image_docs)
            print(f"Migrated images for {len(result.inserted_ids)} plants successfully")
        else:
            print("No image data found to migrate")
            
    except Exception as e:
        print(f"Error migrating images: {str(e)}")

# Function to run all migrations
def run_all_migrations(metadata_path, relationships_path, images_path):
    print("Starting database migration...")
    
    # Clear existing collections (optional, be careful in production!)
    if input("Clear existing collections before migration? (y/n): ").lower() == 'y':
        plants_collection.delete_many({})
        relationships_collection.delete_many({})
        images_collection.delete_many({})
        print("Collections cleared")
    
    # Run migrations
    migrate_plants(metadata_path)
    migrate_relationships(relationships_path)
    migrate_images(images_path)
    
    # Verification
    plants_count = plants_collection.count_documents({})
    relationships_count = relationships_collection.count_documents({})
    images_count = images_collection.count_documents({})
    
    print("\nMigration Summary:")
    print(f"- Plants: {plants_count} documents")
    print(f"- Relationships: {relationships_count} documents")
    print(f"- Images: {images_count} documents")
    print("\nMigration completed!")

# Usage example
if __name__ == "__main__":
    # Paths to your JSON files
    METADATA_PATH = "/home/sora/pretrain_llm/infer/data/plant_data.json"
    RELATIONSHIPS_PATH = "/home/sora/pretrain_llm/infer/data/plant_relationships.json"
    IMAGES_PATH = "/home/sora/pretrain_llm/infer/data/plant_images_db.json"
    
    # Run migrations
    run_all_migrations(METADATA_PATH, RELATIONSHIPS_PATH, IMAGES_PATH)