# db_service.py
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import os
import logging
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDBService:
    """
    Singleton service for interacting with the MongoDB database.
    Provides methods to access plant data, relationships, and images.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PlantDBService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, connection_string=None, db_name=None):
        # Only initialize once
        if self._initialized:
            return
            
        # Get connection details from env vars or parameters
        self.connection_string = connection_string or os.environ.get("MONGODB_URI", "mongodb+srv://hoquocthienanh:11102003@cluster0.ezt6v0r.mongodb.net/")
        self.db_name = db_name or os.environ.get("MONGODB_DB", "plant_database")
        
        # Initialize without connecting
        self.client = None
        self.db = None
        self._connected = False
        self._initialized = True
        
        # Collections
        self.plants_collection = None
        self.relationships_collection = None
        self.images_collection = None
        
        logger.info(f"PlantDBService initialized with database: {self.db_name}")

    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        Returns True if successful, False otherwise.
        """
        if self._connected:
            return True
            
        try:
            logger.info("Connecting to MongoDB...")
            self.client = MongoClient(self.connection_string)
            
            # Test connection
            self.client.admin.command('ping')
            
            # Set database and collections
            self.db = self.client[self.db_name]
            self.plants_collection = self.db.plants
            self.relationships_collection = self.db.relationships
            self.images_collection = self.db.images
            
            self._connected = True
            logger.info("Successfully connected to MongoDB")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error when connecting to MongoDB: {str(e)}")
            return False

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("MongoDB connection closed")

    # Plant Metadata Methods
    def get_all_plants(self) -> Dict[str, Any]:
        """
        Get all plant metadata, formatted as in the original JSON.
        Returns dict with scientific names as keys.
        """
        if not self._ensure_connection():
            return {}
            
        try:
            plants = {}
            cursor = self.plants_collection.find({}, {'_id': 0})
            
            for plant in cursor:
                scientific_name = plant.pop('scientificName', None)
                if scientific_name:
                    # Remove created_at/updated_at fields which don't exist in original JSON
                    plant.pop('created_at', None)
                    plant.pop('updated_at', None)
                    plants[scientific_name] = plant
                    
            return plants
            
        except Exception as e:
            logger.error(f"Error fetching plants: {str(e)}")
            return {}

    def get_plant_by_scientific_name(self, scientific_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific plant by its scientific name"""
        if not self._ensure_connection():
            return None
        print(self.plants_collection)
        try:
            return self.plants_collection.find_one(
                {'scientificName': scientific_name}, 
                {'_id': 0}
            )
        except Exception as e:
            logger.error(f"Error fetching plant {scientific_name}: {str(e)}")
            return None

    # Relationship Methods
    def get_plant_relationships(self, plant: Optional[str] = None, condition: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get plant relationships, optionally filtered by plant name or condition.
        Mimics the behavior of the original API endpoint.
        """
        if not self._ensure_connection():
            return []
            
        try:
            query = {}
            
            if plant:
                # Search by either scientific or Vietnamese name (case-insensitive)
                plant_lower = plant.lower()
                query['$or'] = [
                    {'plant_from_scientific': {'$regex': plant_lower, '$options': 'i'}},
                    {'plant_from_vietnamese': {'$regex': plant_lower, '$options': 'i'}},
                    {'plant_to_scientific': {'$regex': plant_lower, '$options': 'i'}},
                    {'plant_to_vietnamese': {'$regex': plant_lower, '$options': 'i'}}
                ]
            
            if condition:
                # Filter by condition (case-insensitive)
                query['conditions'] = {'$regex': condition.lower(), '$options': 'i'}
            
            # Exclude _id, created_at, and updated_at fields
            projection = {'_id': 0, 'created_at': 0, 'updated_at': 0}
            
            return list(self.relationships_collection.find(query, projection))
            
        except Exception as e:
            logger.error(f"Error fetching relationships: {str(e)}")
            return []

    # Images Methods
    def get_plant_images(self, scientific_name: str) -> Optional[Dict[str, Any]]:
        """Get images for a specific plant"""
        if not self._ensure_connection():
            return None
            
        try:
            result = self.images_collection.find_one(
                {'scientific_name': scientific_name},
                {'_id': 0, 'created_at': 0, 'updated_at': 0}
            )
            
            if result:
                # Format response to match original API
                return {
                    'plant': scientific_name,
                    'total_images': result.get('total_images', 0),
                    'images': result.get('images', [])
                }
            else:
                return {
                    'plant': scientific_name,
                    'images': [],
                    'error': "Plant not found in image database"
                }
                
        except Exception as e:
            logger.error(f"Error fetching images for {scientific_name}: {str(e)}")
            return {
                'plant': scientific_name,
                'images': [],
                'error': str(e)
            }
    
    def _ensure_connection(self) -> bool:
        """Ensure MongoDB connection is established"""
        if not self._connected:
            return self.connect()
        return True
    