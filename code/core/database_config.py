"""
Database Config Managerment, which support both Qdrant 
vector database and Neo4j graph database configuration.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables early so DB configs pick them up


class QdrantConfig(BaseModel):
    """Qdrant Vector Database Configuration"""
    
    # connection config
    url: Optional[str] = Field(
        default=None,
        description="Qdrant服务URL (云服务或自定义URL)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API密钥 (云服务需要)"
    )
    
    # set config
    collection_name: str = Field(
        default="hello_agents_vectors",
        description="vector collection name"
    )
    vector_size: int = Field(
        default=384,
        description="vector dimension"
    )
    distance: str = Field(
        default="cosine",
        description="distance metric"
    )
    
    # connection pool config
    timeout: int = Field(
        default=30,
        description="connection timeout"
    )
    
    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """create config from environment variables"""
        return cls(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "384")),
            distance=os.getenv("QDRANT_DISTANCE", "cosine"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", "30"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class Neo4jConfig(BaseModel):
    """Neo4j Graph Database Configuration"""
    
    # connection config
    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j连接URI"
    )
    username: str = Field(
        default="neo4j",
        description="用户名"
    )
    password: str = Field(
        default="hello-agents-password",
        description="密码"
    )
    database: str = Field(
        default="neo4j",
        description="数据库名称"
    )
    
    # connection pool config
    max_connection_lifetime: int = Field(
        default=3600,
        description="最大连接生命周期(秒)"
    )
    max_connection_pool_size: int = Field(
        default=50,
        description="最大连接池大小"
    )
    connection_acquisition_timeout: int = Field(
        default=60,
        description="连接获取超时(秒)"
    )
    
    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """从环境变量创建配置"""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "hello-agents-password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            max_connection_lifetime=int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "3600")),
            max_connection_pool_size=int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")),
            connection_acquisition_timeout=int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "60"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class DatabaseConfig(BaseModel):
    """Database Config Manager"""
    
    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig,
        description="Qdrant vector database config"
    )
    neo4j: Neo4jConfig = Field(
        default_factory=Neo4jConfig,
        description="Neo4j graph database config"
    )
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """create config from environment variables"""
        return cls(
            qdrant=QdrantConfig.from_env(),
            neo4j=Neo4jConfig.from_env()
        )
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """get Qdrant config dict"""
        return self.qdrant.to_dict()
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """get Neo4j config dict"""
        return self.neo4j.to_dict()
    
    def validate_connections(self) -> Dict[str, bool]:
        """validate database connections and return results"""
        results = {}
        
        # validate Qdrant config
        try:
            from ..memory.storage.qdrant_store import QdrantVectorStore
            qdrant_store = QdrantVectorStore(**self.get_qdrant_config())
            results["qdrant"] = qdrant_store.health_check()
            logger.info(f"✅ Qdrant connection validation: {'Success' if results['qdrant'] else 'Failed'}")
        except Exception as e:
            results["qdrant"] = False
            logger.error(f"❌ Qdrant connection validation failed: {e}")
        
        # validate Neo4j config
        try:
            from ..memory.storage.neo4j_store import Neo4jGraphStore
            neo4j_store = Neo4jGraphStore(**self.get_neo4j_config())
            results["neo4j"] = neo4j_store.health_check()
            logger.info(f"✅ Neo4j connection validation: {'Success' if results['neo4j'] else 'Failed'}")
        except Exception as e:
            results["neo4j"] = False
            logger.error(f"❌ Neo4j connection validation failed: {e}")
        
        return results


# global database config instance
db_config = DatabaseConfig.from_env()


def get_database_config() -> DatabaseConfig:
    """get database config instance"""
    return db_config


def update_database_config(**kwargs) -> None:
    """update database config"""
    global db_config
    
    if "qdrant" in kwargs:
        db_config.qdrant = QdrantConfig(**kwargs["qdrant"])
    
    if "neo4j" in kwargs:
        db_config.neo4j = Neo4jConfig(**kwargs["neo4j"])
    
    logger.info("✅ database config updated successfully")
