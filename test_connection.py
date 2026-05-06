import os
from dotenv import load_dotenv
load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "neo4j+s://fe6abf34.databases.neo4j.io",
    auth=("neo4j", NEO4J_PASSWORD)
)

with driver.session() as session:
    result = session.run("RETURN 1 AS test")
    print(result.single())