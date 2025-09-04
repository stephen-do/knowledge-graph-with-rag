import os
from rdflib import Graph
from rdflib_neo4j import Neo4jStore, Neo4jStoreConfig, HANDLE_VOCAB_URI_STRATEGY
from dotenv import load_dotenv
from neo4j import GraphDatabase   # cần cài neo4j-driver

load_dotenv()

# Config Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "adgjmptw1")

NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ---- Neo4j Desktop local ----
auth = {
    "uri": NEO4J_URI,
    "database": NEO4J_DATABASE,
    "user": NEO4J_USER,
    "pwd":  NEO4J_PASSWORD,
}

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
with driver.session(database=NEO4J_DATABASE) as session:
    session.run("MATCH (n) DETACH DELETE n")
driver.close()
print("Đã xóa toàn bộ dữ liệu cũ trong Neo4j.")

# Import dữ liệu RDF
config = Neo4jStoreConfig(
    auth_data=auth,
    handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,
    batching=False,
)

store = Neo4jStore(config=config)
g = Graph(store=store)

def file_url(p: str) -> str:
    return "file:///" + os.path.abspath(p).replace("\\", "/")

# import data
g.parse(file_url("data/rdf/healthcare_data.ttl"), format="turtle")
# g.parse(file_url("data/rdf/healthcare_ontology.ttl"), format="turtle")

print("Triples in store:", len(g))

# Index KG
with driver.session(database=NEO4J_DATABASE) as session:
    result = session.run("SHOW INDEXES YIELD name RETURN name")
    for record in result:
        idx_name = record["name"]
        if idx_name == "n10s_unique_uri":
            continue
        session.run(f"DROP INDEX {idx_name} IF EXISTS")
        print(f"Dropped index: {idx_name}")

    # Tạo B-Tree index
    session.run("""
    CREATE INDEX hp_name_idx IF NOT EXISTS
    FOR (hp:HealthcareProvider) ON (hp.name)
    """)

    session.run("""
    CREATE INDEX p_name_idx IF NOT EXISTS
    FOR (p:Patient) ON (p.name)
    """)

    session.run("""
    CREATE INDEX spec_name_idx IF NOT EXISTS
    FOR (s:Specialization) ON (s.name)
    """)

    session.run("""
    CREATE INDEX loc_name_idx IF NOT EXISTS
    FOR (l:Location) ON (l.name)
    """)

    # Tạo Full-text index
    session.run("""
    CREATE FULLTEXT INDEX ft_hp IF NOT EXISTS
    FOR (hp:HealthcareProvider) ON EACH [hp.name]
    """)

    session.run("""
    CREATE FULLTEXT INDEX ft_patient IF NOT EXISTS
    FOR (p:Patient) ON EACH [p.name]
    """)

    session.run("""
    CREATE FULLTEXT INDEX ft_spec IF NOT EXISTS
    FOR (s:Specialization) ON EACH [s.name]
    """)

    session.run("""
    CREATE FULLTEXT INDEX ft_loc IF NOT EXISTS
    FOR (l:Location) ON EACH [l.name]
    """)

    # Đợi index sẵn sàng
    session.run("CALL db.awaitIndexes()")
driver.close()

print("Create index on name successfully!")
