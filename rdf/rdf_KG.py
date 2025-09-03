# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from langchain_neo4j import Neo4jGraph

# load_dotenv()
# NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
# DB_NAME = os.getenv("NEO4J_DATABASE", "neo4j")

# ONTOLOGY_PATH = "data/rdf/healthcare_ontology.ttl"  # dùng "/" để tránh \r
# DATA_PATH     = "data/rdf/healthcare_data.ttl"

# graph = Neo4jGraph(
#     url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD,
#     database=DB_NAME, refresh_schema=False
# )

# def run(q, params=None): return graph.query(q, params or {})

# def read_text(path): return Path(path).read_text(encoding="utf-8")

# # (tuỳ chọn) xoá sạch để import lại
# run("MATCH (n) DETACH DELETE n")

# # 1) Constraint
# run("""
# CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS
# FOR (r:Resource) REQUIRE r.uri IS UNIQUE
# """)

# 2) n10s config cho đúng mapping
# run("""
# CALL n10s.graphconfig.init({
#   handleRDFTypes: "LABELS",
#   handleVocabUris: "SHORTEN",
#   applyNeo4jNaming: true,
#   maintainLanguageTag: true,
#   handleMultival: "ARRAY"
# })
# """)

# # 3) prefix (đẹp hơn khi xem)
# run("""CALL n10s.nsprefixes.add("hc","http://example.org/healthcare#")""")

# # 4) Import ontology trước
# # res1 = run("""
# # CALL n10s.rdf.import.inline($payload, "Turtle", {commitSize: 5000})
# # """, {"payload": read_text(ONTOLOGY_PATH)})
# # print("Ontology import:", res1)

# # 5) Import data sau
# res2 = run("""
# CALL n10s.rdf.import.inline($payload, "Turtle", {commitSize: 5000})
# """, {"payload": read_text(DATA_PATH)})
# print("Data import:", res2)

# # 6) Sanity check
# print("Labels:", run("""CALL db.labels() YIELD label RETURN label ORDER BY label"""))
# print("Rels:",   run("""CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"""))
# print("Props:",  run("""CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey"""))
# pip install rdflib rdflib-neo4j
