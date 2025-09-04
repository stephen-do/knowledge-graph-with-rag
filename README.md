# KG-RAG: Knowledge Graph-based Retrieval Augmented Generation

This repository contains a collection of implementations for Knowledge Graph-based RAG (Retrieval Augmented Generation) approaches and baseline methods for comparison. The code is structured as a Python package with modular components.

## Overview

The repository implements several RAG approaches:

**KG-RAG approaches**:
   - **Cypher-based approach**: Uses Cypher queries to retrieve information from a Neo4j graph database
## Installation

### Using uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/stephen-do/knowledge-graph-with-rag.git
cd kg-rag

# Install uv if you don't have it
curl -sSf https://astral.sh/uv/install.sh | bash

uv sync --dev
source .venv/bin/activate

#Import RDF into KG:
1. Install APOC plugin on Neo4j
    docker run --name neo4j \
      -p7474:7474 -p7687:7687 \
      -e NEO4J_AUTH=neo4j/adgjmptw1 \
      -e NEO4JLABS_PLUGINS='["apoc"]' \
      -e NEO4J_apoc_export_file_enabled=true \
      -e NEO4J_apoc_import_file_enabled=true \
      -e NEO4J_apoc_import_file_use__neo4j__config=true \
      -e NEO4J_apoc_meta_data_enabled=true \
      neo4j
2. Inst libs : rdflib + rdflib_neo4j  -> to run file run_rdf_to_kg.py 
```


## Environment Variables

Export the following environment variables:

```
OPENAI_API_KEY=your_openai_api_key
```

For the Cypher-based approach, also add:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## Usage

### 1. Building Knowledge Graphs

Build a knowledge graph for KG-RAG methods:

```bash
python -m scripts.build_cypher_graph
```
### 2. Running Evaluation
To interactively query using KG-RAG methods:

```bash
python -m scripts.run_cypher_rag 
```

## Development

### Pre-commit hooks

This project uses pre-commit hooks to ensure code quality:

```bash
# Run pre-commit hooks on all files
pre-commit run --all-files
```

## References

- [kg-rag](https://github.com/VectorInstitute/kg-rag)