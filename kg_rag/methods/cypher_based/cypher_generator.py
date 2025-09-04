"""Cypher query generator for Cypher-based KG-RAG approach."""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


CYPHER_GENERATION_TEMPLATE = """
Task:
Generate Cypher statement to query a healthcare knowledge graph.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
If you cannot generate a Cypher statement based on the provided schema, explain the reason to the user.
When you use MATCH with WHERE clauses, always first check the entities' or relationships' id property rather than name.
Schema:
{schema}
IMPORTANT:
Do not include any explanations or apologies in your response.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Do not include markdown: ```cypher
Examples:
Here are a few examples of generated Cypher statements for particular healthcare questions:

# Example 1: Which patients are treated by Dr. Smith?
CALL db.index.fulltext.queryNodes('ft_hp', 'Dr. Smith') 
YIELD node AS hp, score AS hpScore
WITH hp, hpScore
ORDER BY hpScore DESC LIMIT 1
MATCH (hp)-[:TREATS]->(p:Patient)
RETURN
  p,
  hp.name AS matched_provider,
  hpScore AS provider_score
ORDER BY provider_score DESC
LIMIT 100;

# Example 2: What specialization does Dr. Brown have?
CALL db.index.fulltext.queryNodes('ft_hp', 'Dr. Brown')
YIELD node AS hp, score AS hpScore
WITH hp, hpScore
ORDER BY hpScore DESC LIMIT 1
MATCH (hp)-[:SPECIALIZES_IN]->(s:Specialization)
RETURN
s,
hp.name AS matched_provider,
hpScore AS provider_score
ORDER BY provider_score DESC
LIMIT 5;

# Example 3: Which healthcare providers are located in New York?
CALL db.index.fulltext.queryNodes('ft_loc', 'New York')
YIELD node AS l, score AS locScore
WITH l, locScore
ORDER BY locScore DESC LIMIT 1
MATCH (hp:HealthcareProvider)-[:LOCATED_AT]->(l)
RETURN
DISTINCT hp,
l.name AS matched_location,
locScore AS location_score
ORDER BY location_score DESC
LIMIT 25;

# Example 4 (Multi-hop):Which patients are treated by healthcare providers named Sarah located in Los Angeles?
CALL db.index.fulltext.queryNodes('ft_loc', 'Los Angeles')
YIELD node AS l, score AS locScore
WITH l, locScore
ORDER BY locScore DESC LIMIT 1
CALL db.index.fulltext.queryNodes('ft_hp', 'Sarah')
YIELD node AS hp, score AS hpScore
WITH l, locScore, hp, hpScore
MATCH (hp)-[:LOCATED_AT]->(l)
MATCH (hp)-[:TREATS]->(p:Patient)
RETURN
  p,
  hp.name AS matched_provider,
  l.name AS matched_location,
  hpScore AS provider_score,
  locScore AS location_score
ORDER BY provider_score DESC, p.name ASC
LIMIT 25;

# Example 5 (Multi-hop + Aggregation): For Dr. Sarah Johnson in Los Angeles, what is the total number of patients she treats and what is their average age?
CALL db.index.fulltext.queryNodes('ft_loc', 'Los Angeles')
YIELD node AS l, score AS locScore
WITH l
ORDER BY locScore DESC LIMIT 1

CALL db.index.fulltext.queryNodes('ft_hp', 'Sarah')
YIELD node AS hp, score AS hpScore
WITH l, hp
MATCH (hp)-[:LOCATED_AT]->(l)
MATCH (hp)-[:TREATS]->(p:Patient)
RETURN
  hp.name AS matched_provider,
  l.name AS matched_location,
  count(DISTINCT p) AS total_patients,
  round(avg(p.age), 1) AS avg_age;

The question is:
{question}

"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)


class CypherGenerator:
    """Generates Cypher queries from natural language questions."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        cypher_prompt: PromptTemplate | None = None,
        verbose: bool = False,
    ):
        """
        Initialize the Cypher generator.

        Args:
            llm: LLM to use for query generation (default: ChatOpenAI with gpt-4o)
            cypher_prompt: Prompt template for Cypher generation
            verbose: Whether to print verbose output
        """
        # Set up LLM if not provided
        if llm is None:
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        else:
            self.llm = llm

        # Set up prompt template
        self.prompt = cypher_prompt or CYPHER_GENERATION_PROMPT

        # Other settings
        self.verbose = verbose

    def get_schema(self, graph):
        query = """
        CALL apoc.meta.schema()
        YIELD value
        RETURN value
        """
        schema_result = graph.query(query)

        if not schema_result:
            return "No schema information available."

        schema_map = schema_result[0]["value"]

        node_info = []
        relation_patterns = []

        for label, info in schema_map.items():
            # Node properties
            properties = info.get("properties", {})
            property_str = ", ".join([f"{k}: {v}" for k, v in properties.items()])
            node_info.append(f"{label} {{{property_str}}}")

            # Relationships
            relationships = info.get("relationships", {})
            for rel_type, targets in relationships.items():
                for target in targets:
                    relation_patterns.append(f"(:{label})-[:{rel_type}]->(:{target})")

        schema = []
        if node_info:
            schema.append("Node properties:")
            schema.append("\n".join(node_info))

        if relation_patterns:
            schema.append("\nRelationships:")
            schema.append("\n".join(relation_patterns))

        return "\n".join(schema)

    def generate_cypher(self, question: str, schema: str) -> str:
        """
        Generate a Cypher query from a natural language question.

        Args:
            question: Natural language question
            schema: Schema of the graph

        Returns
        -------
            Generated Cypher query
        """
        if self.verbose:
            print(f"Generating Cypher query for: {question}")

        # Generate Cypher query
        response = self.llm.invoke(self.prompt.format(schema=schema, question=question))

        query = response.content if hasattr(response, "content") else str(response)

        if self.verbose:
            print(f"Generated Cypher query: {query}")

        if isinstance(query, str):
            return query
        raise ValueError(f"Error generating Cypher query: {query}")
