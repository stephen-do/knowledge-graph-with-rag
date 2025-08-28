#!/usr/bin/env python
"""Gradio Chat App for Cypher-based KG-RAG."""

import os
import json
import gradio as gr
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from kg_rag.methods.cypher_based.kg_rag import CypherBasedKGRAG


# Load environment variables
load_dotenv()

# Config Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "adgjmptw1")

# Init Neo4j graph + LLM + KG-RAG system
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
llm = ChatOpenAI(temperature=0, model="gpt-4o")
kg_rag = CypherBasedKGRAG(graph=graph, llm=llm, max_depth=2, max_hops=3, verbose=True)


def chat(query, show_cypher):
    """Process a user query with KG-RAG (QA or Cypher mode)."""
    try:
        if show_cypher:
            cypher = kg_rag.get_explicit_cypher(query)
            result = kg_rag.run_cypher(cypher)
            return f"**Cypher Query:**\n```\n{cypher}\n```\n\n**Result:**\n```json\n{json.dumps(result, indent=2)}\n```"
        else:
            result = kg_rag.query(query)
            return f"**Answer:** {result.get('answer', 'N/A')}\n\n**Reasoning:** {result.get('reasoning', 'N/A')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Build Gradio UI
with gr.Blocks(title="Cypher-based KG-RAG Chat") as demo:
    gr.Markdown("# 🏥 Cypher-based KG-RAG Chat (Healthcare)")
    gr.Markdown("Ask questions about healthcare providers, patients, specializations, and locations.")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="KG-RAG Chat", height=500)
            user_input = gr.Textbox(placeholder="Ask me anything about the healthcare knowledge graph...", label="Your Question")
            with gr.Row():
                send_btn = gr.Button("Send")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            show_cypher = gr.Checkbox(label="Show Cypher Mode", value=False)

    history = []

    def respond(message, history, show_cypher):
        response = chat(message, show_cypher)
        history = history + [(message, response)]
        return history, history

    send_btn.click(respond, [user_input, chatbot, show_cypher], [chatbot, chatbot])
    user_input.submit(respond, [user_input, chatbot, show_cypher], [chatbot, chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860)
