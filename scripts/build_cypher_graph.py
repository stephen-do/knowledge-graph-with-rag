import csv
import os
from neo4j import GraphDatabase


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "adgjmptw1")
AUTH = (NEO4J_USER, NEO4J_PASSWORD)


# Function to connect and run a Cypher query
def execute_query(driver, cypher_query, parameters=None):
    try:
        with driver.session() as session:
            session.run(cypher_query, parameters)
    except Exception as e:
        print(f"Error: {e}")


def create_healthcare_provider_node(driver, provider, bio):
    create_provider_query = """
    MERGE (hp:HealthcareProvider {name: $provider})
    SET hp.bio = $bio
    """
    parameters = {"provider": provider, "bio": bio}
    execute_query(driver, create_provider_query, parameters)


def create_patient_node(driver, patient, patient_age, patient_gender, patient_condition):
    create_patient_query = """
    MERGE (p:Patient {name: $patient})
    SET p.age = $patient_age,
        p.gender = $patient_gender,
        p.condition = $patient_condition
    """
    parameters = {
        "patient": patient,
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "patient_condition": patient_condition,
    }
    execute_query(driver, create_patient_query, parameters)


def create_specialization_node(driver, specialization):
    create_specialization_query = """
    MERGE (s:Specialization {name: $specialization})
    """
    parameters = {"specialization": specialization}
    execute_query(driver, create_specialization_query, parameters)


def create_location_node(driver, location):
    create_location_query = """
    MERGE (l:Location {name: $location})
    """
    parameters = {"location": location}
    execute_query(driver, create_location_query, parameters)


def create_relationships(driver, provider, patient, specialization, location):
    create_relationships_query = """
    MATCH (hp:HealthcareProvider {name: $provider}), (p:Patient {name: $patient})
    MERGE (hp)-[:TREATS]->(p)
    WITH hp
    MATCH (hp), (s:Specialization {name: $specialization})
    MERGE (hp)-[:SPECIALIZES_IN]->(s)
    WITH hp
    MATCH (hp), (l:Location {name: $location})
    MERGE (hp)-[:LOCATED_AT]->(l)
    """
    parameters = {
        "provider": provider,
        "patient": patient,
        "specialization": specialization,
        "location": location,
    }
    execute_query(driver, create_relationships_query, parameters)


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)

    with open("healthcare.csv", mode="r") as file:  # chỉnh lại path file csv của bạn
        reader = csv.DictReader(file)
        print("Reading CSV file...")

        for row in reader:
            provider = row["Provider"]
            patient = row["Patient"]
            specialization = row["Specialization"]
            location = row["Location"]
            bio = row["Bio"]
            patient_age = row["Patient_Age"]
            patient_gender = row["Patient_Gender"]
            patient_condition = row["Patient_Condition"]

            create_healthcare_provider_node(driver, provider, bio)
            create_patient_node(driver, patient, patient_age, patient_gender, patient_condition)
            create_specialization_node(driver, specialization)
            create_location_node(driver, location)
            create_relationships(driver, provider, patient, specialization, location)

    driver.close()
    print("Graph populated successfully!")


if __name__ == "__main__":
    main()
