from rdflib import Graph, Namespace
from rdflib.namespace import XSD, RDF

g = Graph()
g.parse("data\rdf\healthcare_ontology.ttl", format="turtle")
g.parse("data\rdf\healthcare_data.ttl", format="turtle")

HC = Namespace("http://example.org/healthcare#")
g.bind("", HC)

# Q1: Patients treated by Dr. Jessica Lee (name + condition)
q1 = """
PREFIX : <http://example.org/healthcare#>
SELECT ?patientName ?cond WHERE {
  :Dr_Jessica_Lee :TREATS ?p .
  ?p :name ?patientName ;
     :condition ?cond .
}
"""
print("Q1 Results:")
for row in g.query(q1):
    print(row)

# Q2: Doctors in Los_Angeles and their specializations
q2 = """
PREFIX : <http://example.org/healthcare#>
SELECT ?doc ?specName WHERE {
  ?doc :LOCATED_AT :Los_Angeles ;
       :SPECIALIZES_IN ?spec .
  ?spec :name ?specName .
}
"""
print("\nQ2 Results:")
for row in g.query(q2):
    print(row)

# Q3: Patients >= 65 with Asthma
q3 = """
PREFIX : <http://example.org/healthcare#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?pName ?age ?c WHERE {
  ?p a :Patient ;
     :name ?pName ; :age ?age ; :condition ?c .
  FILTER( xsd:integer(?age) >= 65 && lcase(str(?c)) = "asthma" )
}

"""
print("\nQ3 Results:")
for row in g.query(q3):
    print(row)
