# rdf_build_healthcare.py
# pip install rdflib

import csv
import re
import os
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

# ---------- Config ----------
BASE = Namespace("http://example.org/healthcare#")

DATA_DIR = "data"
RDF_DIR = os.path.join(DATA_DIR, "rdf")
os.makedirs(RDF_DIR, exist_ok=True)

INPUT_CSV    = os.path.join(DATA_DIR, "healthcare.csv")
OUT_ONTOLOGY = os.path.join(RDF_DIR, "healthcare_ontology.ttl")
OUT_DATA     = os.path.join(RDF_DIR, "healthcare_data.ttl")

# CSV columns expected:
# Provider,Patient,Specialization,Location,Bio,Patient_Age,Patient_Gender,Patient_Condition

# ---------- Utilities ----------
def slugify(name: str) -> str:
    """Convert to safe local name for URIs."""
    s = re.sub(r"\s+", "_", (name or "").strip())
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"

def make_uri(local: str) -> URIRef:
    return BASE[slugify(local)]

def split_multi(val: str):
    """Split multi-values: supports | ; ,"""
    if not val:
        return []
    parts = re.split(r"[|;,]", val)
    return [p.strip() for p in parts if p.strip()]

def normpath(p: str) -> str:
    """Normalize path for OS (avoids issues with stray backslash escapes)."""
    return os.path.normpath(p)

# ---------- Ontology (RDF/RDFS) ----------
def build_ontology() -> Graph:
    g = Graph()
    g.bind("", BASE)
    g.bind("hc", BASE)
    g.bind("xsd", XSD)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)

    # Classes
    HealthcareProvider = BASE.HealthcareProvider
    Patient = BASE.Patient
    Specialization = BASE.Specialization
    Location = BASE.Location

    for cls, label, comment in [
        (HealthcareProvider, "Healthcare Provider", "A medical professional who provides healthcare services."),
        (Patient, "Patient", "An individual who receives healthcare services."),
        (Specialization, "Medical Specialization", "A medical specialty, e.g., Pediatrics, Cardiology."),
        (Location, "Location", "A city or place where providers/patients are located."),
    ]:
        g.add((cls, RDF.type, RDFS.Class))
        g.add((cls, RDFS.label, Literal(label, lang="en")))
        if comment:
            g.add((cls, RDFS.comment, Literal(comment, lang="en")))

    # Object properties
    LOCATED_AT = BASE.LOCATED_AT
    SPECIALIZES_IN = BASE.SPECIALIZES_IN
    TREATS = BASE.TREATS

    g.add((LOCATED_AT, RDF.type, RDF.Property))
    g.add((LOCATED_AT, RDFS.label, Literal("located at", lang="en")))
    g.add((LOCATED_AT, RDFS.domain, HealthcareProvider))
    g.add((LOCATED_AT, RDFS.range, Location))

    g.add((SPECIALIZES_IN, RDF.type, RDF.Property))
    g.add((SPECIALIZES_IN, RDFS.label, Literal("specializes in", lang="en")))
    g.add((SPECIALIZES_IN, RDFS.domain, HealthcareProvider))
    g.add((SPECIALIZES_IN, RDFS.range, Specialization))

    g.add((TREATS, RDF.type, RDF.Property))
    g.add((TREATS, RDFS.label, Literal("treats", lang="en")))
    g.add((TREATS, RDFS.domain, HealthcareProvider))
    g.add((TREATS, RDFS.range, Patient))

    # Datatype properties
    for dp, rng in [
        (BASE.name, XSD.string),
        (BASE.bio, XSD.string),
        (BASE.age, XSD.int),
        (BASE.gender, XSD.string),
        (BASE.condition, XSD.string),
    ]:
        g.add((dp, RDF.type, RDF.Property))
        g.add((dp, RDFS.range, rng))

    # Domains for datatype properties
    g.add((BASE.bio, RDFS.domain, HealthcareProvider))
    g.add((BASE.age, RDFS.domain, Patient))
    g.add((BASE.gender, RDFS.domain, Patient))
    g.add((BASE.condition, RDFS.domain, Patient))

    return g

# ---------- Data graph from CSV ----------
def build_data(csv_path: str) -> Graph:
    g = Graph()
    g.bind("", BASE)
    g.bind("hc", BASE)
    g.bind("xsd", XSD)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)

    # Cache + single-valued guards (first-wins)
    uri_cache = {}
    single_set = {
        "name": set(),
        "bio": set(),
        "age": set(),
        "gender": set(),
    }

    def get_entity(kind: str, label_text: str) -> URIRef:
        key = (kind, (label_text or "").strip())
        if key in uri_cache:
            return uri_cache[key]

        uri = make_uri(label_text or f"{kind}_unnamed")
        uri_cache[key] = uri

        # type
        if kind == "Provider":
            g.add((uri, RDF.type, BASE.HealthcareProvider))
        elif kind == "Patient":
            g.add((uri, RDF.type, BASE.Patient))
        elif kind == "Specialization":
            g.add((uri, RDF.type, BASE.Specialization))
        elif kind == "Location":
            g.add((uri, RDF.type, BASE.Location))

        # name (single)
        if label_text and uri not in single_set["name"]:
            g.add((uri, BASE.name, Literal(label_text)))
            single_set["name"].add(uri)

        return uri

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            provider_name = (row.get("Provider") or "").strip()
            patient_name = (row.get("Patient") or "").strip()
            specs_raw   = (row.get("Specialization") or "").strip()
            locs_raw    = (row.get("Location") or "").strip()
            bio         = (row.get("Bio") or "").strip()
            age_raw     = (row.get("Patient_Age") or "").strip()
            gender_raw  = (row.get("Patient_Gender") or "").strip()
            cond_raw    = (row.get("Patient_Condition") or "").strip()

            if not provider_name or not patient_name:
                continue

            prov = get_entity("Provider", provider_name)
            pat  = get_entity("Patient", patient_name)

            # multi-values supported
            for s in (split_multi(specs_raw) or ([specs_raw] if specs_raw else [])):
                spec = get_entity("Specialization", s)
                g.add((prov, BASE.SPECIALIZES_IN, spec))

            for c in (split_multi(locs_raw) or ([locs_raw] if locs_raw else [])):
                loc = get_entity("Location", c)
                g.add((prov, BASE.LOCATED_AT, loc))

            # relation
            g.add((prov, BASE.TREATS, pat))

            # provider attrs (single)
            if bio and prov not in single_set["bio"]:
                g.add((prov, BASE.bio, Literal(bio)))
                single_set["bio"].add(prov)

            # patient attrs
            if gender_raw and pat not in single_set["gender"]:
                g.add((pat, BASE.gender, Literal(gender_raw)))
                single_set["gender"].add(pat)

            if cond_raw:
                for c in (split_multi(cond_raw) or [cond_raw]):
                    g.add((pat, BASE.condition, Literal(c)))

            if age_raw and pat not in single_set["age"]:
                try:
                    g.add((pat, BASE.age, Literal(int(age_raw), datatype=XSD.int)))
                except ValueError:
                    g.add((pat, BASE.age, Literal(age_raw)))
                single_set["age"].add(pat)

    return g

# ---------- Main ----------
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"CSV not found: {INPUT_CSV}")

    ont_graph = build_ontology()
    ont_path = normpath(OUT_ONTOLOGY)
    ont_graph.serialize(ont_path, format="turtle")
    print(f"[OK] Wrote ontology to {ont_path}")

    data_graph = build_data(INPUT_CSV)
    data_path = normpath(OUT_DATA)
    data_graph.serialize(data_path, format="turtle")
    print(f"[OK] Wrote data to {data_path}")
