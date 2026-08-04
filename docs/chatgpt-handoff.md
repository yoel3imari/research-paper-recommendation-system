# Research Paper Recommendation System – Design Handoff

## Project Goal

Design a scalable recommendation system that helps users discover relevant research papers **without storing or serving the full paper content**. The system stores only bibliographic metadata and references users to the original publication (DOI, publisher, arXiv, etc.).

The recommendation engine should be modular, scalable, explainable, and suitable for production deployment.

---

# Functional Requirements

The system should:

* Recommend papers relevant to a user's interests.
* Recommend similar papers from a paper detail page.
* Support keyword/topic search.
* Personalize recommendations based on user interactions.
* Return links to the original source rather than hosting documents.
* Continuously improve recommendations using interaction data.

The system **does not**:

* Store PDFs.
* Parse or chunk full-text documents.
* Build a RAG system.
* Answer questions about paper contents.

---

# High-Level Architecture

```
External Metadata APIs
        │
        ▼
Data Ingestion Pipeline
        │
Normalization & Deduplication
        │
─────────┬────────────────────────
         │
Metadata Database
(PostgreSQL)

Citation Relationships
(PostgreSQL or Neo4j)

Search Index
(OpenSearch)

Recommendation Engine

REST / GraphQL API

Frontend
```

The architecture should be modular so individual services can evolve independently.

---

# Recommendation Strategy

The recommendation system should use a hybrid approach.

## Candidate Generation

Generate candidate papers from multiple independent sources.

Examples include:

* Shared research topics
* Shared keywords
* Citation graph
* Co-citation
* Bibliographic coupling
* Same venue
* Same author
* Recent papers in the same field
* Popular papers
* User history

Each retrieval strategy contributes candidates to a common pool.

---

## Ranking

Each candidate receives a feature vector.

Potential features:

* Number of shared keywords
* Number of shared topics
* Citation overlap
* Co-citation score
* Bibliographic coupling score
* Same venue
* Same author
* Publication year
* Citation count
* User affinity toward authors
* User affinity toward topics
* Popularity score
* Freshness score

Initially, ranking can use weighted scoring.

As interaction data grows, replace this with a Learning-to-Rank model (e.g., LambdaMART using LightGBM).

---

## Personalization

Maintain a user profile containing:

* Saved papers
* Viewed papers
* Click history
* Favorite topics
* Favorite authors
* Preferred venues

Recommendations should combine:

* Paper similarity
* User interests
* Popularity
* Freshness
* Diversity

---

## Diversity

Avoid returning nearly identical papers.

Use a re-ranking stage (e.g., Maximum Marginal Relevance) to increase topic diversity while preserving relevance.

---

# Search

Provide traditional search using metadata fields.

Searchable fields:

* Title
* Authors
* Keywords
* Topics
* Venue
* Year

This search is independent from the recommendation engine.

---

# User Feedback Loop

Track events such as:

* View paper
* Click external link
* Save/bookmark
* Like
* Remove recommendation
* Follow author
* Follow topic

These events become training data for future ranking models.

---

# Suggested Technology Stack

## Backend

* FastAPI

## Metadata Database

* PostgreSQL

## Search

* OpenSearch or Elasticsearch

## Graph Layer

* PostgreSQL recursive queries (initially)
* Neo4j (optional for large-scale graph traversal)

## Cache

* Redis

## Workflow Orchestration

* Apache Airflow

## Experiment Tracking

* MLflow

## Ranking

* LightGBM LambdaMART
* XGBoost Ranker

---

# Scalability Considerations

The system should be designed around independent services:

* Metadata ingestion
* Search
* Recommendation
* User profile
* Ranking
* Analytics

This separation enables horizontal scaling and independent deployment.

---

# Non-Goals

The following are explicitly out of scope for this project:

* Hosting PDFs
* Parsing or indexing full paper text
* Chunking documents
* Building a Retrieval-Augmented Generation (RAG) pipeline
* Answering questions about paper contents
* Summarizing papers using their full text

The platform serves as a metadata-driven discovery and recommendation engine that directs users to the original publication source.
