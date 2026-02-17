import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
import uuid
import logging
import json
import re
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeCore:
    def __init__(self):
        # Initialize directed graph for RDF-like triples
        self.graph = nx.DiGraph()
        # Store ontologies (entity types and their hierarchies)
        self.ontologies: Dict[str, Dict[str, Any]] = {}
        # Store temporal metadata for relations
        self.temporal_data: Dict[str, Dict[str, datetime]] = {}
        # Store relation types (predicate constraints)
        self.relation_types: Dict[str, Dict[str, str]] = {}
        # Index entities by type for faster queries
        self.entity_type_index: Dict[str, Set[str]] = {}
        # Index relations by predicate for faster queries
        self.predicate_index: Dict[str, Set[Tuple[str, str]]] = {}
        # Schema version for persistence
        self.schema_version = "1.2"

    def define_ontology(self, entity_type: str, parent_type: str = None, 
                       attributes: Dict[str, type] = None) -> None:
        """Define or update an ontology with hierarchical relationships and required attributes."""
        if attributes is None:
            attributes = {}
        old_parent = self.ontologies.get(entity_type, {}).get('parent')
        if entity_type not in self.ontologies:
            self.ontologies[entity_type] = {'parent': parent_type, 'children': set(), 'attributes': attributes}
            self.entity_type_index[entity_type] = set()
        else:
            self.ontologies[entity_type].update({'parent': parent_type, 'attributes': attributes})
        
        if parent_type:
            if parent_type not in self.ontologies:
                logger.warning(f"Parent type {parent_type} not defined. Creating default ontology.")
                self.define_ontology(parent_type)
            self.ontologies[parent_type]['children'].add(entity_type)
        
        if old_parent and old_parent != parent_type:
            self.ontologies[old_parent]['children'].discard(entity_type)
            for entity_id in self.entity_type_index.get(entity_type, set()):
                self._validate_entity_attributes(entity_id)
        logger.info(f"Defined ontology for {entity_type} with parent {parent_type}.")

    def remove_ontology(self, entity_type: str) -> None:
        """Remove an ontology, its entities, relations, and dependent relation types."""
        if entity_type not in self.ontologies:
            logger.warning(f"Ontology {entity_type} not found.")
            return
        for t in self.ontologies:
            if self.ontology_alignment(t, entity_type) and t != entity_type:
                if self.entity_type_index.get(t):
                    logger.warning(f"Cannot remove ontology {entity_type} due to entities in derived type {t}.")
                    return
        for child in self.ontologies[entity_type]['children'].copy():
            self.remove_ontology(child)
        for entity_id in self.entity_type_index.get(entity_type, set()).copy():
            self.remove_entity(entity_id)
        predicates_to_remove = [p for p, v in self.relation_types.items() 
                              if v['subject_type'] == entity_type or v['object_type'] == entity_type]
        for predicate in predicates_to_remove:
            self.remove_relation_type(predicate)
        parent = self.ontologies[entity_type]['parent']
        if parent and parent in self.ontologies:
            self.ontologies[parent]['children'].discard(entity_type)
        del self.ontologies[entity_type]
        del self.entity_type_index[entity_type]
        logger.info(f"Removed ontology {entity_type}.")

    def define_relation_type(self, predicate: str, subject_type: str, object_type: str) -> None:
        """Define a relation type with allowed subject and object types."""
        if subject_type not in self.ontologies or object_type not in self.ontologies:
            logger.error(f"Invalid relation type: {subject_type} or {object_type} not in ontologies.")
            raise ValueError(f"Subject type {subject_type} or object type {object_type} not defined.")
        self.relation_types[predicate] = {'subject_type': subject_type, 'object_type': object_type}
        edges_to_remove = []
        for u, v, data in self.graph.edges(data=True):
            if data['predicate'] == predicate:
                subject_type = self.graph.nodes[u]['type']
                object_type = self.graph.nodes[v]['type']
                if not (self.ontology_alignment(subject_type, self.relation_types[predicate]['subject_type']) and 
                        self.ontology_alignment(object_type, self.relation_types[predicate]['object_type'])):
                    edges_to_remove.append((u, v, data['edge_id']))
        for u, v, edge_id in edges_to_remove:
            self.graph.remove_edge(u, v)
            self.temporal_data.pop(edge_id, None)
            self.predicate_index.get(predicate, set()).discard((u, v))
        if edges_to_remove:
            self.path_reasoning.cache_clear()
            self.query_triples.cache_clear()
            logger.info(f"Removed {len(edges_to_remove)} invalid relations for redefined predicate {predicate}.")
        logger.info(f"Defined relation type {predicate} for {subject_type} -> {object_type}.")

    def remove_relation_type(self, predicate: str) -> None:
        """Remove a relation type and associated relations."""
        if predicate not in self.relation_types:
            logger.warning(f"Relation type {predicate} not found.")
            return
        del self.relation_types[predicate]
        edges_to_remove = [(u, v) for u, v, d in self.graph.edges(data=True) if d['predicate'] == predicate]
        for u, v in edges_to_remove:
            edge_id = self.graph[u][v]['edge_id']
            self.graph.remove_edge(u, v)
            self.temporal_data.pop(edge_id, None)
            self.predicate_index.get(predicate, set()).discard((u, v))
        self.path_reasoning.cache_clear()
        self.query_triples.cache_clear()
        logger.info(f"Removed relation type {predicate} and associated relations.")

    def _validate_entity_attributes(self, entity_id: str) -> None:
        """Validate an entity's attributes against its ontology and parent ontologies."""
        if entity_id not in self.graph:
            logger.error(f"Entity {entity_id} not found in graph.")
            raise ValueError(f"Entity {entity_id} not found in graph.")
        entity_type = self.graph.nodes[entity_id]['type']
        attributes = self.graph.nodes[entity_id]
        required_attrs = {}
        current_type = entity_type
        while current_type in self.ontologies:
            required_attrs.update(self.ontologies[current_type].get('attributes', {}))
            current_type = self.ontologies[current_type].get('parent')
        
        for attr, attr_type in required_attrs.items():
            if attr not in attributes:
                logger.error(f"Missing required attribute '{attr}' for entity {entity_id} of type {entity_type}.")
                raise ValueError(f"Missing required attribute '{attr}' for entity {entity_id}.")
            if not isinstance(attributes[attr], attr_type):
                logger.error(f"Attribute '{attr}' for entity {entity_id} must be {attr_type}, got {type(attributes[attr])}.")
                raise ValueError(f"Attribute '{attr}' must be {attr_type}, got {type(attributes[attr])}.")

    def _validate_entity_attributes_dict(self, entity_type: str, attributes: Dict[str, Any]) -> None:
        """Validate attributes against ontology and parent ontologies."""
        required_attrs = {}
        current_type = entity_type
        while current_type in self.ontologies:
            required_attrs.update(self.ontologies[current_type].get('attributes', {}))
            current_type = self.ontologies[current_type].get('parent')
        
        for attr, attr_type in required_attrs.items():
            if attr not in attributes:
                logger.error(f"Missing required attribute '{attr}' for entity type {entity_type}.")
                raise ValueError(f"Missing required attribute '{attr}' for entity type {entity_type}.")
            if not isinstance(attributes[attr], attr_type):
                logger.error(f"Attribute '{attr}' must be of type {attr_type}, got {type(attributes[attr])}.")
                raise ValueError(f"Attribute '{attr}' must be of type {attr_type}, got {type(attributes[attr])}.")

    def add_entity(self, entity_id: str, entity_type: str, attributes: Dict[str, Any] = None) -> None:
        """Add an entity to the knowledge graph with attribute validation."""
        if attributes is None:
            attributes = {}
        if entity_id in self.graph:
            logger.warning(f"Entity {entity_id} already exists in the graph.")
            return
        if entity_type not in self.ontologies:
            logger.warning(f"Entity type {entity_type} not defined. Creating default ontology.")
            self.define_ontology(entity_type)
        
        self._validate_entity_attributes_dict(entity_type, attributes)
        attributes['type'] = entity_type
        self.graph.add_node(entity_id, **attributes)
        self.entity_type_index[entity_type].add(entity_id)
        self.path_reasoning.cache_clear()
        self.query_triples.cache_clear()
        logger.info(f"Added entity {entity_id} of type {entity_type}.")

    def add_entities(self, entities: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        """Add multiple entities in a batch, clearing caches once."""
        for entity_id, entity_type, attributes in entities:
            if attributes is None:
                attributes = {}
            if entity_id in self.graph:
                logger.warning(f"Entity {entity_id} already exists in the graph.")
                continue
            if entity_type not in self.ontologies:
                logger.warning(f"Entity type {entity_type} not defined. Creating default ontology.")
                self.define_ontology(entity_type)
            try:
                self._validate_entity_attributes_dict(entity_type, attributes)
                attributes['type'] = entity_type
                self.graph.add_node(entity_id, **attributes)
                self.entity_type_index[entity_type].add(entity_id)
                logger.info(f"Added entity {entity_id} of type {entity_type}.")
            except ValueError as e:
                logger.error(f"Failed to add entity {entity_id}: {e}")
                raise
        self.path_reasoning.cache_clear()
        self.query_triples.cache_clear()

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity and its associated relations from the graph."""
        if entity_id not in self.graph:
            logger.warning(f"Entity {entity_id} not found in graph.")
            return
        entity_type = self.graph.nodes[entity_id]['type']
        for u, v, data in list(self.graph.edges(data=True)):
            if u == entity_id or v == entity_id:
                predicate = data['predicate']
                self.predicate_index.get(predicate, set()).discard((u, v))
                self.temporal_data.pop(data['edge_id'], None)
        self.graph.remove_node(entity_id)
        self.entity_type_index[entity_type].discard(entity_id)
        self.path_reasoning.cache_clear()
        self.query_triples.cache_clear()
        logger.info(f"Removed entity {entity_id}.")

    def remove_entities(self, entity_ids: List[str]) -> None:
        """Remove multiple entities in a batch, clearing caches once."""
        for entity_id in entity_ids:
            if entity_id not in self.graph:
                logger.warning(f"Entity {entity_id} not found in graph.")
                continue
            entity_type = self.graph.nodes[entity_id]['type']
            for u, v, data in list(self.graph.edges(data=True)):
                if u == entity_id or v == entity_id:
                    predicate = data['predicate']
                    self.predicate_index.get(predicate, set()).discard((u, v))
                    self.temporal_data.pop(data['edge_id'], None)
            self.graph.remove_node(entity_id)
            self.entity_type_index[entity_type].discard(entity_id)
            logger.info(f"Removed entity {entity_id}.")
        self.path_reasoning.cache_clear()
        self.query_triples.cache_clear()

    def add_relation(self, subject_id: str, predicate: str, object_id: str, 
                    start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, 
                    weight: float = 1.0) -> None:
        """Add a relation (triple) to the knowledge graph with temporal range and weight."""
        if subject_id not in self.graph or object_id not in self.graph:
            logger.error(f"Cannot add relation: Subject {subject_id} or object {object_id} not found.")
            raise ValueError(f"Subject {subject_id} or object {object_id} not found in graph.")
        if weight <= 0 or weight > 1.0:
            logger.error(f"Invalid weight {weight}. Weight must be in (0, 1].")
            raise ValueError(f"Weight must be in (0, 1].")
        if start_time and end_time and start_time > end_time:
            logger.error(f"Invalid temporal range: start_time {start_time} > end_time {end_time}.")
            raise ValueError(f"start_time must be <= end_time.")
        
        if predicate in self.relation_types:
            subject_type = self.graph.nodes[subject_id]['type']
            object_type = self.graph.nodes[object_id]['type']
            if not (self.ontology_alignment(subject_type, self.relation_types[predicate]['subject_type']) and 
                    self.ontology_alignment(object_type, self.relation_types[predicate]['object_type'])):
                logger.error(f"Invalid relation: {predicate} requires {self.relation_types[predicate]['subject_type']} -> {self.relation_types[predicate]['object_type']}.")
                raise ValueError(f"Invalid relation: {predicate} requires {self.relation_types[predicate]['subject_type']} -> {self.relation_types[predicate]['object_type']}.")
        
        edge_id = str(uuid.uuid4())
        self.graph.add_edge(subject_id, object_id, predicate=predicate, edge_id=edge_id, weight=weight)
        self.predicate_index.setdefault(predicate, set()).add((subject_id, object_id))
        if start_time or end_time:
            self.temporal_data[edge_id] = {}
            if start_time:
                self.temporal_data[edge_id]['start_time'] = start_time
            if end_time:
                self.temporal_data[edge_id]['end_time'] = end_time
        self.path_reasoning.cache_clear()
        self.query_triples.cache_clear()
        logger.info(f"Added relation {subject_id} -[{predicate}]-> {object_id} with weight {weight}.")

    def add_relations(self, relations: List[Tuple[str, str, str, Optional[datetime], Optional[datetime], float]]) -> None:
        """Add multiple relations in a batch, clearing caches once."""
        for subject_id, predicate, object_id, start_time, end_time, weight in relations:
            try:
                self.add_relation(subject_id, predicate, object_id, start_time, end_time, weight)
            except ValueError as e:
                logger.error(f"Failed to add relation {subject_id} -[{predicate}]-> {object_id}: {e}")
                raise
        logger.info(f"Added {len(relations)} relations in batch.")

    def remove_relation(self, edge_id: str) -> None:
        """Remove a relation by its edge_id."""
        for u, v, data in list(self.graph.edges(data=True)):
            if data['edge_id'] == edge_id:
                predicate = data['predicate']
                self.graph.remove_edge(u, v)
                self.predicate_index.get(predicate, set()).discard((u, v))
                self.temporal_data.pop(edge_id, None)
                self.path_reasoning.cache_clear()
                self.query_triples.cache_clear()
                logger.info(f"Removed relation with edge_id {edge_id}.")
                return
        logger.warning(f"Edge {edge_id} not found in graph.")

    def remove_relations(self, edge_ids: List[str]) -> None:
        """Remove multiple relations by their edge_ids, clearing caches once."""
        for edge_id in edge_ids:
            for u, v, data in list(self.graph.edges(data=True)):
                if data['edge_id'] == edge_id:
                    predicate = data['predicate']
                    self.graph.remove_edge(u, v)
                    self.predicate_index.get(predicate, set()).discard((u, v))
                    self.temporal_data.pop(edge_id, None)
                    logger.info(f"Removed relation with edge_id {edge_id}.")
                    break
            else:
                logger.warning(f"Edge {edge_id} not found in graph.")
        self.path_reasoning.cache_clear()
        self.query_triples.cache_clear()

    def update_relation(self, edge_id: str, new_weight: Optional[float] = None, 
                       new_start_time: Optional[datetime] = None, 
                       new_end_time: Optional[datetime] = None) -> None:
        """Update an existing relation's weight or temporal metadata."""
        for u, v, data in self.graph.edges(data=True):
            if data['edge_id'] == edge_id:
                if new_weight is not None:
                    if new_weight <= 0 or new_weight > 1.0:
                        logger.error(f"Invalid weight {new_weight}. Weight must be in (0, 1].")
                        raise ValueError(f"Weight must be in (0, 1].")
                    data['weight'] = new_weight
                if new_start_time or new_end_time:
                    self.temporal_data[edge_id] = self.temporal_data.get(edge_id, {})
                    if new_start_time:
                        self.temporal_data[edge_id]['start_time'] = new_start_time
                    if new_end_time:
                        self.temporal_data[edge_id]['end_time'] = new_end_time
                    start_time = self.temporal_data[edge_id].get('start_time')
                    end_time = self.temporal_data[edge_id].get('end_time')
                    if start_time and end_time and start_time > end_time:
                        logger.error(f"Invalid temporal range: start_time {start_time} > end_time {end_time}.")
                        raise ValueError(f"start_time must be <= end_time.")
                self.path_reasoning.cache_clear()
                self.query_triples.cache_clear()
                logger.info(f"Updated relation with edge_id {edge_id}.")
                return
        logger.warning(f"Edge {edge_id} not found in graph.")

    def graph_search(self, start_id: str, target_id: str, max_depth: int = 5, max_paths: int = 100) -> List[List[str]]:
        """Find up to max_paths simple paths between two entities in the graph."""
        if start_id not in self.graph or target_id not in self.graph:
            logger.warning(f"Node {start_id} or {target_id} not found in graph.")
            return []
        try:
            paths = []
            for i, path in enumerate(nx.shortest_simple_paths(self.graph, start_id, target_id)):
                if i >= max_paths or len(path) > max_depth + 1:
                    break
                paths.append(path)
            logger.info(f"Found {len(paths)} paths from {start_id} to {target_id}.")
            return paths
        except nx.NetworkXNoPath:
            logger.info(f"No paths found from {start_id} to {target_id}.")
            return []

    @lru_cache(maxsize=1000)
    def path_reasoning(self, start_id: str, target_id: str, max_depth: int = 5, 
                      confidence_model: str = 'length_normalized', decay_factor: float = 0.8, 
                      min_confidence: float = 0.0) -> List[Tuple[List[str], float]]:
        """Perform path reasoning with confidence scores based on path length and edge weights."""
        if decay_factor <= 0 or decay_factor > 1.0:
            logger.error(f"Invalid decay_factor {decay_factor}. Must be in (0, 1].")
            raise ValueError(f"decay_factor must be in (0, 1].")
        if min_confidence < 0 or min_confidence > 1.0:
            logger.error(f"Invalid min_confidence {min_confidence}. Must be in [0, 1].")
            raise ValueError(f"min_confidence must be in [0, 1].")
        
        paths = self.graph_search(start_id, target_id, max_depth)
        weighted_paths = []
        for path in paths:
            path_weight = 1.0
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                path_weight *= edge_data.get('weight', 1.0)
            if confidence_model == 'length_normalized':
                confidence = path_weight / (len(path) - 1) if len(path) > 1 else path_weight
            elif confidence_model == 'exponential':
                confidence = path_weight * (decay_factor ** (len(path) - 1))
            else:
                confidence = path_weight
            if confidence >= min_confidence:
                weighted_paths.append((path, confidence))
        logger.info(f"Computed {len(weighted_paths)} weighted paths from {start_id} to {target_id}.")
        return weighted_paths

    def hybrid_query(self, entity_x: str, entity_y: str, 
                    temporal_start: Optional[datetime] = None, 
                    temporal_end: Optional[datetime] = None, 
                    confidence_model: str = 'length_normalized', 
                    min_confidence: float = 0.0) -> float:
        """Calculate probability of connection between entities with temporal range rules."""
        if temporal_start and temporal_end and temporal_start > temporal_end:
            logger.error(f"Invalid temporal range: temporal_start {temporal_start} > temporal_end {temporal_end}.")
            raise ValueError(f"temporal_start must be <= temporal_end.")
        
        paths = self.path_reasoning(entity_x, entity_y, confidence_model=confidence_model, min_confidence=min_confidence)
        if not paths:
            logger.info(f"No valid paths found for hybrid query between {entity_x} and {entity_y}.")
            return 0.0
        
        total_confidence = 0.0
        for path, confidence in paths:
            valid_path = True
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                edge_id = edge_data['edge_id']
                if edge_id in self.temporal_data:
                    edge_time = self.temporal_data[edge_id]
                    edge_start = edge_time.get('start_time')
                    edge_end = edge_time.get('end_time')
                    if temporal_start and edge_end and temporal_start > edge_end:
                        valid_path = False
                        break
                    if temporal_end and edge_start and temporal_end < edge_start:
                        valid_path = False
                        break
            if valid_path:
                total_confidence += confidence
        
        probability = min(total_confidence, 1.0)
        logger.info(f"Hybrid query probability between {entity_x} and {entity_y}: {probability}.")
        return probability

    @lru_cache(maxsize=1000)
    def query_triples(self, subject: Optional[str] = None, predicate: Optional[str] = None, 
                      object_id: Optional[str] = None, 
                      temporal_start: Optional[datetime] = None, 
                      temporal_end: Optional[datetime] = None) -> List[Tuple[str, str, str]]:
        """Query RDF-like triples matching a pattern with optional temporal constraints."""
        if temporal_start and temporal_end and temporal_start > temporal_end:
            logger.error(f"Invalid temporal range: temporal_start {temporal_start} > temporal_end {temporal_end}.")
            raise ValueError(f"temporal_start must be <= temporal_end.")
        
        results = []
        candidates = self.predicate_index.get(predicate, self.graph.edges()) if predicate else self.graph.edges()
        subject_candidates = self.graph.nodes
        object_candidates = self.graph.nodes
        if predicate in self.relation_types:
            subject_type = self.relation_types[predicate]['subject_type']
            object_type = self.relation_types[predicate]['object_type']
            subject_candidates = set()
            object_candidates = set()
            for t in self.ontologies:
                if self.ontology_alignment(t, subject_type):
                    subject_candidates.update(self.entity_type_index.get(t, set()))
                if self.ontology_alignment(t, object_type):
                    object_candidates.update(self.entity_type_index.get(t, set()))
        
        for u, v, data in candidates:
            edge_id = data['edge_id']
            valid_time = True
            if edge_id in self.temporal_data:
                edge_time = self.temporal_data[edge_id]
                edge_start = edge_time.get('start_time')
                edge_end = edge_time.get('end_time')
                if temporal_start and edge_end and temporal_start > edge_end:
                    valid_time = False
                if temporal_end and edge_start and temporal_end < edge_start:
                    valid_time = False
            if valid_time and \
               (subject is None or u == subject) and \
               (predicate is None or data['predicate'] == predicate) and \
               (object_id is None or v == object_id) and \
               (u in subject_candidates) and (v in object_candidates):
                results.append((u, data['predicate'], v))
        logger.info(f"Found {len(results)} triples matching pattern ({subject}, {predicate}, {object_id}).")
        return results

    def parse_query(self, query: str, 
                   temporal_start: Optional[datetime] = None, 
                   temporal_end: Optional[datetime] = None,
                   min_confidence: float = 0.0) -> List[Tuple[str, str, str]]:
        """Parse a query pattern with optional per-pattern temporal constraints."""
        if not query.strip():
            logger.error("Empty query provided.")
            raise ValueError("Query cannot be empty.")
        if temporal_start and temporal_end and temporal_start > temporal_end:
            logger.error(f"Invalid temporal range: temporal_start {temporal_start} > temporal_end {temporal_end}.")
            raise ValueError(f"temporal_start must be <= temporal_end.")
        if min_confidence < 0 or min_confidence > 1.0:
            logger.error(f"Invalid min_confidence {min_confidence}. Must be in [0, 1].")
            raise ValueError(f"min_confidence must be in [0, 1].")
        
        # Regex for ISO 8601 datetime (supports YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
        datetime_pattern = r'(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?)'
        pattern_regex = rf'([^ ]+)\s+([^ ]+)\s+([^ ]+)(?:\s*{{{datetime_pattern},{datetime_pattern}}})?'
        patterns = [p.strip() for p in query.split(' . ') if p.strip()]
        if not patterns:
            logger.error("No valid patterns in query.")
            raise ValueError("No valid patterns in query.")
        
        bindings: Dict[str, Set[str]] = {}
        results = []
        pattern_triples = []
        for pattern in patterns:
            match = re.match(pattern_regex, pattern)
            if not match:
                logger.error(f"Invalid query pattern: {pattern}. Expected format: subject predicate object [{{start_time,end_time}}]")
                raise ValueError(f"Invalid query pattern: {pattern}")
            subj, pred, obj, start_str, end_str = match.groups()
            if (subj.startswith('?') and not subj[1:].isalnum()) or (obj.startswith('?') and not obj[1:].isalnum()):
                logger.error(f"Invalid variable name in pattern: {pattern}. Variables must be alphanumeric.")
                raise ValueError(f"Variable names must be alphanumeric: {pattern}")
            
            pattern_start = temporal_start
            pattern_end = temporal_end
            if start_str and end_str:
                try:
                    pattern_start = datetime.fromisoformat(start_str)
                    pattern_end = datetime.fromisoformat(end_str)
                    if pattern_start > pattern_end:
                        logger.error(f"Invalid temporal range in pattern {pattern}: {start_str} > {end_str}")
                        raise ValueError(f"Invalid temporal range in pattern {pattern}")
                except ValueError as e:
                    logger.error(f"Invalid datetime format in pattern {pattern}: {e}")
                    raise ValueError(f"Invalid datetime format in pattern {pattern}")
            
            triples = self.query_triples(subject=None if subj.startswith('?') else subj, 
                                       predicate=pred, 
                                       object_id=None if obj.startswith('?') else obj,
                                       temporal_start=pattern_start,
                                       temporal_end=pattern_end)
            if not triples:
                logger.info(f"No matches for pattern {pattern}.")
                return []
            pattern_triples.append((pattern, triples))
            for s, p, o in triples:
                if subj.startswith('?'):
                    bindings.setdefault(subj, set()).add(s)
                if obj.startswith('?'):
                    bindings.setdefault(obj, set()).add(o)
        
        if len(patterns) > 1:
            for var in bindings:
                var_values = set.intersection(*(set(t[0] if var == p.split()[0] else t[2] 
                                                  for t in triples) 
                                              for p, triples in pattern_triples 
                                              if var in p.split()))
                if not var_values:
                    logger.info(f"No consistent bindings for variable {var}.")
                    return []
                bindings[var] = var_values
            
            results = []
            for pattern, triples in pattern_triples:
                subj, _, obj = pattern.split(maxsplit=2)[:3]
                for s, p, o in triples:
                    if (not subj.startswith('?') or s in bindings.get(subj, {s})) and \
                       (not obj.startswith('?') or o in bindings.get(obj, {o})):
                        if (s, p, o) not in results:
                            results.append((s, p, o))
        else:
            results = pattern_triples[0][1]
        
        logger.info(f"Query '{query}' returned {len(results)} results.")
        return results

    def ontology_alignment(self, entity_type_a: str, entity_type_b: str) -> bool:
        """Check if two entity types are compatible (same or hierarchically related)."""
        if entity_type_a == entity_type_b:
            return True
        
        current = entity_type_a
        while current in self.ontologies and self.ontologies[current]['parent']:
            if self.ontologies[current]['parent'] == entity_type_b:
                return True
            current = self.ontologies[current]['parent']
        
        current = entity_type_b
        while current in self.ontologies and self.ontologies[current]['parent']:
            if self.ontologies[current]['parent'] == entity_type_a:
                return True
            current = self.ontologies[current]['parent']
        
        return False

    def get_entity_attributes(self, entity_id: str) -> Dict[str, Any]:
        """Retrieve attributes of an entity."""
        attrs = self.graph.nodes.get(entity_id, {})
        if not attrs:
            logger.warning(f"Entity {entity_id} not found in graph.")
        return attrs

    def get_relations(self, entity_id: str) -> List[Tuple[str, str, str]]:
        """Get all relations involving an entity (as subject or object)."""
        if entity_id not in self.graph:
            logger.warning(f"Entity {entity_id} not found in graph.")
            return []
        
        relations = []
        for _, target_id, data in self.graph.out_edges(entity_id, data=True):
            relations.append((entity_id, data['predicate'], target_id))
        for source_id, _, data in self.graph.in_edges(entity_id, data=True):
            relations.append((source_id, data['predicate'], entity_id))
        logger.info(f"Retrieved {len(relations)} relations for entity {entity_id}.")
        return relations

    def _migrate_schema(self, data: Dict, loaded_version: str) -> None:
        """Migrate data from an older schema version to the current one."""
        if loaded_version == '0.0':
            for node in data['graph']['nodes']:
                if 'type' not in node['attributes']:
                    node['attributes']['type'] = 'Unknown'
        if loaded_version in ['0.0', '1.0', '1.1']:
            data['predicate_index'] = {}
            for u, v, d in data['graph']['edges']:
                predicate = d['predicate']
                data['predicate_index'].setdefault(predicate, []).append([u, v])
            data['predicate_index'] = {k: list(set(tuple(edge) for edge in edges)) 
                                     for k, edges in data['predicate_index'].items()}
            data['entity_type_index'] = data.get('entity_type_index', {})
            for node in data['graph']['nodes']:
                entity_id = node['id']
                entity_type = node['attributes'].get('type', 'Unknown')
                if entity_type not in data['entity_type_index']:
                    data['entity_type_index'][entity_type] = []
                data['entity_type_index'][entity_type].append(entity_id)
            data['entity_type_index'] = {k: list(set(v)) for k, v in data['entity_type_index'].items()}
            for k, v in data['temporal_data'].items():
                for t in ['start_time', 'end_time']:
                    if t in v and isinstance(v[t], str):
                        try:
                            v[t] = datetime.fromisoformat(v[t])
                        except ValueError:
                            logger.warning(f"Invalid datetime in temporal_data for edge {k}, key {t}. Setting to None.")
                            v[t] = None
        logger.info(f"Migrated schema from {loaded_version} to {self.schema_version}.")

    def save_graph(self, filename: str) -> None:
        """Save the knowledge graph to a JSON file."""
        graph_data = nx.node_link_data(self.graph)
        with open(filename, 'w') as f:
            json.dump({
                'schema_version': self.schema_version,
                'graph': graph_data,
                'ontologies': self.ontologies,
                'temporal_data': {k: {kk: vv.isoformat() if isinstance(vv, datetime) else vv 
                                    for kk, vv in v.items()} 
                                 for k, v in self.temporal_data.items()},
                'relation_types': self.relation_types,
                'entity_type_index': {k: list(v) for k, v in self.entity_type_index.items()},
                'predicate_index': {k: list(v) for k, v in self.predicate_index.items()}
            }, f, indent=2)
        logger.info(f"Saved graph to {filename}.")

    def load_graph(self, filename: str) -> None:
        """Load the knowledge graph from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            loaded_version = data.get('schema_version', '0.0')
            if loaded_version != self.schema_version:
                logger.warning(f"Schema version mismatch: expected {self.schema_version}, got {loaded_version}.")
                self._migrate_schema(data, loaded_version)
            
            self.graph = nx.node_link_graph(data['graph'])
            self.ontologies = data['ontologies']
            self.temporal_data = {k: {kk: datetime.fromisoformat(vv) if kk in ['start_time', 'end_time'] else vv 
                                     for kk, vv in v.items()} 
                                 for k, v in data['temporal_data'].items()}
            self.relation_types = data['relation_types']
            self.entity_type_index = {k: set(v) for k, v in data['entity_type_index'].items()}
            self.predicate_index = {k: set(tuple(edge) for edge in v) for k, v in data.get('predicate_index', {}).items()}
            
            for entity_type in self.ontologies:
                if entity_type not in self.entity_type_index:
                    self.entity_type_index[entity_type] = set()
            for entity_id in self.graph.nodes:
                entity_type = self.graph.nodes[entity_id].get('type')
                if entity_type:
                    try:
                        self._validate_entity_attributes(entity_id)
                        if entity_type in self.ontologies:
                            self.entity_type_index[entity_type].add(entity_id)
                        else:
                            logger.warning(f"Entity {entity_id} has undefined type {entity_type}.")
                    except ValueError as e:
                        logger.error(f"Validation failed for entity {entity_id}: {e}")
                        raise
            self.temporal_data = {k: v for k, v in self.temporal_data.items() 
                                 if any(d['edge_id'] == k for _, _, d in self.graph.edges(data=True))}
            for k, v in self.temporal_data.items():
                start_time = v.get('start_time')
                end_time = v.get('end_time')
                if start_time and end_time and start_time > end_time:
                    logger.error(f"Invalid temporal range in edge {k}: start_time > end_time.")
                    raise ValueError(f"Invalid temporal range in edge {k}.")
            for predicate in self.relation_types:
                if predicate not in self.predicate_index and predicate in [d['predicate'] for _, _, d in self.graph.edges(data=True)]:
                    self.predicate_index[predicate] = set((u, v) for u, v, d in self.graph.edges(data=True) if d['predicate'] == predicate)
                if self.relation_types[predicate]['subject_type'] not in self.ontologies or \
                   self.relation_types[predicate]['object_type'] not in self.ontologies:
                    logger.error(f"Invalid relation type {predicate}: subject or object type not in ontologies.")
                    raise ValueError(f"Invalid relation type {predicate}.")
            self.path_reasoning.cache_clear()
            self.query_triples.cache_clear()
            logger.info(f"Loaded graph from {filename}.")
        except FileNotFoundError:
            logger.error(f"File {filename} not found.")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {filename}.")
            raise
        except KeyError as e:
            logger.error(f"Missing key {e} in JSON data.")
            raise

# Example usage
if __name__ == "__main__":
    kc = KnowledgeCore()
    
    # Define ontologies with type constraints
    kc.define_ontology("Person", attributes={"name": str, "age": int})
    kc.define_ontology("Employee", parent_type="Person", attributes={"name": str, "age": int, "role": str})
    kc.define_ontology("Organization", attributes={"name": str})
    
    # Define relation types
    kc.define_relation_type("works_for", subject_type="Employee", object_type="Organization")
    kc.define_relation_type("manages", subject_type="Person", object_type="Employee")
    
    # Add entities with validation
    try:
        kc.add_entities([
            ("alice", "Employee", {"name": "Alice Smith", "age": 30, "role": "Engineer"}),
            ("bob", "Person", {"name": "Bob Johnson", "age": 40}),
            ("company", "Organization", {"name": "Tech Corp"})
        ])
    except ValueError as e:
        print(f"Error: {e}")
    
    # Add relations with validation
    try:
        kc.add_relations([
            ("alice", "works_for", "company", datetime(2023, 1, 1), datetime(2024, 12, 31), 0.9),
            ("bob", "manages", "alice", datetime(2023, 6, 1), datetime(2024, 6, 1), 0.8)
        ])
    except ValueError as e:
        print(f"Error: {e}")
    
    # Perform graph search
    paths = kc.graph_search("bob", "company", max_depth=5, max_paths=10)
    print("Paths from Bob to Company:", paths)
    
    # Perform path reasoning with exponential model
    weighted_paths = kc.path_reasoning("bob", "company", max_depth=5, confidence_model="exponential", decay_factor=0.7, min_confidence=0.1)
    print("Weighted paths from Bob to Company:", weighted_paths)
    
    # Hybrid query with temporal range
    probability = kc.hybrid_query("bob", "company", temporal_start=datetime(2023, 1, 1), temporal_end=datetime(2024, 12, 31), confidence_model="exponential")
    print("Probability of connection (2023-2024):", probability)
    
    # Query triples with temporal constraints
    triples = kc.query_triples(predicate="manages", temporal_start=datetime(2023, 1, 1), temporal_end=datetime(2024, 12, 31))
    print("Manages relations:", triples)
    
    # Parse query with per-pattern temporal constraints
    try:
        query_results = kc.parse_query(
            "?x manages ?y {2023-06-01T00:00:00,2024-06-01T00:00:00} . ?y works_for company {2023-01-01T00:00:00,2024-12-31T00:00:00}",
            min_confidence=0.1
        )
        print("Query results:", query_results)
    except ValueError as e:
        print(f"Query error: {e}")
    
    # Ontology alignment
    is_aligned = kc.ontology_alignment("Employee", "Person")
    print("Is Employee aligned with Person?", is_aligned)
    
    # Get entity attributes
    alice_attrs = kc.get_entity_attributes("alice")
    print("Alice's attributes:", alice_attrs)
    
    # Get relations
    alice_relations = kc.get_relations("alice")
    print("Alice's relations:", alice_relations)
    
    # Remove entities in batch
    kc.remove_entities(["alice", "bob"])
    
    # Save and load graph
    kc.save_graph("knowledge_graph.json")
    kc_new = KnowledgeCore()
    kc_new.load_graph("knowledge_graph.json")
    print("Loaded graph attributes for Alice:", kc_new.get_entity_attributes("alice"))


# ### Potential Future Enhancements
# - **Confidence-Based Triple Filtering**: Integrate `min_confidence` into `parse_query` by computing path confidences for each triple using `path_reasoning`.
# - **Custom Cache for Queries**: Replace `@lru_cache` on `query_triples` with a custom cache that invalidates only affected triples based on modified entities or predicates.
# - **Extended Temporal Formats**: Support more datetime formats (e.g., relative dates like `now-1y`) in `parse_query`.
# - **Query Optimization**: Use `nx.shortest_path` in `hybrid_query` for single-path queries with high confidence.
