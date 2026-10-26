"""Catalogs and complete answer computation; ingredient constraints exist only in the Recipe catalog."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from privacyqa.common import atoms, identifier, iter_rows, read_json, require

ITEM_FIELDS = {'movie': 'movieId', 'beauty': 'parent_asin', 'recipe_text': 'item_id'}


def ingredient_atom(operation, name):
    require(operation in ('include', 'exclude') and isinstance(name, str) and bool(name), 'invalid_ingredient',
            'Use include or exclude and an exact catalog ingredient name')
    return 'ingredient_' + operation + '::' + json.dumps(name, ensure_ascii=False)


def ingredient_predicate(atom):
    prefix, value = atom.split('::', 1)
    if prefix not in ('ingredient_include', 'ingredient_exclude'):
        return None
    try:
        name = json.loads(value)
    except ValueError:
        require(False, 'invalid_ingredient', 'Ingredient values are JSON strings')
    require(isinstance(name, str) and bool(name) and atom == ingredient_atom(prefix.removeprefix('ingredient_'), name),
            'invalid_ingredient', 'Ingredient constraints are canonical')
    return prefix.removeprefix('ingredient_'), name


class Catalog:
    """Items with their attributes (and Recipe ingredients), indexed for exact execution of constraint sets."""

    def __init__(self, rows, domain, schema_atoms):
        self.domain = domain
        self.schema = frozenset(atoms(list(schema_atoms)))
        self.items, self.tags, self.ingredients = {}, {}, {}
        field = ITEM_FIELDS[domain]
        for row in rows:
            item = identifier(row[field], field)
            attributes = frozenset(atoms(row['attributes']))
            require(attributes <= self.schema, 'unknown_catalog_atom', 'Catalog attributes come from the schema')
            names = row.get('ingredients', [])
            require(isinstance(names, list) and all(isinstance(n, str) and n for n in names), 'invalid_ingredient',
                    'Catalog ingredients are nonempty strings')
            record = attributes, frozenset(names)
            require(item not in self.items or self.items[item] == record, 'catalog_item_conflict', 'One definition per item')
            self.items[item] = record
        for item, (attributes, names) in self.items.items():
            for tag in attributes:
                self.tags.setdefault(tag, set()).add(item)
            for name in names:
                self.ingredients.setdefault(name, set()).add(item)

    @classmethod
    def load(cls, path, domain, schema_atoms):
        return cls(iter_rows(path, 'parquet'), domain, schema_atoms)

    def execute(self, evidence, additions=(), *, max_conditions=10):
        """The complete, sorted answer set of a constraint set."""
        core, additions = atoms(list(evidence)), atoms(list(additions))
        require(len(core) + len(additions) <= max_conditions, 'condition_budget_exhausted', 'Too many conditions')
        require(not set(core) & set(additions), 'constraint_role_overlap', 'Core and auxiliary constraints are distinct')
        require(all(ingredient_predicate(a) is None for a in core), 'ingredient_is_catalog_only',
                'Ingredient constraints are never core constraints')
        postings, exclusions = [], []
        for atom in core + additions:
            predicate = ingredient_predicate(atom)
            if predicate:
                require(self.domain == 'recipe_text', 'invalid_ingredient_domain', 'Only Recipe has ingredients')
                operation, name = predicate
                require(name in self.ingredients, 'unknown_ingredient', 'Ingredients come from the catalog')
                (postings if operation == 'include' else exclusions).append(self.ingredients[name])
            else:
                require(atom in self.schema, 'unmapped_catalog_atom', 'Constraints are executable catalog attributes')
                postings.append(self.tags.get(atom, set()))
        if postings:
            ordered = sorted(postings, key=len)
            result = set(ordered[0])
            for values in ordered[1:]:
                result.intersection_update(values)
        else:
            result = set(self.items)
        for values in exclusions:
            result.difference_update(values)
        return tuple(sorted(result))


def domain_schema(config, domain):
    """Versions and atoms of a domain: evidence atoms may enter cores, catalog atoms are executable."""
    schema = read_json(config.file(config['domains'][domain]['schema']))
    require(set(schema) == {'domain', 'schema_version', 'kb_version', 'evidence_atoms', 'catalog_atoms'}
            and schema['domain'] == domain, 'invalid_schema', 'Use the schema file of the domain')
    return schema


def load_catalog(config, domain):
    return Catalog.load(config.file(config['domains'][domain]['catalog']), domain, domain_schema(config, domain)['catalog_atoms'])


def load_recipe_search(path):
    """The RecipeADQA constraint search (Apache-2.0), loaded from an explicit path."""
    path = Path(path)
    require(path.is_file(), 'missing_recipe_search', 'Provide RecipeADQA code/common/qa/constraint_search.py')
    code_root = path.parents[2]
    if 'common' in sys.modules:
        require(Path(sys.modules['common'].__file__).resolve().is_relative_to(code_root), 'source_module_collision',
                'Another package named common is already imported')
    sys.path.insert(0, str(code_root))
    try:
        spec = importlib.util.spec_from_file_location('_privacyqa_recipe_search', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def extend_recipe(catalog, core, extra_tags, search, max_conditions=10):
    """Narrow a Recipe request to at most 25 answers with the RecipeADQA ingredient beam search."""
    answers = catalog.execute(core, extra_tags, max_conditions=max_conditions)
    names = sorted({name for item in answers for name in catalog.items[item][1]})
    positions = {name: i for i, name in enumerate(names)}
    postings = {}
    for index, item in enumerate(answers):
        for name in catalog.items[item][1]:
            position = positions[name]
            postings[position] = postings.get(position, 0) | (1 << index)
    status, mask, predicates = search.find_representative_split(
        len(answers), postings, names, lower=1, upper=25, target=20,
        max_constraints=max_conditions - len(core) - len(extra_tags),
        min_initial_coverage=0.001, candidate_limit=256, beam_width=24)
    additions = tuple(sorted((*extra_tags, *(ingredient_atom(p['operation'], p['ingredient']) for p in predicates))))
    complete = catalog.execute(core, additions, max_conditions=max_conditions)
    require(complete == tuple(a for i, a in enumerate(answers) if mask & (1 << i)), 'ingredient_search_mismatch',
            'The search result equals the complete execution')
    return status, additions, complete
