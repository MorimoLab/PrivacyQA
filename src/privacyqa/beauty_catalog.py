"""The Beauty catalog: the attributes of every Amazon Reviews 2023 Beauty_and_Personal_Care product, cleaned by fixed rules.

Every step reads product metadata only; ratings and users never enter. The rules are in configs/beauty_rules.json.

1. Extraction: the category path and the detail fields (brand, item form, material, hair type, skin type, scent) of
   every product, after Unicode, whitespace and case normalization.
2. Cleaning of every attribute value. Placeholders are excluded. Values that match a suspicious pattern, mismatch their
   field or cannot be fully interpreted are held back (pending). Finite lists, applicability phrases and form-quantity
   phrases are parsed into known literals, and reviewed literal aliases are applied.
3. Splitting: a list value of a split field becomes one attribute per listed member.
4. Merging of reviewed whole-attribute aliases.
5. An attribute enters the catalog when at least `minimum_catalog_products` products carry it. Universal applicability
   (`hair_type::all`, `skin_type::all`) is a catalog constraint only, never part of a preference core. Aspects listed
   in `excluded_aspects` (brand) are not extracted, so their rules stay in the file without effect.
"""
from collections import Counter
import gzip
import json
import re
import unicodedata

import pyarrow as pa

from privacyqa.common import progress, read_json, require, write_json, write_parquet

FIELDS = ('brand', 'category', 'hair_type', 'item_form', 'material', 'scent', 'skin_type')
DETAILS = {'brand': 'Brand', 'item_form': 'Item Form', 'material': 'Material', 'hair_type': 'Hair Type',
           'skin_type': 'Skin Type', 'scent': 'Scent'}
MISSING = {'', 'none', 'null', 'n/a', 'unknown', 'not available'}
APPLICABILITY = ('skin_type', 'hair_type')
UNIVERSAL = frozenset(('hair_type::all', 'skin_type::all'))
LIST_FIELDS = ('hair_type', 'item_form', 'material', 'scent', 'skin_type')
RECOVERABLE = {'unparsed_compound', 'universal_with_other_descriptors', 'size_or_pack_description', 'unparsed_applicability_claim'}
LABELS = {'hair_type': ('hair type', 'hair types'), 'skin_type': ('skin type', 'skin types'), 'item_form': ('item form', 'form'),
          'material': ('material', 'materials'), 'scent': ('scent', 'fragrance')}
REVIEWED_FIELDS = ('brand', 'hair_type', 'item_form', 'material', 'scent', 'skin_type')
REVIEW_BASES = {'field_suffix', 'material_abbreviation', 'multilingual_literal', 'literal_typography', 'brand_spelling', 'brand_display_wrapper'}
RULE_KEYS = {'schema_version', 'kb_version', 'excluded_aspects', 'minimum_catalog_products', 'placeholders', 'aliases', 'enumerations',
             'compound_fields', 'pending_patterns', 'numeric_pending_fields', 'max_value_characters', 'max_value_words',
             'compound_vocabularies', 'quantity_units', 'brand_character_replacements', 'brand_display_marks', 'screening',
             'reviewed_literals', 'split_fields', 'attribute_aliases'}
SCHEMA = pa.schema([('parent_asin', pa.string()), ('attributes', pa.list_(pa.string()))])


def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', value).split()).casefold()


def extract(row, aspects):
    """Attributes of one metadata record: the category path (JSON list) and the declared detail fields."""
    attributes = []
    categories = row.get('categories')
    if 'category' in aspects and isinstance(categories, list) and categories and all(isinstance(x, str) and normalize(x) for x in categories):
        attributes.append('category::' + json.dumps([normalize(x) for x in categories], ensure_ascii=False, separators=(',', ':')))
    details = row.get('details') or {}
    require(isinstance(details, dict), 'invalid_metadata', 'Product details are a JSON object')
    for aspect, field in DETAILS.items():
        value = details.get(field)
        if aspect in aspects and isinstance(value, str) and normalize(value) not in MISSING:
            attributes.append(aspect + '::' + normalize(value))
    return sorted(set(attributes))


# ---------------------------------------------------------------- cleaning of one attribute

def classify(atom, rules):
    """Normalize a value; exclude placeholders; hold back suspicious values; apply field aliases and sort enumerations."""
    require(isinstance(atom, str) and '::' in atom, 'invalid_beauty_atom', 'Expected a field::value attribute')
    field, raw = atom.split('::', 1)
    require(field in FIELDS, 'unknown_beauty_field', 'Only the seven metadata fields are accepted')
    reasons, components = [], []
    value, action, universal = normalize(raw), 'keep', False
    if value != raw:
        reasons.append('unicode_case_whitespace')
    if field == 'category':
        try:
            parts = json.loads(raw)
        except ValueError:
            parts = None
        if not (isinstance(parts, list) and parts and all(isinstance(p, str) and normalize(p) for p in parts)):
            action, reasons = 'pending', ['malformed_category_path']
        else:
            value = json.dumps([normalize(p) for p in parts], ensure_ascii=False, separators=(',', ':'))
    elif value in rules['placeholders'] or not any(c.isalnum() for c in value):
        action, reasons = 'exclude', ['empty_placeholder_or_punctuation']
    else:
        for pattern in rules['pending_patterns']:
            if field in pattern['fields'] and re.search(pattern['regex'], value):
                reasons.append(pattern['reason'])
        if len(value) > rules['max_value_characters'] or len(value.split()) > rules['max_value_words']:
            reasons.append('long_description')
        if field in rules['numeric_pending_fields'] and re.fullmatch(r'[\d\s.,+%/-]+', value):
            reasons.append('numeric_or_specification_only')
        if any(r != 'unicode_case_whitespace' for r in reasons):
            action = 'pending'
        else:
            aliases = rules['aliases'][field]
            if field in rules['compound_fields'] and re.search('[,;]', value):
                tokens = [normalize(t) for t in re.split('[,;]', value)]
                mapped = [aliases.get(t, t) for t in tokens]
                if field not in rules['enumerations'] or not all(t in rules['enumerations'][field] for t in mapped):
                    action = 'pending'
                    reasons.append('unparsed_compound')
                elif 'all' in mapped and set(mapped) != {'all'}:
                    action = 'pending'
                    reasons.append('universal_with_other_descriptors')
                else:
                    components = sorted(set(mapped))
                    value = ', '.join(components)
                    reasons.append('enumeration_sorted_deduplicated')
            elif value in aliases:
                value = aliases[value]
                reasons.append('explicit_field_alias')
            universal = field in APPLICABILITY and value == 'all' and action != 'pending'
    canonical = field + '::' + value if action not in ('pending', 'exclude') else None
    if canonical is not None and canonical != atom:
        action = 'merge'
    return dict(raw_atom=atom, aspect=field, canonical_atom=canonical, normalization_action=action, action=action,
                reasons=reasons, components=components, universal_applicability=universal)


def split_list(value, extra_delimiters=''):
    """Split at top-level delimiters only; punctuation inside brackets never splits."""
    pairs, stack, parts, start = {')': '(', ']': '[', '}': '{'}, [], [], 0
    for match in re.finditer(r'[()\[\]{}]|[,;/&' + re.escape(extra_delimiters) + r']|\band\b', value):
        token = match.group()
        if token in '([{':
            stack.append(token)
        elif token in pairs:
            if not stack or stack.pop() != pairs[token]:
                return [], 'unbalanced_brackets'
        elif not stack:
            parts.append(value[start:match.start()].strip())
            start = match.end()
    if stack:
        return [], 'unbalanced_brackets'
    parts.append(value[start:].strip())
    return parts, None


def applicability_body(field, value):
    noun = 'skin' if field == 'skin_type' else 'hair'
    # Strip only this complete phrase grammar; an unknown remainder is never accepted.
    body = re.sub(rf'^{noun} types?\s*:\s*', '', value)
    body = re.sub(r'^(?:suitable for|for)\s+', '', body)
    body = body.removesuffix('.').strip()
    for suffix in (f' types of {noun}', f' type of {noun}', f' {noun} types', f' {noun} type', f' {noun}'):
        if body.endswith(suffix):
            body = body[:-len(suffix)].strip()
            break
    return body


def list_member(field, value, rules):
    if field in APPLICABILITY:
        value = applicability_body(field, value)
    return rules['aliases'][field].get(value, value)


def extract_form_quantity(value, rules):
    units = '|'.join(re.escape(u) for u in sorted(rules['quantity_units'], key=len, reverse=True))
    quantity = rf'(?P<spec>(?P<number>\d+(?:\.\d+)?)\s*(?:{units}))'
    patterns = (rf'{quantity}\s+(?P<form>.+)', rf'(?P<form>.+?)\s+{quantity}',
                rf'(?P<form>.+?)\s*\({quantity}\)', rf'\({quantity}\)\s*(?P<form>.+)')
    for pattern in patterns:
        match = re.fullmatch(pattern, value)
        if match and float(match['number']) > 0:
            form = rules['aliases']['item_form'].get(match['form'], match['form'])
            if form in rules['compound_vocabularies']['item_form']:
                return form, [match['spec']]
    return value, []


def pending(base, reason):
    return dict(base, canonical_atom=None, normalization_action='pending', action='pending', universal_applicability=False,
                components=[], reasons=list(dict.fromkeys([*base['reasons'], reason])))


def parse(atom, rules):
    """Brand typography, and finite lists, applicability phrases and form-quantity phrases of known literals."""
    base = classify(atom, rules)
    base['parsed_specifications'] = []
    field, raw = atom.split('::', 1)
    if field == 'category' or base['normalization_action'] == 'exclude':
        return base
    if field == 'brand':
        value = raw.translate(str.maketrans(rules['brand_character_replacements']))
        # Before NFKC: a genuine trademark sign is typography, while the letters "tm" are kept.
        value = re.sub(r'[®™](?=\s|$)', '', value)
        result = classify(field + '::' + value, rules)
        result['parsed_specifications'] = []
        result['raw_atom'] = atom
        if value != raw:
            result['reasons'].append('brand_typography')
        if result['canonical_atom'] is not None:
            result['action'] = result['normalization_action'] = 'keep' if result['canonical_atom'] == atom else 'merge'
        return result
    if base['action'] == 'pending' and any(r not in RECOVERABLE | {'unicode_case_whitespace'} for r in base['reasons']):
        return base
    value = normalize(raw)
    specifications = []
    if field == 'item_form':
        value, specifications = extract_form_quantity(value, rules)
    if 'size_or_pack_description' in base['reasons'] and not specifications:
        return base
    body = applicability_body(field, value) if field in APPLICABILITY else value
    parts, problem = split_list(body)
    if problem:
        return pending(base, problem)
    members = [list_member(field, part, rules) for part in parts]
    known = rules['enumerations'][field] if field in APPLICABILITY else rules['compound_vocabularies'][field]
    if not all(member in known for member in members):
        if len(parts) > 1:
            return pending(base, 'unparsed_compound')
        return base
    if field in APPLICABILITY and 'all' in members and set(members) != {'all'}:
        return pending(base, 'universal_with_other_descriptors')
    if field == 'scent' and {'unscented', 'fragrance-free'} & set(members) and len(set(members)) > 1:
        return pending(base, 'conflicting_scent_descriptors')
    # Fragrance order and repeated notes stay meaningful; other explicit lists are sets.
    components = members if field == 'scent' else sorted(set(members))
    canonical = field + '::' + ', '.join(components)
    reasons = [r for r in base['reasons'] if r not in RECOVERABLE]
    if field in APPLICABILITY and body != value:
        reasons.append('full_applicability_phrase')
    if any(a != b for a, b in zip(parts, members)):
        reasons.append('explicit_member_normalization')
    if len(parts) > 1:
        reasons.append('ordered_scent_list' if field == 'scent' else 'finite_list_sorted_deduplicated')
    if specifications:
        reasons.append('exact_form_quantity_extracted')
    action = 'keep' if canonical == atom else 'merge'
    return dict(raw_atom=atom, aspect=field, canonical_atom=canonical, normalization_action=action, action=action,
                reasons=list(dict.fromkeys(reasons)), components=components if len(parts) > 1 else [],
                universal_applicability=field in APPLICABILITY and components == ['all'], parsed_specifications=specifications)


def vocabulary(field, rules):
    listed = rules['enumerations'][field] if field in APPLICABILITY else rules['compound_vocabularies'][field]
    return set(listed) | set(rules['screening']['literal_vocabularies'][field])


def unwrapped_literal(field, value):
    body = value
    for label in LABELS[field]:
        if body.startswith(label + ':'):
            body = body[len(label) + 1:].strip()
            break
    body = body.removesuffix('.').strip()
    if len(body) > 1 and (body[0], body[-1]) in (('"', '"'), ("'", "'"), ('“', '”'), ('(', ')')):
        body = body[1:-1].strip()
    return body


def screening_reasons(field, value, rules):
    screening = rules['screening']
    spellings = {value, unwrapped_literal(field, value)}
    reasons = []
    if spellings & set(screening['placeholders']):
        reasons.append('screening_metadata_placeholder')
    if spellings & set(screening['ambiguous_values'][field]):
        reasons.append('screening_underspecified_value')
    if spellings & set(screening['field_mismatches'][field]):
        reasons.append('screening_field_mismatch')
    for pattern in screening['pending_patterns']:
        if field in pattern['fields'] and re.search(pattern['regex'], value):
            reasons.append(pattern['reason'])
    return reasons


def known_literal(field, value, rules):
    """The known literal of a value without its wrappers, only when the whole remainder is recognized."""
    aliases = {**rules['aliases'][field], **rules['screening']['orthographic_aliases'][field]}
    known = vocabulary(field, rules)
    body = unwrapped_literal(field, value)
    direct = aliases.get(body, body)
    if direct in known:
        return direct
    if field in APPLICABILITY:
        body = applicability_body(field, body)
    if field == 'scent':
        for suffix in ('-scented', ' scented', ' fragrance', ' scent'):
            if body.endswith(suffix):
                body = body[:-len(suffix)].strip()
                break
    # Hyphens and underscores change only when the result is a known literal.
    for spelling in (body, re.sub(r'[-_]+', ' ', body)):
        if field in APPLICABILITY:
            spelling = applicability_body(field, spelling)
        target = aliases.get(spelling, spelling)
        if target in known:
            return target
    return None


def screen(atom, rules):
    """Hold back placeholders, underspecified values and field mismatches; map wrapped and spelled-out known literals."""
    result = parse(atom, rules)
    field = result['aspect']
    if field not in LIST_FIELDS or result['canonical_atom'] is None:
        return result
    value = result['canonical_atom'].split('::', 1)[1]
    reasons = screening_reasons(field, value, rules)
    if reasons:
        for reason in reasons:
            result = pending(result, reason)
        return result
    target = known_literal(field, value, rules)
    components = []
    if target is None:
        parts, problem = split_list(value, '+|^')
        if problem or len(parts) <= 1:
            return result
        members = [known_literal(field, part, rules) for part in parts]
        if any(member is None for member in members):
            return result
        if field in APPLICABILITY and 'all' in members and set(members) != {'all'}:
            return pending(result, 'universal_with_other_descriptors')
        if field == 'scent' and {'unscented', 'fragrance-free'} & set(members) and len(set(members)) > 1:
            return pending(result, 'conflicting_scent_descriptors')
        components = members if field == 'scent' else sorted(set(members))
        target = ', '.join(components)
    canonical = field + '::' + target
    if canonical == result['canonical_atom']:
        return result
    # The new literal must be stable under parsing; otherwise the parsed value is kept.
    if parse(canonical, rules)['canonical_atom'] != canonical:
        return result
    new_reasons = screening_reasons(field, target, rules)
    if new_reasons:
        for reason in new_reasons:
            result = pending(result, reason)
        return result
    action = 'keep' if canonical == atom else 'merge'
    return dict(result, canonical_atom=canonical, action=action, normalization_action=action,
                reasons=list(dict.fromkeys([*result['reasons'], 'screening_exact_literal_normalization'])),
                components=components, universal_applicability=field in APPLICABILITY and target == 'all')


def format_literal(value, reviewed):
    value = value.translate({ord(c): None for c in reviewed['format_characters']})
    # Zero-width spaces inside a name may separate words; only trim them at the edges.
    return normalize(normalize(value).strip(''.join(reviewed['edge_format_characters'])))


def reviewed_pending_reasons(field, value, reviewed):
    reasons = []
    if value in reviewed['pending_values'][field]:
        reasons.append('reviewed_brand_placeholder_or_unresolved_identity' if field == 'brand' else 'reviewed_invalid_or_ambiguous_field_literal')
    for pattern in reviewed['pending_patterns']:
        if field in pattern['fields'] and re.search(pattern['regex'], value):
            reasons.append(pattern['reason'])
    return reasons


def clean_atom(atom, rules):
    """The cleaned attribute of a raw attribute: remove invisible display characters and apply reviewed literal aliases."""
    result = screen(atom, rules)
    field = result['aspect']
    if field == 'category' or result['canonical_atom'] is None:
        return result
    reviewed = rules['reviewed_literals']
    value = result['canonical_atom'].split('::', 1)[1]
    formatted = format_literal(value, reviewed)
    if formatted != value:
        checked = screen(field + '::' + formatted, rules)
        if checked['canonical_atom'] is None:
            return pending(result, 'reviewed_invalid_after_display_cleanup')
        result = dict(checked, raw_atom=atom, reasons=list(dict.fromkeys([*result['reasons'], *checked['reasons'], 'reviewed_invisible_display_characters'])))
        value = result['canonical_atom'].split('::', 1)[1]
    reasons = reviewed_pending_reasons(field, value, reviewed)
    if reasons:
        for reason in reasons:
            result = pending(result, reason)
        return result
    alias = reviewed['aliases'][field].get(value)
    if alias:
        value = alias['target']
        result = dict(result, components=[], reasons=list(dict.fromkeys([*result['reasons'], 'reviewed_' + alias['basis']])))
    canonical = field + '::' + value
    action = 'keep' if canonical == atom else 'merge'
    return dict(result, canonical_atom=canonical, action=action, normalization_action=action,
                universal_applicability=field in APPLICABILITY and value == 'all')


def split_targets(atom, rules):
    """The member attributes of a cleaned list value of a split field, or the attribute itself."""
    record = clean_atom(atom, rules)
    require(record['canonical_atom'] == atom, 'unstable_cleaned_attribute', 'Cleaned attributes are stable under cleaning')
    components = record['components'] if record['aspect'] in rules['split_fields'] else []
    targets = sorted({record['aspect'] + '::' + value for value in components}) if len(components) > 1 else [atom]
    require(all(clean_atom(target, rules)['canonical_atom'] == target for target in targets), 'unstable_list_member',
            'Every list member is itself a cleaned attribute')
    return targets


# ---------------------------------------------------------------- rules

def literal_list(values, empty=False):
    """Unique normalized literals; the empty literal only where `empty` allows it."""
    return (isinstance(values, list) and len(values) == len(set(values))
            and all(isinstance(v, str) and (empty or v) and normalize(v) == v for v in values))


def validate_rules(rules):
    require(isinstance(rules, dict) and set(rules) == RULE_KEYS, 'invalid_beauty_rules', 'Declare every group of the Beauty rules')
    require(set(rules['excluded_aspects']) <= set(FIELDS) and rules['split_fields'] == list(LIST_FIELDS)
            and type(rules['minimum_catalog_products']) is int and rules['minimum_catalog_products'] >= 2
            and all(type(rules[k]) is int and rules[k] > 0 for k in ('max_value_characters', 'max_value_words')),
            'invalid_beauty_rules', 'Declare the excluded aspects, the split fields and a catalog minimum of at least two products')
    require(set(rules['aliases']) == set(FIELDS) and rules['aliases']['category'] == {} and set(rules['enumerations']) == set(APPLICABILITY)
            and rules['compound_fields'] == list(LIST_FIELDS) and set(rules['numeric_pending_fields']) <= set(rules['compound_fields']),
            'invalid_beauty_rules', 'Category paths are kept whole; only declared fields are parsed as lists')
    require(literal_list(rules['placeholders'], empty=True) and all('all' in values for values in rules['enumerations'].values()),
            'invalid_beauty_rules', 'Placeholders are normalized literals; applicability enumerations include all')
    for field, aliases in rules['aliases'].items():
        require(isinstance(aliases, dict) and all(isinstance(a, str) and isinstance(b, str) and a and b and normalize(a) == a and normalize(b) == b
                                                  and a != b and b not in aliases and '::' not in a + b and not any(c in a + b for c in ',;')
                                                  for a, b in aliases.items()),
                'invalid_beauty_rules', 'Field aliases are direct mappings, never chains or list rewrites')
    for pattern in rules['pending_patterns'] + rules['screening']['pending_patterns'] + rules['reviewed_literals']['pending_patterns']:
        require(set(pattern) == {'reason', 'fields', 'regex'} and pattern['reason'] and set(pattern['fields']) <= set(FIELDS) - {'category'},
                'invalid_beauty_rules', 'Patterns name their reason and fields')
        re.compile(pattern['regex'])
    require(set(rules['compound_vocabularies']) == {'item_form', 'material', 'scent'}, 'invalid_beauty_rules', 'Declare the list vocabularies')
    for field, values in {**rules['enumerations'], **rules['compound_vocabularies']}.items():
        require(literal_list(values) and set(rules['aliases'][field].values()) <= set(values), 'invalid_beauty_rules',
                'List members and alias targets belong to the declared vocabulary')
    require(literal_list(rules['quantity_units']) and rules['brand_display_marks'] == ['®', '™']
            and rules['brand_character_replacements'] == {'‘': "'", '’': "'", 'ʼ': "'"},
            'invalid_beauty_rules', 'Only declared quantity units and brand typography are normalized')
    screening = rules['screening']
    require(set(screening) == {'placeholders', 'ambiguous_values', 'field_mismatches', 'pending_patterns', 'orthographic_aliases', 'literal_vocabularies'}
            and all(set(screening[k]) == set(LIST_FIELDS) for k in ('ambiguous_values', 'field_mismatches', 'orthographic_aliases', 'literal_vocabularies'))
            and all(literal_list(v) for v in [screening['placeholders'], *screening['ambiguous_values'].values(),
                                              *screening['field_mismatches'].values(), *screening['literal_vocabularies'].values()]),
            'invalid_beauty_rules', 'Screening lists are unique normalized literals of the list fields')
    for field, aliases in screening['orthographic_aliases'].items():
        blocked = set(screening['placeholders'] + screening['ambiguous_values'][field] + screening['field_mismatches'][field])
        require(all(a and b and a == normalize(a) and b == normalize(b) and a != b and b not in aliases and '::' not in a + b
                    and b in vocabulary(field, rules) for a, b in aliases.items())
                and not blocked & (vocabulary(field, rules) | set(aliases)),
                'invalid_beauty_rules', 'Spelling aliases land on known literals, which are never screened out')
    reviewed = rules['reviewed_literals']
    require(set(reviewed) == {'aliases', 'pending_values', 'format_characters', 'edge_format_characters', 'pending_patterns'}
            and reviewed['format_characters'] == ['‎', '‏', '﻿'] and reviewed['edge_format_characters'] == ['​']
            and set(reviewed['aliases']) == set(reviewed['pending_values']) == set(REVIEWED_FIELDS),
            'invalid_beauty_rules', 'Reviewed literals declare aliases, pending values and the invisible characters to remove')
    for field in REVIEWED_FIELDS:
        aliases = reviewed['aliases'][field]
        require(literal_list(reviewed['pending_values'][field]), 'invalid_beauty_rules', 'Pending values are unique normalized literals')
        for source, entry in aliases.items():
            target = entry['target']
            require(set(entry) == {'target', 'basis'} and entry['basis'] in REVIEW_BASES and source and target and source != target
                    and target not in aliases and (entry['basis'].startswith('brand_')) == (field == 'brand')
                    and (entry['basis'] != 'field_suffix' or field in APPLICABILITY)
                    and (entry['basis'] != 'material_abbreviation' or field == 'material')
                    and all(normalize(v) == v and '::' not in v and format_literal(v, reviewed) == v
                            and screen(field + '::' + v, rules)['canonical_atom'] == field + '::' + v
                            and not reviewed_pending_reasons(field, v, reviewed) for v in (source, target)),
                    'invalid_beauty_rules', 'Reviewed aliases map whole accepted literals with a review basis of their field')
    sources, targets = set(), set()
    for entry in rules['attribute_aliases']:
        pair = [entry['source_atom'], entry['target_atom']]
        require(set(entry) == {'source_atom', 'target_atom', 'basis', 'evidence_urls', 'notes'}
                and all(isinstance(a, str) and a.count('::') == 1 and a.split('::')[1] and normalize(a) == a for a in pair)
                and pair[0].split('::')[0] in FIELDS and pair[0].split('::')[0] == pair[1].split('::')[0] and pair[0] != pair[1]
                and pair[0] not in sources and pair[0] not in UNIVERSAL
                and (pair[1] not in UNIVERSAL or entry['basis'] == 'reviewed_universal_applicability')
                and all(isinstance(entry[k], str) and entry[k].strip() for k in ('basis', 'notes'))
                and all(isinstance(u, str) and u.startswith('https://') for u in entry['evidence_urls']),
                'invalid_beauty_rules', 'Attribute aliases are reviewed, unique, within one field and never narrow universal applicability')
        if pair[0].startswith('category::'):
            paths = [json.loads(a.split('::', 1)[1]) for a in pair]
            require(all(isinstance(p, list) and p and all(isinstance(v, str) and v and normalize(v) == v for v in p) for p in paths)
                    and len(paths[0]) == len(paths[1]), 'invalid_beauty_rules', 'Category aliases keep complete paths of equal depth')
        sources.add(pair[0])
        targets.add(pair[1])
    require(not sources & targets, 'invalid_beauty_rules', 'Attribute aliases are direct, without chains')


# ---------------------------------------------------------------- stage

def build_beauty_catalog(config, output):
    spec = config['domains']['beauty']
    rules = read_json(config.file(spec['rules']))
    validate_rules(rules)
    aspects = [f for f in FIELDS if f not in rules['excluded_aspects']]
    products = {}
    with gzip.open(config.file(spec['metadata']), 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            item = row.get('parent_asin')
            require(isinstance(item, str) and item.strip(), 'invalid_metadata', 'Every product has a parent_asin')
            attributes = extract(row, aspects)
            require(products.get(item, attributes) == attributes, 'conflicting_metadata', 'A product has one set of attributes')
            products[item] = attributes
    progress(output, 'metadata_extracted', products=len(products))
    mapping = {atom: clean_atom(atom, rules) for atom in sorted({a for attributes in products.values() for a in attributes})}
    progress(output, 'attributes_cleaned', attributes=len(mapping))
    aliases = {r['source_atom']: r['target_atom'] for r in rules['attribute_aliases'] if r['source_atom'].split('::', 1)[0] in aspects}
    members, cleaned_counts, counts = {}, Counter(), Counter()
    final = {}
    for item in sorted(products):
        cleaned = {mapping[a]['canonical_atom'] for a in products[item]} - {None}
        split = set()
        for atom in cleaned:
            if atom.split('::', 1)[0] in rules['split_fields']:
                if atom not in members:
                    members[atom] = split_targets(atom, rules)
                split.update(members[atom])
            else:
                split.add(atom)
        cleaned_counts.update(split)
        final[item] = {aliases.get(a, a) for a in split}
        counts.update(final[item])
    minimum = rules['minimum_catalog_products']
    active = {a for a, n in counts.items() if n >= minimum}
    evidence = active - UNIVERSAL
    write_parquet(output / 'catalog.parquet', (dict(parent_asin=item, attributes=sorted(final[item] & active)) for item in final), SCHEMA)
    schema = dict(domain='beauty', schema_version=rules['schema_version'], kb_version=rules['kb_version'],
                  evidence_atoms=sorted(evidence), catalog_atoms=sorted(active))
    write_json(output / 'schema.json', schema)
    actions = Counter(r['action'] for r in mapping.values())
    report = dict(status='beauty_catalog_built', products=len(final), extracted_aspects=aspects, extracted_attributes=len(mapping),
                  cleaning_actions=dict(sorted(actions.items())), catalog_directions=len(active), evidence_directions=len(evidence),
                  catalog_directions_by_aspect=dict(sorted(Counter(a.split('::', 1)[0] for a in active).items())),
                  products_without_attributes=sum(not (attributes & active) for attributes in final.values()),
                  attribute_aliases=len(aliases), attribute_aliases_without_source=len(set(aliases) - set(cleaned_counts)), user_data_read=False)
    write_json(output / 'manifest.json', report)
    return report
