"""Fixed pseudorandom assignment of the users of every source to five disjoint roles.

The assignment depends only on the user identifier and a private salt, so nobody without the salt can tell which role a
user has. Construction users (role `protected`) enter the releases; the other roles never do.
"""
from collections import Counter
import hashlib
import json
import math

import pyarrow as pa

from privacyqa.common import DOMAINS, iter_rows, require, write_json, write_parquet
from privacyqa.evidence import index_users

ROLES = ('auxiliary', 'protected', 'attack_reference', 'attack_nonmember', 'utility_reference')
NAMESPACES = {'movie': 'movielens:userId', 'beauty': 'amazon_2023:user_id', 'recipe_text': 'foodcom:AuthorId'}
SCHEMA = pa.schema([('domain', pa.string()), ('user_key', pa.string()), ('role', pa.string())])


def assign_role(domain, user_key, salt, proportions):
    token = json.dumps([salt, domain, NAMESPACES[domain], user_key], separators=(',', ':')).encode()
    value = int.from_bytes(hashlib.sha256(token).digest(), 'big') / 2**256
    cumulative = 0.0
    for role in ROLES:
        cumulative += proportions[role]
        if value < cumulative:
            return role
    raise AssertionError('The role proportions cover the unit interval')


def assign_roles(config, output):
    proportions, salt = config['roles']['proportions'], config['private']['role_salt']
    require(set(proportions) == set(ROLES) and all(v >= 0 for v in proportions.values())
            and math.isclose(sum(proportions.values()), 1.0, abs_tol=1e-12), 'invalid_role_proportions',
            'Give a nonnegative proportion for every role, summing to one')
    require(isinstance(salt, str) and len(salt) >= 16, 'invalid_role_salt', 'Use a private salt of at least 16 characters')
    counts = Counter()

    def rows():
        for domain in DOMAINS:
            for user in index_users(config, domain):
                role = assign_role(domain, user, salt, proportions)
                counts[domain, role] += 1
                yield dict(domain=domain, user_key=user, role=role)

    write_parquet(output / 'roles.parquet', rows(), SCHEMA)
    manifest = dict(status='roles_assigned', proportions=proportions,
                    roles={d: {r: counts[d, r] for r in ROLES} for d in DOMAINS})
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_roles(config, domain):
    """{user: role} of one domain."""
    config.finished('roles', 'roles_assigned')
    return {row['user_key']: row['role'] for row in iter_rows(config.stage('roles') / 'roles.parquet', 'parquet') if row['domain'] == domain}
