"""Run one stage of the pipeline: python -m privacyqa STAGE --config CONFIG.

Each stage reads the outputs of earlier stages under the run root of the configuration and writes a new directory
named after the stage. Model training and inference run in the model environment through privacyqa.qa.train and
privacyqa.qa.infer.
"""
import argparse
import json
import sys

from privacyqa.common import ContractError
from privacyqa.config import Config


def stages():
    from privacyqa.attacks.audit import attack_audit
    from privacyqa.attacks.metrics import attack_metrics
    from privacyqa.attacks.synthetic import mechanism_audit
    from privacyqa.attacks.trained import run_attacks
    from privacyqa.banks import build_banks
    from privacyqa.beauty_catalog import build_beauty_catalog
    from privacyqa.movie_catalog import build_movie_catalog, fetch_tmdb
    from privacyqa.recipe_catalog import build_recipe_catalog
    from privacyqa.cohorts import freeze_cohorts
    from privacyqa.evidence import build_index
    from privacyqa.qa.analysis import analyze_qa
    from privacyqa.qa.data import prepare_qa, reference_fidelity
    from privacyqa.qa.preference import analyze_preference, prepare_preference
    from privacyqa.release import run_releases
    from privacyqa.roles import assign_roles
    from privacyqa.supports import build_supports
    return {
        'fetch-tmdb': ('tmdb', fetch_tmdb),
        'build-movie-catalog': ('movie-catalog', build_movie_catalog),
        'build-beauty-catalog': ('beauty-catalog', build_beauty_catalog),
        'build-recipe-catalog': ('recipe-catalog', build_recipe_catalog),
        'build-index': ('index', build_index),
        'assign-roles': ('roles', assign_roles),
        'build-banks': ('banks', build_banks),
        'freeze-cohorts': ('cohorts', freeze_cohorts),
        'build-supports': ('supports', build_supports),
        'release': ('releases', run_releases),
        'attack': ('attacks', run_attacks),
        'attack-metrics': ('attack-metrics', attack_metrics),
        'attack-audit': ('attack-audit', attack_audit),
        'mechanism-audit': ('mechanism-audit', mechanism_audit),
        'qa-data': ('qa-data', prepare_qa),
        'qa-fidelity': ('qa-fidelity', reference_fidelity),
        'qa-analysis': ('qa-analysis', analyze_qa),
        'preference-data': ('preference-data', prepare_preference),
        'preference-analysis': ('preference-analysis', analyze_preference),
    }


STAGES = ('fetch-tmdb', 'build-movie-catalog', 'build-beauty-catalog', 'build-recipe-catalog', 'build-index', 'assign-roles', 'build-banks', 'freeze-cohorts', 'build-supports', 'release', 'attack', 'attack-metrics',
          'attack-audit', 'mechanism-audit', 'qa-data', 'qa-fidelity', 'qa-analysis', 'preference-data', 'preference-analysis')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m privacyqa', description=__doc__.split('\n')[0])
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dry-run', action='store_true', help='release only: plan the measurements and write release-plan/ without taking any')
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
        name, run = stages()[args.stage]
        if args.stage == 'release':
            output = config.new_stage('release-plan' if args.dry_run else name)
            result = run(config, output, dry_run=args.dry_run)
        else:
            if args.dry_run:
                parser.error('--dry-run applies to the release stage only')
            result = run(config, config.new_stage(name))
    except ContractError as error:
        print(json.dumps(dict(status='failed', code=error.code, message=str(error))), file=sys.stderr)
        return 2
    print(json.dumps(dict(status=result['status'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
