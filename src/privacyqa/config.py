"""The run configuration: inputs, private parameters and the plan of every stage, in one JSON file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from privacyqa.common import PROJECT_ROOT, read_json, require

SECTIONS = ('run_root', 'domains', 'recipe_search_module', 'private', 'roles', 'banks', 'test', 'cohorts', 'release',
            'attack', 'attack_metrics', 'attack_audit', 'mechanism_audit', 'qa', 'preference')


@dataclass
class Config:
    path: Path
    data: dict

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = read_json(path)
        require(isinstance(data, dict) and set(data) == set(SECTIONS), 'invalid_config',
                'Declare every section of the run configuration: ' + ', '.join(SECTIONS))
        return cls(path, data)

    def __getitem__(self, name):
        return self.data[name]

    def file(self, value):
        """An input path; relative paths start at the directory of the configuration file."""
        path = Path(value)
        return (path if path.is_absolute() else self.path.parent / path).resolve()

    @property
    def root(self):
        return self.file(self.data['run_root'])

    def stage(self, name):
        return self.root / name

    def new_stage(self, name):
        root = self.root
        require(not root.is_relative_to(PROJECT_ROOT), 'unsafe_run_root', 'Keep run outputs outside the repository')
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        output = root / name
        require(not output.exists(), 'stage_exists', f'{name} exists; outputs are never overwritten')
        output.mkdir(mode=0o700)
        return output

    def finished(self, name, status):
        """The manifest of a completed earlier stage."""
        path = self.stage(name) / 'manifest.json'
        require(path.is_file(), 'missing_stage', f'Run the {name} stage first')
        manifest = read_json(path)
        require(manifest['status'] == status, 'incomplete_stage', f'The {name} stage did not complete')
        return manifest
