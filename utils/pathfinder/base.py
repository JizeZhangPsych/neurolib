from __future__ import annotations
import re, glob
from pathlib import Path
from string import Formatter
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Set
import parse


@dataclass(frozen=True)
class BasePathfinder(ABC):
    """
    Immutable double-level dictionary of file_id + kind → Path,
    with field-based parsing and reverse lookup.

    Placeholders starting with 'foo' are reserved and ignored
    for file_id computation. Only the remaining core fields
    are used in dict2id / id2dict.

    Attributes
    ----------
    file_patterns : Dict[str, str]
        Mapping from file kinds (e.g., 'raw', 'preproc') to filename patterns.
        Patterns can contain `{field}` placeholders, e.g.
        '/path/to/your/dataset/{subject}/{session}/{subject}_{task}_raw.fif'.
    fieldnames : Set[str]
        Set of all core fieldnames found in patterns (excluding foo* placeholders).
    files : Dict[str, Dict[str, Path]]
        Double-level mapping of file_id → kind → Path.
    
    Usage
    -----
    1. Define a subclass implementing dict2id / id2dict:
    
        class EEGPathfinder(BasePathfinder):
            def dict2id(self, fields):
                # Example: 'subject_session_task' → '010102'
                # run defaults to '01' if is None. this could happens if some files are shared across runs.
                # if run-01 is not always present in your dataset, you can also iterate over all possible runs to find an available one.
                
                runs = fields['run'] if fields.get('run') is not None else '01'
                return f"{fields['subject']}{fields['session']}{runs}"
                
            def id2dict(self, file_id):
                # Reverse mapping from '010102' → dict
                return {'subject': file_id[:2], 'session': file_id[2:4], 'run': file_id[4:]}

    2. Instantiate with file patterns:

        file_patterns = {
            'raw': '/path/to/your/dataset/{subject}/{session}/{subject}_{run}_{foo1}raw{foo2}.fif',
            'preproc': '/path/to/your/dataset/{subject}/{session}/{subject}_{run}_preproc.fif',
            'polhemus': '/path/to/your/dataset/{subject}/{session}/{subject}_polhemus.txt', 
        }
        pf = DatasetPathfinder(file_patterns)
        
        This example assumes different runs shares the same polhemus file, so with both 010101 and 010102,
        pf['010101']['polhemus'] and pf['010102']['polhemus'] will point to the same file.
        
        Any placeholders starting with 'foo' are ignored and not included in fieldnames or file_id.

    3. Access files by file_id:

        file_id = '010101'  # subject 01, session 01, run 01
        raw_path = pf[file_id]['raw']
        preproc_path = pf[file_id]['preproc']

    4. Refresh from disk if files change:

        pf.refresh()

    Notes
    -----
    - The class automatically parses filenames based on `file_patterns` and
      builds the `files` dictionary upon initialization.
    - Missing fields in filenames (e.g., files shared across runs) are set to None.
    - Duplicate file_ids will raise a ValueError.
    - Use `get_file_ids()` to iterate over all file_ids.
    """

    file_patterns: Dict[str, str]
    fieldnames: Set[str] = field(init=False)
    files: Dict[str, Dict[str, Path]] = field(init=False)

    def __post_init__(self):
        # Extract all placeholders from patterns
        all_placeholders: Set[str] = set()
        for pattern in self.file_patterns.values():
            all_placeholders.update(
                fname for _, fname, _, _ in Formatter().parse(pattern) if fname
            )

        # Ignore any placeholder starting with 'foo'
        core_fields = {f for f in all_placeholders if not f.startswith("foo")}
        object.__setattr__(self, "fieldnames", core_fields)

        # Populate files dict by scanning disk
        self.refresh()
        
    def refresh(self):
        """
        Two-step refresh to handle ambiguous patterns with consecutive placeholders.

        1. Scan unambiguous patterns (no '}{') to build canonical file_ids.
        2. Use these file_ids to resolve ambiguous patterns safely.
        """
        files: dict[str, dict[str, Path]] = {}

        # Step 1: unambiguous patterns
        unambiguous = {k: p for k, p in self.file_patterns.items() if "}{" not in p}
        if not unambiguous:
            raise ValueError("No unambiguous pattern found! Cannot generate initial file_ids.")
        first_kind = sorted(unambiguous.keys())[0]
        matched = self._glob_pattern(first_kind)
        for file_id, fname in matched.items():
            files[file_id] = {first_kind: Path(fname)}

        # Step 2: ambiguous patterns
        for kind, pattern in self.file_patterns.items():
            if kind == first_kind:
                continue
            for file_id in files.keys():
                fields = self.id2dict(file_id)
                # Fill foo placeholders with '*' to allow globbing
                all_placeholders = {fname for _, fname, _, _ in Formatter().parse(pattern) if fname}
                placeholders = {p: "*" for p in all_placeholders if p.startswith("foo")}
                fmt_dict = {**fields, **placeholders}
                try:
                    expected_fname = pattern.format(**fmt_dict)
                except KeyError:
                    continue  # missing field, skip

                candidates = glob.glob(expected_fname)
                if not candidates:
                    continue
                # assign first matching file
                if len(candidates) > 1:
                    raise ValueError(
                        f"Multiple ambiguous files found for file_id '{file_id}', kind '{kind}': {candidates}"
                    )
                files[file_id][kind] = Path(candidates[0])

        object.__setattr__(self, "files", files)

    def __getitem__(self, file_id: str) -> dict[str, Path]:
        """Return dict of {kind: Path} for a file_id."""
        return self.files[file_id]

    def get_file_ids(self) -> list[str]:
        """Return all file IDs for this dataset."""
        return list(self.files.keys())

    def filename2id(self, filename: str, kind: str) -> str:
        """
        Convert a filename into a file_id using the core fields
        of the pattern for the given kind.
        """
        return self.dict2id(self.filename2dict(filename, kind))

    def filename2dict(self, filename: str, kind: str) -> dict[str, str | None]:
        """
        Convert a filename into a dictionary of core fields → parsed values.
        Fields corresponding to placeholders starting with 'foo' are ignored.
        Missing core fields are set to None.

        Raises a warning if any core field is missing in the filename.
        """
        if kind not in self.file_patterns:
            raise KeyError(f"Kind '{kind}' not found in file_patterns.")

        pattern = self.file_patterns[kind]
        parsed = parse.parse(pattern, filename)
        if parsed is None:
            raise ValueError(
                f"Failed to parse filename '{filename}' with pattern of kind '{kind}'"
            )

        parsed_dict = parsed.named
        core_dict = {}
        for field in self.fieldnames:
            value = parsed_dict.get(field, None)
            if value is None:
                print(f"Warning: field '{field}' missing in file '{filename}'")
            core_dict[field] = value
        return core_dict

    def _glob_pattern(self, kind: str) -> dict[str, str]:
        """
        Scan filesystem for all files matching self.file_patterns[kind].
        Returns a dict mapping file_id → filename for unique matches.
        """
        pattern = self.file_patterns[kind]
        glob_pattern = re.sub(r"{.*?}", "*", pattern)
        candidate_files = sorted(glob.glob(glob_pattern))

        matched_files: dict[str, str] = {}
        for fname in candidate_files:
            if parse.parse(pattern, fname) is None:
                continue

            fields = self.filename2dict(fname, kind)
            file_id = self.dict2id(fields)

            if file_id in matched_files:
                raise ValueError(
                    f"Duplicate file_id '{file_id}' for multiple files: "
                    f"{matched_files[file_id]} and {fname}"
                )

            matched_files[file_id] = fname

        return matched_files

    @abstractmethod
    def dict2id(self, fields: Dict[str, str | None]) -> str:
        """Convert a dictionary of core fields into a canonical file_id string."""

    @abstractmethod
    def id2dict(self, file_id: str) -> Dict[str, str | None]:
        """Convert a canonical file_id string back into a dictionary of core fields."""