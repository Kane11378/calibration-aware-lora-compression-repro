"""Authenticate the reproducibility record without importing scientific code."""
from __future__ import annotations

import ast
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from urllib.parse import unquote, urlsplit, urlunsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]|/(?:Users|home)/[^\s/]+")
SECRETS = (
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
WORKFLOW = re.compile(r"\b(?:barrier\s+[A-Z]|Issue[- ]?\d+\s+(?:candidate|run)|stop[p]ed|(?:prompt|session)\s+(?:log|transcript|id)|(?:internal|private)\s+(?:chat|conversation|audit|workspace)|(?:agent|assistant|tool)\s+(?:session|workflow|handoff|prompt|task|run)|(?:workflow|repository)\s+(?:orchestrator|automation|handoff))\b", re.I)
FORBIDDEN_PARTS = {"__pycache__", "paper", "submission", "review", "draft_reference"}
TOP = {"code", "configs", "data_or_data_access", "environment", "figure_source_data", "figures", "manifests", "protocols", "results", "tables"}
SUFFIXES = {".py", ".json", ".csv", ".md", ".txt", ".cff", ".npz", ".pdf", ".svg", ".sha256", ""}
URL = re.compile(r"(?:https?|ssh|git)://[^\s<>\"'`\\]+|git@[A-Za-z0-9.-]+:[^\s<>\"'`\\]+", re.I)
PROVIDER_URLS = {
    ("orcid.org", "/0009-0006-7727-1214"),
    ("opensource.org", "/license/mit"),
    ("creativecommons.org", "/licenses/by/4.0/legalcode.en"),
    ("creativecommons.org", "/policies/"),
    ("huggingface.co", "/datasets/nyu-mll/glue"),
    ("huggingface.co", "/datasets/google/boolq"),
    ("huggingface.co", "/microsoft/deberta-v3-base"),
    ("huggingface.co", "/FacebookAI/roberta-base"),
    ("docs.github.com", "/en/rest/releases/releases"),
    ("github.com", "/citation-file-format/citation-file-format/blob/main/schema-guide.md"),
    ("help.zenodo.org", "/docs/deposit/describe-records/reserve-doi/"),
    ("zenodo.org", "/help/versioning"),
    ("help.zenodo.org", "/docs/deposit/describe-records/publication-date/"),
    ("help.zenodo.org", "/docs/github/describe-software/citation-file/"),
    ("help.zenodo.org", "/docs/deposit/manage-records/"),
    ("json-schema.org", "/draft/2020-12/schema"),
}
AI_DISCLOSURE_SHA256 = "4e0656406857f1c9f0df494bb71da91a223ff16cb7dacaaaff5116bbff305a8b"
SOURCE_BASELINE = "12ed6fc8ac36ef4c117f203879446769c711e41d"
SOURCE_PROVENANCE_SHA256 = "50632273ac8109ad197f59891165ffaa01c1fe853f983285997c6b07c767caf9"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    require(name and "\\" not in name and not p.is_absolute() and
            all(x not in {"", ".", ".."} for x in name.split("/")) and
            not any(":" in x or x.rstrip(" .") != x for x in p.parts), "unsafe name: " + name)
    return p


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load_json(text):
    return json.loads(text, object_pairs_hook=unique_object)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


def literal_strings(tree):
    """Inspect literal string constructions without evaluating Python code."""
    known = {}

    def fold(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
            return node.value
        if isinstance(node, ast.Name):
            return known.get(node.id)
        if isinstance(node, ast.BinOp):
            left, right = fold(node.left), fold(node.right)
            if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
                return left + right if len(left) + len(right) <= 1000000 else None
            if isinstance(node.op, ast.Mult):
                if isinstance(left, int):
                    left, right = right, left
                if isinstance(left, str) and isinstance(right, int) and 0 <= right <= 10000 and len(left)*right <= 1000000:
                    return left*right
        if isinstance(node, ast.JoinedStr):
            values = []
            for part in node.values:
                value = fold(part.value) if isinstance(part, ast.FormattedValue) and part.format_spec is None else fold(part)
                if not isinstance(value, (str, int)):
                    return None
                values.append(str(value))
            return ''.join(values) if sum(map(len, values)) <= 1000000 else None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'join' and
                not node.keywords and len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple))):
            delimiter = fold(node.func.value)
            values = [fold(item) for item in node.args[0].elts]
            if (isinstance(delimiter, str) and all(isinstance(value, str) for value in values) and
                    sum(map(len, values)) + len(delimiter)*len(values) <= 1000000):
                return delimiter.join(values)
        return None

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = fold(node.value)
            if isinstance(value, str):
                yield value
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    known.pop(target.id, None)
                    if isinstance(value, (str, int)):
                        known[target.id] = value
    for node in ast.walk(tree):
        value = fold(node)
        if isinstance(value, str):
            yield value


def scan_text(text, relative, approved_urls, check_urls=True):
    decoded = unquote(text)
    if decoded != text:
        # At most one extra decoding is inspected; residual URL percent escapes
        # are rejected below instead of interpreted as program instructions.
        text = text + '\n' + decoded + '\n' + unquote(decoded)
    require(PRIVATE_PATH.search(text) is None, "private path/link: " + relative)
    require(not any(rule.search(text) for rule in SECRETS), "credential: " + relative)
    # The exact formal declaration is protected by its complete byte identity.
    if relative != "AI_USE.md":
        require(WORKFLOW.search(text) is None, "private workflow: " + relative)
    if not check_urls:
        return
    for match in URL.finditer(text):
        raw = match.group().rstrip('.,;:)]}')
        parsed = urlsplit(raw)
        decoded_path = unquote(parsed.path)
        if (parsed.scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment) == ('http', 'www.w3.org', '/2000/svg', '', ''):
            continue
        require(parsed.scheme == "https" and parsed.username is None and parsed.password is None and
                parsed.port in {None, 443} and not parsed.query and not parsed.fragment and
                '%' not in decoded_path and '\\' not in decoded_path and
                all(part not in {'.', '..'} for part in decoded_path.split('/')),
                "unapproved source URL: " + relative)
        require((parsed.hostname, decoded_path) in PROVIDER_URLS or raw in approved_urls,
                "unapproved source URL: " + relative)


def check_aliases(value, actual):
    require(isinstance(value, dict) and value.get("schema") == "gri-h1-data-aliases-v2" and
            set(value) == {"schema", "alias_namespace", "duplicate_exports", "source_exports", "comparison_basis"}, "alias schema")
    require(isinstance(value["duplicate_exports"], dict) and isinstance(value["source_exports"], dict) and
            set(value["duplicate_exports"]) == set(value["source_exports"]), "alias source membership")
    aliases = value["duplicate_exports"]
    require(len({name.casefold() for name in aliases}) == len(aliases), "alias duplicate/case collision")
    seen = set()
    canonical = {name.casefold() for name in actual}
    for source, target in aliases.items():
        safe_name(source)
        require(source.casefold() not in seen, "alias duplicate/case collision")
        seen.add(source.casefold())
        require(isinstance(target, str), "alias target schema")
        safe_name(target)
        require(source.casefold() != target.casefold(), "alias self mapping")
        require(source.casefold() not in canonical, "alias shadows canonical file")
        require(target in actual, "alias target missing")
        require(actual[target].suffix == '.csv', "alias target is not CSV")
        require(target.casefold() not in {x.casefold() for x in aliases}, "alias chain/cycle")
        record = value['source_exports'][source]
        require(isinstance(record, dict) and set(record) == {'source_path', 'bytes', 'sha256', 'explanation'}, "alias provenance schema")
        safe_name(record['source_path'])
        require(record['source_path'] == 'paper/'+source and record['source_path'] not in actual, "alias source provenance")
        require(type(record.get("bytes")) is int and record['bytes'] == actual[target].stat().st_size and
                record.get("sha256") == sha(actual[target].read_bytes()), "alias target identity")
        require(isinstance(record.get("explanation"), str) and record["explanation"].strip(), "alias explanation")
    return len(aliases)


def check_provenance(value, actual):
    """Validate origin content identifiers separately from current file integrity."""
    require(isinstance(value, dict) and set(value) == {"source_baseline", "members"}, "source provenance fields")
    require(value["source_baseline"] == SOURCE_BASELINE, "source baseline identity")
    members = value["members"]
    exclusions = {"checksums.sha256", "manifests/file_inventory.json",
                  "manifests/file_provenance.json", "manifests/license_scope.json"}
    require(isinstance(members, dict) and set(members) == set(actual) - exclusions, "source provenance membership")
    count = 0
    for name, record in members.items():
        safe_name(name)
        require(isinstance(record, dict) and set(record) == {"source_sha256"}, "origin content fields")
        digest = record["source_sha256"]
        require(digest is None or isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "origin content identifier")
        count += digest is not None
    identity = sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    require(identity == SOURCE_PROVENANCE_SHA256, "source provenance identity")
    return {"source_baseline": SOURCE_BASELINE, "members": len(members),
            "source_hashes": count, "members_without_source_hash": len(members) - count,
            "identity_sha256": identity}


def cff_scalars(text):
    """Parse this package's generated scalar/list subset, not arbitrary YAML."""
    top = {}
    active = None
    current = None
    def scalar(value):
        if value.startswith('"'):
            result = json.loads(value)
            require(isinstance(result, str), "CFF string scalar")
            return result
        if value.startswith("'"):
            require(value.endswith("'"), "CFF scalar quoting")
            return value[1:-1].replace("''", "'")
        require(value not in {'null', '~', 'true', 'false', ''} and not any(c in value for c in '{}[]'), "CFF absent/complex scalar")
        return value
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        if line[0].isspace():
            require(active in {'authors', 'identifiers'}, "CFF unsupported nested field")
            match = re.fullmatch(r'  - ([a-z][a-z-]*):\s*(.+)', line)
            if match:
                current = {};top[active].append(current)
            else:
                match = re.fullmatch(r'    ([a-z][a-z-]*):\s*(.+)', line)
            require(match is not None and current is not None, "CFF list syntax")
            key, value = match.groups()
            require(key not in current, "CFF duplicate list field")
            current[key] = scalar(value)
            continue
        match = re.fullmatch(r'([a-z][a-z-]*):(?:\s*(.*))?', line)
        require(match is not None, "CFF top-level syntax")
        key, value = match.groups()
        require(key not in top, "CFF duplicate field")
        active = key;current = None
        top[key] = [] if key in {'authors', 'identifiers'} and not value else scalar(value)
    require(set(top) <= {'cff-version', 'message', 'title', 'type', 'authors', 'version', 'repository-code',
                         'date-released', 'commit', 'doi', 'url', 'identifiers', 'abstract'}, "CFF unsupported field")
    return top


def iso_date(value, name):
    require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value), name)
    parsed = date.fromisoformat(value)
    require(parsed <= datetime.now(timezone.utc).date(), "future " + name)
    return parsed


def check_metadata(meta, cff_text, version):
    required = {'schema_version', 'title', 'version', 'resource_type', 'creator', 'scientific_scope', 'reproduction'}
    optional = {'repository_url', 'release_url', 'release_commit', 'release_date', 'doi', 'concept_doi', 'zenodo_record_url'}
    require(isinstance(meta, dict) and required <= set(meta) <= required | optional and
            meta['schema_version'] == 1 and all(value is not None for value in meta.values()), 'publication metadata fields')
    require(meta['version'] == version == '1.0.1' and meta['resource_type'] == 'software', 'package identity')
    require(meta['title'] == 'Beyond Rank: Calibration-Aware Dependency Control for Post-Training LoRA Compression', 'publication title')
    require(meta['creator'] == {'name':'Kang Wang', 'email':'Kane.97022@gmail.com', 'orcid':'0009-0006-7727-1214',
            'affiliation':'College of Finance and Economics, Taiyuan University of Technology, Taiyuan 030024, Shanxi, China'}, 'creator identity')
    reproduction = meta['reproduction']
    require(set(reproduction) == {'static_verification_command', 'experimental_runner_complete',
            'document_rebuild_supported', 'full_preprint_included', 'required_assets'} and
            reproduction['static_verification_command'] == 'python -I -B code/verify_release_static.py' and
            reproduction['experimental_runner_complete'] is False and reproduction['document_rebuild_supported'] is False and
            reproduction['full_preprint_included'] is False, 'reproduction scope')
    require(reproduction['required_assets'] == ['Pinned upstream models and benchmark text and labels',
            'Trained adapters', 'Jacobians and endpoint execution assets', 'Original scientific runtime'], 'experimental requirements')
    require(meta['scientific_scope'] == {'methods':['H1','S1','G1','constrained solver'],
            'stored_evidence':'Complete stored results, including adverse and inconclusive comparisons, paired KL values, table and figure source data, and experimental protocols.'}, 'scientific scope')
    cff = cff_scalars(cff_text)
    require(cff.get('version') == version and cff.get('cff-version') == '1.2.0' and cff.get('type') == 'software' and
            cff.get('title') == meta['title'] and cff.get('authors') == [{'family-names':'Wang', 'given-names':'Kang',
            'affiliation':meta['creator']['affiliation'], 'email':meta['creator']['email'],
            'orcid':urlunsplit(('https', 'orcid.org', '/'+meta['creator']['orcid'], '', ''))}], 'citation identity')
    require(cff.get('message') == 'Please cite Kang Wang and version '+version+' when using these reproducibility materials.', 'citation message')
    abstract = 'Version '+version+' reproducibility materials accompanying '+meta['title']+'. Kang Wang is the sole author. Stored results include adverse and inconclusive comparisons. Standard-library static verification authenticates membership, hashes and structures without importing scientific code or recomputing experiments. The package includes the four-file H1/S1/G1/solver mathematical implementation, but omits upstream models, benchmark text and labels, trained adapters, Jacobians and endpoint execution assets required to repeat experiments. The article, Supplement and submission materials are excluded. Project-authored Python code is MIT licensed; project-authored results, data and documentation are CC BY 4.0 under the per-file allocation in LICENSE and manifests/license_scope.json. Third-party rights and exceptions remain in force.'
    require(cff.get('abstract') == abstract, 'description and licensing scope')
    urls = set()
    expected_identifiers = []
    if 'repository_url' in meta:
        repository = meta['repository_url']
        parsed = urlsplit(repository)
        require(parsed.scheme == 'https' and parsed.netloc == 'github.com' and
                re.fullmatch(r'/[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',parsed.path) and
                not parsed.query and not parsed.fragment, 'repository URL')
        require(cff.get('repository-code') == repository, 'citation repository')
        urls.add(repository)
    else:
        require('repository-code' not in cff, 'citation repository field')
    if 'release_url' in meta:
        require('repository_url' in meta and meta['release_url'] == meta['repository_url']+'/releases/tag/v'+version, 'release URL')
        urls.add(meta['release_url'])
        expected_identifiers.append({'type':'url','value':meta['release_url'],'description':'GitHub release for version '+version})
    for key, cff_key in [('release_commit','commit'),('release_date','date-released')]:
        if key in meta:
            if key == 'release_commit':
                require(isinstance(meta[key],str) and re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})',meta[key]), 'release commit')
            else:
                iso_date(meta[key],'release date')
            require(cff.get(cff_key) == meta[key], 'citation '+cff_key)
        else:
            require(cff_key not in cff, 'citation '+cff_key+' field')
    if 'doi' in meta:
        doi = meta['doi']
        require(isinstance(doi,str) and re.fullmatch(r'10\.5281/zenodo\.[1-9]\d*',doi), 'version DOI syntax')
        require(cff.get('doi') == doi, 'citation DOI')
        urls.add(urlunsplit(('https', 'doi.org', '/'+doi, '', '')))
        expected_identifiers.append({'type':'doi','value':doi,'description':'Version DOI for version '+version})
    else:
        require('doi' not in cff and 'concept_doi' not in meta and 'zenodo_record_url' not in meta, 'citation DOI fields')
    if 'concept_doi' in meta:
        concept = meta['concept_doi']
        require(isinstance(concept,str) and re.fullmatch(r'10\.5281/zenodo\.[1-9]\d*',concept) and concept != meta['doi'], 'DOI roles')
        urls.add(urlunsplit(('https', 'doi.org', '/'+concept, '', '')))
        expected_identifiers.append({'type':'doi','value':concept,'description':'Concept DOI for all versions'})
    if 'zenodo_record_url' in meta:
        require(meta['zenodo_record_url'] == urlunsplit(('https', 'zenodo.org', '/records/'+meta['doi'].rsplit('.',1)[1], '', '')), 'record DOI consistency')
        urls.add(meta['zenodo_record_url'])
    landing = meta.get('zenodo_record_url',meta.get('release_url',meta.get('repository_url')))
    if landing is None:
        require('url' not in cff, 'citation landing field')
    else:
        require(cff.get('url') == landing, 'citation landing URL')
    require(cff.get('identifiers',[]) == expected_identifiers, 'citation identifier roles')
    return urls


def check_reader_contract(meta, readme, reproduce):
    require('static' in reproduce and 'not a complete retraining runner' in reproduce and
            'Those assets are omitted' in reproduce and 'no\nmanuscript or Supplement build command is provided' in reproduce,
            'reproduction scope contract')
    require(meta['version'] in readme, 'README package identity')
    for key in ['repository_url','release_url','doi','zenodo_record_url']:
        if key in meta:
            require(meta[key] in readme, 'README citation consistency')


def npz_schema(raw, relative):
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        require(z.testzip() is None, "NPZ CRC: " + relative)
        names = z.namelist()
        require(len(names) == len(set(names)), "NPZ duplicate")
        require(set(names) == {"ids.npy", "H1_minus_G1.npy"}, "NPZ fields: " + relative)
        for item in z.infolist():
            safe_name(item.filename)
            data = z.read(item)
            require(data[:6] == b"\x93NUMPY", "NPY magic")
            major = data[6]
            require(major in {1, 2, 3}, "NPY version")
            n = struct.unpack("<H" if major == 1 else "<I", data[8:10 if major == 1 else 12])[0]
            start = 10 if major == 1 else 12
            h = ast.literal_eval(data[start:start+n].decode("latin1" if major < 3 else "utf-8").strip())
            require(set(h) == {"descr", "fortran_order", "shape"}, "NPY header")
            dtype, shape = h["descr"], h["shape"]
            require(isinstance(dtype, str) and isinstance(shape, tuple) and all(isinstance(x, int) and x > 0 for x in shape), "NPY object/shape")
            expected_n = 3270 if "boolq" in relative else 5463 if "qnli" in relative else 2048
            require(shape == (expected_n,) and h["fortran_order"] is False, "NPY original shape/order")
            if item.filename == "ids.npy":
                require(dtype in {"<U69", "<i8"}, "NPY IDs type")
                width = 69 * 4 if dtype == "<U69" else 8
            else:
                require(dtype in {"<f8", ">f8", "=f8"}, "NPY numerical type")
                width = 8
            count = 1
            for x in shape:
                count *= x
            require(len(data)-start-n == width*count, "NPY payload length")
            if item.filename == "ids.npy" and dtype == "<U69":
                values = [data[start+n+i*width:start+n+(i+1)*width].decode("utf-32-le").rstrip("\x00") for i in range(count)]
                pattern = r"google/boolq@35b264d03638db9f4ce671b711558bf7ff0f80d5:validation:\d+"
                require(all(re.fullmatch(pattern, value) for value in values) and len(values) == len(set(values)), "NPY benchmark identifier privacy/schema")
            elif item.filename == "ids.npy":
                values = struct.unpack("<" + "q"*count, data[start+n:])
                require(all(value >= 0 for value in values) and len(values) == len(set(values)), "NPY positional identifiers")
            rows.append({"array": item.filename, "dtype": dtype, "shape": shape, "sha256": sha(data)})
    return rows


def verify():
    actual = {}
    case_names = set()
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        parts = safe_name(rel).parts
        require(not any(x.startswith('.') or x.lower() in FORBIDDEN_PARTS for x in parts), "private directory: " + rel)
        require(not p.is_symlink() and not getattr(p, "is_junction", lambda: False)(), "link: " + rel)
        require(p.resolve().is_relative_to(ROOT.resolve()), "escaping path")
        if p.is_file():
            require(rel.casefold() not in case_names, "case-colliding file")
            case_names.add(rel.casefold())
            require(len(parts) == 1 or parts[0] in TOP, "unexpected directory")
            require(p.suffix in SUFFIXES, "unexpected type")
            require(not re.search(r"graphical_abstract|wireframe|issue\d+|cover_letter|highlights|manuscript|supplement\.|rebuild_documents|regenerate_paper|(?:^|/)work/", rel, re.I), "private/obsolete file: " + rel)
            actual[rel] = p
    listed = {}
    for line in (ROOT/"checksums.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        safe_name(rel)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and rel not in listed and rel != "checksums.sha256", "checksum record")
        require(rel in actual and sha(actual[rel].read_bytes()) == digest, "hash mismatch: " + rel)
        listed[rel] = digest
    require(len(actual) == 286, "authorized 286-member release")
    require(set(actual)-{"checksums.sha256"} == set(listed), "exact checksum membership")
    inv = load_json((ROOT/"manifests/file_inventory.json").read_text())
    require(set(inv["members"]) == set(listed)-{"manifests/file_inventory.json"}, "inventory membership")
    for rel, row in inv["members"].items():
        require(row["bytes"] == actual[rel].stat().st_size and row["sha256"] == listed[rel], "inventory identity")
    provenance_sources = check_provenance(load_json((ROOT/"manifests/file_provenance.json").read_text()), actual)
    meta = load_json((ROOT/"manifests/publication_metadata.json").read_text())
    approved_urls = check_metadata(meta, (ROOT/"CITATION.cff").read_text(), (ROOT/"VERSION").read_text().strip())
    for rel in ["README.md", "REPRODUCE.md", "AI_USE.md", "LICENSE", "LICENSE-MIT.txt", "LICENSE-CC-BY-4.0.txt", "THIRD_PARTY_NOTICES.md", "code/README.md", "protocols/h1_method_lock.md"]:
        require(rel in actual, "required file: " + rel)
    require(sha(actual['AI_USE.md'].read_bytes()) == AI_DISCLOSURE_SHA256, "formal AI disclosure identity")
    check_reader_contract(meta, actual['README.md'].read_text(), actual['REPRODUCE.md'].read_text())
    aliases = check_aliases(load_json((ROOT/'manifests/data_aliases.json').read_text()), actual)
    identities = load_json((ROOT/"manifests/scientific_code_identity.json").read_text())["members"]
    scientific = {r for r in actual if r.startswith("code/experiments/") and r.endswith(".py")}
    require(scientific == set(identities) and len(scientific) == 4, "mathematical closure")
    for r, row in identities.items():
        require(listed[r] == row["sha256"], "scientific source identity")
    scope = load_json((ROOT/"manifests/license_scope.json").read_text())["members"]
    require(set(scope) == set(actual)-{"checksums.sha256", "manifests/file_inventory.json", "manifests/license_scope.json"}, "rights membership")
    require(all(x in {"MIT", "CC-BY-4.0", "third-party-legal-text"} for x in scope.values()), "rights allocation")
    csv_rows, arrays, imports, scanned = {}, {}, {}, 0
    for rel, p in actual.items():
        if p.suffix in {".md", ".txt", ".json", ".py", ".csv", ".cff", ".svg"} or p.name in {"LICENSE", "VERSION"}:
            text = p.read_text(encoding="utf-8-sig")
            scanned += len(text.splitlines())
            scan_text(text, rel, approved_urls, check_urls=p.suffix != '.json')
            if p.suffix == ".json":
                value = load_json(text)
                for item in strings(value):
                    scan_text(item, rel, approved_urls)
        if p.suffix == ".csv":
            with p.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                require(header and len(header) == len(set(header)), "CSV header")
                n = 0
                for row in reader:
                    require(len(row) == len(header), "CSV width: " + rel)
                    n += 1
            csv_rows[rel] = {"columns": header, "data_rows": n}
        if p.suffix == ".npz":
            arrays[rel] = npz_schema(p.read_bytes(), rel)
        if p.suffix == ".pdf":
            raw = p.read_bytes()
            require(raw.startswith(b"%PDF-") and b"%%EOF" in raw[-2048:], "PDF framing")
        if p.suffix == ".svg":
            import xml.etree.ElementTree as ET
            tree = ET.fromstring(p.read_bytes())
            for el in tree.iter():
                require(el.tag.split("}")[-1] not in {"script", "foreignObject", "image"}, "active SVG")
                require(not any(k.lower().startswith("on") or k.split("}")[-1] in {"href", "src"} for k in el.attrib), "SVG external reference")
        if p.suffix == ".py":
            tree = ast.parse(p.read_text(encoding="utf-8-sig"))
            for value in literal_strings(tree):
                scan_text(value, rel, approved_urls)
            modules = []
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for name in names:
                    if name.startswith("experiments."):
                        require("code/"+name.replace(".", "/")+".py" in actual, "missing source dependency: " + name)
                    modules.append(name)
            if rel == "code/verify_release_static.py":
                require(all(x.split(".")[0] in sys.stdlib_module_names for x in modules), "verifier imports")
                require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"eval", "exec", "__import__"} for n in ast.walk(tree)), "dynamic execution")
            imports[rel] = sorted(set(modules))
    result = {"status": "PASS", "files": len(actual), "checksum_members": len(listed), "csv_tables": csv_rows, "npz_schemas": arrays,
              "source_imports": imports, "text_lines_scanned": scanned, "package_version": meta['version'], "data_aliases": aliases, "provenance_sources": provenance_sources, "citation_metadata_consistent": True,
              "external_publication_verified": False, "cff_validation": "generated scalar/list subset; full YAML/CFF schema validation separate",
              "full_preprint_included": False, "document_rebuild_supported": False, "scientific_execution": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    verify()
