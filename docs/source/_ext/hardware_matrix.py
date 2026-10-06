"""Render and validate the structured hardware inventory."""

from datetime import date
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.errors import ExtensionError

from _data.hardware import HARDWARE


_REQUIRED_FIELDS = {"name",
                    "object",
                    "kind",
                    "distribution",
                    "platforms",
                    "transport",
                    "dependencies",
                    "backends",
                    "maintenance",
                    "verification",
                    "verified_on",
                    "verification_details",
                    "example"}
_KINDS = {"Camera", "InOut", "Actuator"}
_DISTRIBUTIONS = {"core", "collection"}
_MAINTENANCE = {"maintained", "legacy"}
_VERIFICATION = {"verified", "unverified", "software-only"}


def _fail(index: int, message: str) -> None:
  raise ExtensionError(f"Hardware entry {index}: {message}")


def validate_hardware(app) -> None:
  """Validate all inventory entries before Sphinx reads any source page."""

  repository = Path(app.srcdir).parents[1]
  seen_names: set[tuple[str, str]] = set()
  seen_objects: set[str] = set()

  for index, entry in enumerate(HARDWARE, start=1):
    fields = set(entry)
    if fields != _REQUIRED_FIELDS:
      missing = sorted(_REQUIRED_FIELDS - fields)
      extra = sorted(fields - _REQUIRED_FIELDS)
      _fail(index, f"invalid fields; missing={missing}, extra={extra}")

    if entry["kind"] not in _KINDS:
      _fail(index, f"invalid kind {entry['kind']!r}")
    if entry["distribution"] not in _DISTRIBUTIONS:
      _fail(index, f"invalid distribution {entry['distribution']!r}")
    if entry["maintenance"] not in _MAINTENANCE:
      _fail(index, f"invalid maintenance status {entry['maintenance']!r}")
    if entry["verification"] not in _VERIFICATION:
      _fail(index, f"invalid verification status {entry['verification']!r}")

    expected_maintenance = {
      "core": "maintained",
      "collection": "legacy",
    }[entry["distribution"]]
    if entry["maintenance"] != expected_maintenance:
      _fail(index, "distribution and maintenance status disagree")

    in_collection = entry["object"].startswith("crappy.collection.")
    if in_collection != (entry["distribution"] == "collection"):
      _fail(index, "object path and distribution disagree")

    name_key = (entry["kind"], entry["name"])
    if name_key in seen_names:
      _fail(index, f"duplicate name {entry['name']!r}")
    if entry["object"] in seen_objects:
      _fail(index, f"duplicate object {entry['object']!r}")
    seen_names.add(name_key)
    seen_objects.add(entry["object"])

    for field in ("platforms", "transport", "dependencies", "backends"):
      values = entry[field]
      if not isinstance(values, tuple) or not all(
          isinstance(value, str) and value for value in values):
        _fail(index, f"{field} must be a tuple of non-empty strings")
    verified_on = entry["verified_on"]
    if verified_on is not None:
      if entry["verification"] != "verified":
        _fail(index, "only verified entries may have a verification date")
      try:
        date.fromisoformat(verified_on)
      except (TypeError, ValueError):
        _fail(index, f"invalid verification date {verified_on!r}")
    elif entry["verification"] == "verified":
      _fail(index, "verified entries require a verification date")

    if not isinstance(entry["verification_details"], str) or not entry[
        "verification_details"]:
      _fail(index, "verification_details must be a non-empty string")

    example = entry["example"]
    if example is not None:
      if not isinstance(example, str) or not example.startswith("examples/"):
        _fail(index, "example must be a repository-relative examples/ path")
      if not (repository / example).is_file():
        _fail(index, f"example does not exist: {example!r}")


def _joined(values: tuple[str, ...]) -> str:
  return ", ".join(values) if values else "None"


def _verification(entry: dict) -> str:
  label = {"verified": "Verified",
           "unverified": "Unverified",
           "software-only": "Software-only"}[entry["verification"]]
  return f"{label}. {entry['verification_details']}"


def _last_checked(entry: dict) -> str:
  if entry["verification"] == "software-only":
    return "Not applicable"
  return entry["verified_on"] or "Not recorded"


def _example(entry: dict) -> str:
  if entry["example"] is None:
    return "None available"
  url = ("https://github.com/LaboratoireMecaniqueLille/crappy/blob/master/" +
         entry["example"])
  return f"`Open example <{url}>`__"


class HardwareMatrix(Directive):
  """Insert one validated hardware category as a list-table."""

  required_arguments = 1
  final_argument_whitespace = False
  has_content = False

  def run(self) -> list[nodes.Node]:
    kind = self.arguments[0]
    if kind not in _KINDS:
      raise self.error(f"Expected one of {sorted(_KINDS)}, got {kind!r}")

    entries = sorted(
      (entry for entry in HARDWARE if entry["kind"] == kind),
      key=lambda entry: (entry["distribution"] != "core",
                         entry["name"].casefold()),
    )
    if not entries:
      raise self.error(f"No {kind} entries are defined")

    lines = [".. list-table::",
             "   :header-rows: 1",
             "   :widths: 28 12 14 14 18 18 11 11",
             "",
             "   * - Driver",
             "     - Distribution",
             "     - Operating systems",
             "     - Connection",
             "     - Dependency / backend",
             "     - Verification",
             "     - Last checked",
             "     - Example"]

    for entry in entries:
      distribution = {
        "core": "Core — maintained",
        "collection": "Collection — legacy",
      }[entry["distribution"]]
      dependency = _joined(entry["dependencies"])
      if entry["backends"]:
        dependency += f"; backends: {_joined(entry['backends'])}"
      lines.extend((
        f"   * - :class:`{entry['name']} <{entry['object']}>`",
        f"     - {distribution}",
        f"     - {_joined(entry['platforms'])}",
        f"     - {_joined(entry['transport'])}",
        f"     - {dependency}",
        f"     - {_verification(entry)}",
        f"     - {_last_checked(entry)}",
        f"     - {_example(entry)}",
      ))

    content = StringList(lines, source=self.state.document.current_source)
    container = nodes.container()
    self.state.nested_parse(content, 0, container)
    return list(container.children)


def setup(app) -> dict[str, object]:
  """Register the directive and validate the inventory for every build."""

  app.add_directive("hardware-matrix", HardwareMatrix)
  app.connect("builder-inited", validate_hardware)
  return {"parallel_read_safe": True, "parallel_write_safe": True}
