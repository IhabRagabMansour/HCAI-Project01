import re
import shutil
import subprocess
from pathlib import Path

from django.conf import settings
from django.contrib.staticfiles import finders
from django.test import SimpleTestCase, TestCase


class RootUrlTest(TestCase):
    def test_root_redirects_to_home(self):
        response = self.client.get("/")
        self.assertRedirects(response, "/home/", fetch_redirect_response=False)


_STATIC_TAG = re.compile(r"""{%\s*static\s+['"]([^'"]+)['"]""")


def _static_references():
    """Every path passed to {% static %} in any template of the project."""
    refs = set()
    base = Path(settings.BASE_DIR)
    for template in base.glob("**/templates/**/*.html"):
        if "venv" in template.parts:
            continue
        refs.update(_STATIC_TAG.findall(template.read_text(encoding="utf-8")))
    return sorted(refs)


class StaticAssetTest(SimpleTestCase):
    """Charts render client-side, so a missing script leaves the page returning
    200 with an empty chart. These checks catch that before a reviewer does."""

    def test_templates_reference_static_files(self):
        self.assertGreater(len(_static_references()), 10)

    def test_every_referenced_static_file_exists(self):
        missing = [ref for ref in _static_references() if not finders.find(ref)]
        self.assertEqual(missing, [])

    def test_no_referenced_static_file_is_ignored_by_git(self):
        """A file can exist here yet be absent from every clone.

        The Python .gitignore template ignores any `lib/` directory, which once
        dropped the vendored Chart.js from the repository without a trace.
        """
        base = Path(settings.BASE_DIR)
        if not shutil.which("git") or not (base / ".git").exists():
            self.skipTest("not a git checkout")

        paths = [Path(finders.find(ref)).relative_to(base).as_posix()
                 for ref in _static_references() if finders.find(ref)]
        result = subprocess.run(
            ["git", "check-ignore", "--stdin"], cwd=base, input="\n".join(paths),
            capture_output=True, text=True,
        )
        ignored = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(ignored, [])
