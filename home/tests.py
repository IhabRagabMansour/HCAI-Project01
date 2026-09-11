import re
import shutil
import subprocess
from pathlib import Path

from django.apps import apps
from django.conf import settings
from django.contrib.staticfiles import finders
from django.test import SimpleTestCase, TestCase


class RootUrlTest(TestCase):
    def test_root_redirects_to_home(self):
        response = self.client.get("/")
        self.assertRedirects(response, "/home/", fetch_redirect_response=False)

    def test_root_favicon_points_at_a_real_file(self):
        response = self.client.get("/favicon.ico")
        self.assertEqual(response.status_code, 301)
        target = response["Location"].removeprefix(settings.STATIC_URL)
        self.assertIsNotNone(finders.find(target))


_STATIC_TAG = re.compile(r"""{%\s*static\s+['"]([^'"]+)['"]""")
_PACKAGE_DIRS = {"site-packages", "dist-packages"}


def _is_installed_package(path: Path) -> bool:
    return bool(_PACKAGE_DIRS.intersection(path.parts))


def _project_template_dirs():
    """Template folders of this project's own apps, never of installed packages.

    Selected by where each app lives rather than by the virtualenv's folder name,
    which people call venv, .venv, env or anything else, often inside the repo.
    """
    dirs = [Path(d) for d in settings.TEMPLATES[0].get("DIRS", [])]
    dirs += [Path(config.path) / "templates" for config in apps.get_app_configs()
             if not _is_installed_package(Path(config.path))]
    return [d for d in dirs if d.is_dir()]


def _static_references():
    """Every path passed to {% static %} in any template of the project."""
    refs = set()
    for folder in _project_template_dirs():
        for template in folder.glob("**/*.html"):
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

        found = [Path(finders.find(ref)) for ref in _static_references() if finders.find(ref)]
        paths = [p.relative_to(base).as_posix() for p in found
                 if p.is_relative_to(base) and not _is_installed_package(p)]
        self.assertTrue(paths)
        result = subprocess.run(
            ["git", "check-ignore", "--stdin"], cwd=base, input="\n".join(paths),
            capture_output=True, text=True,
        )
        ignored = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(ignored, [])
