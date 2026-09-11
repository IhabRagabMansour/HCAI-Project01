import re
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


def _project_template_dirs():
    """Template folders of this project's own apps, not of installed packages."""
    dirs = [Path(d) for d in settings.TEMPLATES[0].get("DIRS", [])]
    dirs += [Path(config.path) / "templates" for config in apps.get_app_configs()
             if not _PACKAGE_DIRS.intersection(Path(config.path).parts)]
    return [d for d in dirs if d.is_dir()]


def _static_references():
    """Every path passed to {% static %} in any template of the project."""
    refs = set()
    for folder in _project_template_dirs():
        for template in folder.glob("**/*.html"):
            refs.update(_STATIC_TAG.findall(template.read_text(encoding="utf-8")))
    return sorted(refs)


class StaticAssetTest(SimpleTestCase):
    """Charts are drawn in the browser, so a missing script would leave a page
    loading normally with an empty chart."""

    def test_templates_reference_static_files(self):
        self.assertGreater(len(_static_references()), 10)

    def test_every_referenced_static_file_exists(self):
        missing = [ref for ref in _static_references() if not finders.find(ref)]
        self.assertEqual(missing, [])
