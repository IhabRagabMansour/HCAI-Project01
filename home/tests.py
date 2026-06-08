from django.test import TestCase


class RootUrlTest(TestCase):
    def test_root_redirects_to_home(self):
        response = self.client.get("/")
        self.assertRedirects(response, "/home/", fetch_redirect_response=False)
