from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")

    students = [
        {"name": "Ehab Mansour", "matriculation": "645043"},
    ]

    projects = [
        {"name": "Project 1", "url_name": "project1:index"},
    ]

    context = {
        "students": students,
        "projects": projects,
    }

    return HttpResponse(template.render(context, request))
