"""
Sphinx extension to add ReadTheDocs-style "Edit on GitHub" links to the
sidebar.
"""

import os
import warnings

__licence__ = "BSD (3 clause)"


# def get_github_repo(app, path):
#     if path.endswith(".ipynb"):
#         return app.config.github_nb_repo, "/"
#     return app.config.github_repo, "/docs/source/"


def html_page_context(app, pagename, templatename, context, doctree):
    if templatename != "page.html":
        return

    if doctree is not None:
        path = os.path.relpath(doctree.get("source"), app.builder.srcdir)
        if path.endswith(".ipynb"):
            context["display_github"] = True
            context["github_user"] = "huidongchen"
            context["github_repo"] = "simba_tutorials"
            context["github_version"] = "main"
            if path.endswith("rna_10x_mouse_brain_1p3M.ipynb"):
                context["conf_py_path"] = "/v1.1/"
            else:
                context["conf_py_path"] = "/v1.0/"
        else:
            context["display_github"] = True
            context["github_user"] = "pinellolab"
            context["github_repo"] = "simba"
            context["github_version"] = "master"
            context["conf_py_path"] = "/docs/source/"

def setup(app):
    app.add_config_value("github_nb_repo", "", True)
    app.add_config_value("github_repo", "", True)
    app.connect("html-page-context", html_page_context)
