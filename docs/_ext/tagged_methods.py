from sphinx.application import Sphinx
from sphinx.util import logging
import importlib
import inspect
import os
import re

logger = logging.getLogger(__name__)

def collect_tagged_methods(app):
    """Extracts tagged methods from the module and generates a structured page."""
    module_name = "pysampled"
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        logger.warning(f"Could not import module {module_name}")
        return
    
    tags = {}

    for _, obj in inspect.getmembers(mod, inspect.isclass):
        for method_name, method_obj in inspect.getmembers(obj, inspect.isfunction):
            doc = inspect.getdoc(method_obj) or ""
            first_sentence = get_first_sentence(doc)
            
            tag_matches = re.findall(r"@tag:\s*(.*?)\s*$", doc)  # Match tags with spaces
            if tag_matches:
                for tag in tag_matches:
                    formatted_tag = format_tag(tag)
                    if formatted_tag not in tags:
                        tags[formatted_tag] = []
                    tags[formatted_tag].append((f"{obj.__name__}.{method_name}", first_sentence))
    
    if not tags:
        logger.warning("No tagged methods found.")

    output = ["Tagged Methods\n==============\n"]
    for tag, methods in sorted(tags.items()):
        output.append(f"\n{tag}\n" + "-" * len(tag) + "\n")
        for method, desc in methods:
            output.append(f"- :func:`{method}` - {desc}")
            
    output_path = os.path.join(app.srcdir, "source", "tagged_methods.rst")
    with open(output_path, "w") as f:
        f.write("\n".join(output))

def get_first_sentence(docstring):
    """Extract the first sentence from a docstring."""
    if not docstring:
        return ""
    sentences = docstring.split(".")
    return sentences[0] + "." if sentences else ""

def format_tag(tag):
    """Format tag by capitalizing and replacing underscores with spaces."""
    return " ".join([word.capitalize() for word in tag.split()])

def setup(app: Sphinx):
    app.connect("builder-inited", collect_tagged_methods)