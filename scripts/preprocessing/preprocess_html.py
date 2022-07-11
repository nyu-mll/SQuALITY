import os
import re
import bs4
import logging
import argparse
import traceback
import ipdb

from bs4 import BeautifulSoup, Doctype, Comment


PERMITTED_TAGS = [
    "html",
    "[document]",
    "p",
    "b",
    "i",
    "u",
    "hr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ol",
    "ul",
    "pre",
    "br",
]

#PERMITTED_TAGS = ['\n']

TAGS_TO_DECOMPOSE = [
    "img",
    "table",
    "head",
    "",
]

def get_initial_alt_text(node):
    if node.name != "img":
        return None
    ## Check that the alt text contains only a single letter
    if "alt" in node.attrs and len(node["alt"]) == 1 and node["alt"].isalpha():
        return node
    return None

def has_text(input_str):
    return input_str is not None and len(input_str.strip())

def get_text_nodes(input_node):
    if isinstance(input_node, bs4.element.NavigableString):
        return [input_node] if has_text(input_node.string) else []
    return input_node.find_all(string=has_text)

def add_letter_from_initial_img(node):
    """If the text has an initial/drop cap with alt text, then extract the letter from
    the initial and pre-pend it to the next node containing text."""
    found_letter_img = False
    letter_img = get_initial_alt_text(node)
    if letter_img is None:
        return False
    # Find the next node in the doctree to have text in it to prepend the initial to.
    node_to_change = None
    # Check current node.
    if len(node.find_all(string=has_text)):
        node_to_change = node.find_all(string=has_text)[0]
    # Check sibling nodes and subtrees.
    elif True in [len(get_text_nodes(s)) > 0 for s in node.next_siblings]:
        for s in node.next_siblings:
            text_nodes = get_text_nodes(s)
            if len(text_nodes):
                node_to_change = text_nodes[0]
                break
    # Check parent node.
    elif node.parent is not None and has_text(node.parent.string):
        node_to_change = node.parent
    # Check siblings of parent node and subtrees.
    elif node.parent is not None and True in [len(get_text_nodes(s)) > 0 for s in node.parent.next_siblings]:
        for s in node.parent.next_siblings:
            text_nodes = get_text_nodes(s)
            if len(text_nodes):
                node_to_change = text_nodes[0]
                break
    # Check grandparent node.
    elif node.parent.parent is not None and has_text(node.parent.parent.string):
        node_to_change = node.parent.parent
    # Check siblings of grandparent node and subtrees.
    elif node.parent.parent is not None and True in [len(get_text_nodes(s)) > 0 for s in node.parent.parent.next_siblings]:
        for s in node.parent.parent.next_siblings:
            text_nodes = get_text_nodes(s)
            if len(text_nodes):
                node_to_change = text_nodes[0]
                break

    if node_to_change is not None:
        if isinstance(node_to_change, bs4.element.NavigableString):
            node_to_change.replace_with(f"{letter_img['alt']}{node_to_change.string.strip()}")
            logging.warning(f"Changed node text! New string: {letter_img['alt']}{node_to_change.string.strip()}")
        else:
            node_to_change.string = f"{letter_img['alt']}{node_to_change.string.strip()}"
            logging.warning(f"Changed node text! New string: {letter_img['alt']}{node_to_change.string.strip()}")
        return True

    return False


def replace_tags_with_newline_2(text, tags_to_replace_with_nl):
    for tag_to_replace in tags_to_replace_with_nl:
        text = text.replace(tag_to_replace.encode("utf-8"), b" \n\n ")
    return text


def replace_tags_with_newline_1(text, tags_to_replace_with_nl):
    for tag_to_replace in tags_to_replace_with_nl:
        text = text.replace(tag_to_replace.encode("utf-8"), b"\n")
    return text


def replace_tags_with_space(text, tags_to_replace_with_space):
    for tag_to_replace in tags_to_replace_with_space:
        text = text.replace(tag_to_replace.encode("utf-8"), b" ")
    return text


def replace_tags_with_blank(text, tags_to_replace_with_space):
    for tag_to_replace in tags_to_replace_with_space:
        text = text.replace(tag_to_replace.encode("utf-8"), b"")
    return text


def strip_html(
    args,
    filename, permitted_tags=PERMITTED_TAGS, tags_to_decompose=TAGS_TO_DECOMPOSE,
):
    """Decomposes all tags in tags_to_decompose and unwraps all other tags that
       are **not** in permitted_tags. Returns a BeautifulSoup object.
    """
    permitted_tags = set(permitted_tags)
    tags_to_decompose = set(tags_to_decompose)

    with open(filename, "rb") as fp:
        # NOTE: order matters b/c of duplicated <br/> and <p> tags
        fp_contents = fp.read()

        # replace newline+ascii char with space to avoid excessive linebreaks
        #locs = re.findall(b'\n[a-zA-Z]', fp_contents)
        #for loc in locs:
        #    fp_contents = fp_contents.replace(loc, b' ' + loc[-1])

        #fp_contents = replace_tags_with_newline_1(fp_contents, ['<hr class="chap"/>'])

        fp_contents = replace_tags_with_space(fp_contents, ['\n  <i>\n  ', '\n  </i>\n  ',
                                                            '\n  <br/>\n  ', '\n   <br/>\n  ',
                                                            ' </p>\n <p>',
                                                            ])

        # NOTE(AW): needs to be after above to catch </p><p> case
        fp_contents = replace_tags_with_newline_1(fp_contents, ['<p>\n', '</p>\n',
                                                                '<br/><br/>', '<br/> <br/>',
                                                                '\n</p><p>\n', #'\n</p> <p>'
                                                                '<hr class="tb"/>',
                                                               ])

        fp_contents = replace_tags_with_space(fp_contents, args.tags_to_replace_with_space)
        soup = BeautifulSoup(fp_contents, "html.parser")

    if args.strip_initial:
        # First look for initials/drop caps, extract the letter, and insert the letter into
        # the next node containing text.
        nodes = [soup]
        while len(nodes) > 0:
            curr = nodes.pop()
            if not isinstance(curr, bs4.element.Tag):
                continue
            if add_letter_from_initial_img(curr):
                logging.warning(f"Found initial in {filename}.")
            nodes += list(curr.children)

    # Traverse the rest of the tree via BFS; remove or unwrap extraneous tags.
    nodes = [soup]
    # Use a while loop instead of a for loop because |nodes| is changing
    # as we traverse the tree.
    while len(nodes) > 0:
        curr = nodes.pop()
        i = 0
        while i < len(curr.contents):
            # filter out extra comments
            child = curr.contents[i]
            if isinstance(child, Doctype) or isinstance(child, Comment) or isinstance(child, bs4.element.ProcessingInstruction):
                child.extract()
            child = curr.contents[i]
            if child is None or child.name is None:
                i += 1
                continue
            if child.name in tags_to_decompose:
                child.decompose()
                i -= 1
            elif child.name not in permitted_tags:
                parent = child.parent
                child.unwrap()
                i -= 1
            nodes.append(child)
            i += 1

    return soup


def main():
    parser = argparse.ArgumentParser(description="Preprocess HTML files.")
    parser.add_argument(
        "input_dir", type=str, help="Directory containing HTML files to pre-process.",
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to output pre-processed HTML files to.",
    )
    parser.add_argument(
        "--strip_initial", dest="strip_initial", type=bool, default=False, help="Whether to strip initial/drop cap images.",
    )
    parser.add_argument(
        "--tags_to_replace_with_space", nargs='*',
        default=['<br/>'],
        help="Which tags to replace with a space.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="data",
        choices=["web", "data"],
        help="Which mode to strip the document in. 'web' mode will keep all permitted tags and decompose all tags in TAGS_TO_DECOMPOSE. 'data' mode will remove all HTML tags."
    )

    args = parser.parse_args()
    permitted_tags = [] if args.mode == "data" else PERMITTED_TAGS

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for path in os.listdir(input_dir):
        if path.endswith(".html"):
            try:
                soup = strip_html(args, os.path.join(input_dir, path), permitted_tags=permitted_tags)
            except Exception as e:
                logging.warning(f"Failed to preprocess {path}")
                traceback.print_exc()
                continue

            output_path = os.path.join(output_dir, path)
            #try:
            with open(output_path, "w") as write_f:
                #write_f.write(soup.prettify("utf-8"))
                #txt = str(soup.prettify("utf-8"))
                txt = soup.text.strip()
                #re.sub(b'\n[a-zA-Z]', b' ', txt)
                locs = re.findall('\n[a-zA-Z]', txt)
                for loc in locs:
                    txt = txt.replace(loc, f' {loc[-1]}')
                txt = txt.replace('  ', ' ') # handles italics tags
                write_f.write(txt)

            #except Exception as e:
            #    # BeautifulSoup uses recursion to traverse doc trees, and sometimes
            #    # exceeds Python's default recursion limit as a result.
            #    logging.warning(f"Failed to write {path}: {e}")
            #    # Delete file if it failed to write.
            #    if os.path.exists(output_path):
            #        os.remove(output_path)

if __name__ == "__main__":
    main()
