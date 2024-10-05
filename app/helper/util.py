import re

def create_reference(citation: str) -> str:
    """Takes an IEEE citation and converts it into jumpable HTML elements.
    Key assumptions: Starts with the citation number, [<number>], contains an https:// url

    Args:
        citation (str): an IEEE citation, e.g. [1] J. Martínez Torres, C. Iglesias Comesaña and P. J. García-Nieto, "Review: machine learning techniques applied to cybersecurity," International Journal of Machine Learning and Cybernetics, vol. 10, (10), pp. 2823-2836, 2019. Available: https://www.proquest.com/scholarly-journals/review-machine-learning-techniques-applied/docview/2920238591/se-2. [Accessed Oct. 03, 2024]')
    Returns:
        str: a paragraph HTML element of the converted citation with appropriate links
    """
    citation_number = int(re.search(r'\[(\d+)\]', citation).group(1))
    url = re.search(r'https://[^\s]+', citation).group()
    
    if url.endswith('.'):
        url = url[:-1]

    anchor_reference = f'<a name=ref{citation_number}>[{citation_number}]</a>'
    anchor_link = f'<a href={url}/>{url}</a>'

    citation_html = "<p>" + citation.replace(f"[{citation_number}]", anchor_reference).replace(url, anchor_link) + "</p>"
    return citation_html

def convert_in_text_citations(paragraph: str) -> str:
    """Replaces all IEEE in-text citations with markdown links to anchor tags

    Args:
        paragraph (str): a paragraph to replace in-text citations, e.g. text [2]

    Returns:
        str: a modified paragraph with markdown links where in-text citations were, e.g. text [[2]](#ref2)
    """
    matches = re.findall(r'(\[\d+\])', paragraph)
    
    converted_paragraph = paragraph
    for match in set(matches):
        converted_paragraph = converted_paragraph.replace(match, f"[{match}](#ref{match[1:-1]})")

    return converted_paragraph
