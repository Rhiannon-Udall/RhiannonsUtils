def make_header_block(
    header : str
):
    """Make a header block which is nicely formatted, for section headers in code
    
    Parameters
    ==========
    header : str
        The contents of the header block
    """
    base = "#"*80 + "\n"
    if len(header) % 2 == 0:
        header = " "*4 + header + " "*4
    else:
        header = " "*3 + header + " "*4
    pad = int((80 - len(header)) / 2) * "#"
    block = base * 2 + pad + header + pad + "\n" + base * 2
    return block