"""Utility functions and classes"""


def _uniquify(seq, sep='-'):

    """Uniquify a list of strings.

    Adding unique numbers to duplicate values.

    Parameters
    ----------
    seq : `list` or `array-like`
        A list of values
    sep : `str`
        Separator

    Returns
    -------
    seq: `list` or `array-like`
        A list of updated values
    """
    dups = {}

    for i, val in enumerate(seq):
        if val not in dups:
            # Store index of first occurrence and occurrence value
            dups[val] = [i, 1]
        else:
            # Increment occurrence value, index value doesn't matter anymore
            dups[val][1] += 1

            # Use stored occurrence value
            seq[i] += (sep+str(dups[val][1]))

    return(seq)
