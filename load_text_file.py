def load_from_text():
    """
    Loads text file where the first line is current values
    Rest of the lines are voltage traces.
    """
    import numpy as np
    with open('camp_proj.txt','r') as f:
        lines = f.readlines()
    values = []
    for line in lines:
        val=line.split(',')[:-2]
        val = NP.array(val).astype(float)
        values.append(val)
    stim = values[0]
    return stim,np.stack(values[1:])