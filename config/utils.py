def get_membership(dictionary, parent=None, membership={}):
    """
    recursively maps the child-to-parent associations
    in a json (python dictionary) object.
    """
    for key in dictionary:
        membership[key] = parent
        if type(dictionary[key]) == dict:
            get_membership(dictionary[key], key, membership)
    return membership


def update_params(params, args, membership):
    """
    updates <params> with the key-value pairs in <args>.
    """
    for key in args:
        if key not in membership:
            continue
        
        tree = [key]
        while key not in params:
            key = membership[key]
            tree.append(key)
        
        d = params
        for i in range(len(tree)-1, 0, -1):
            d = d[tree[i]]
        d[tree[0]] = args[tree[0]]
    
    return params
