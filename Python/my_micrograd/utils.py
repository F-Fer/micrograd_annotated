

def get_list_dimensions(lst):
    dimensions = 0
    while isinstance(lst, list):
        dimensions += 1
        if len(lst) > 0:
            lst = lst[0]  # Dive into the next dimension
        else:
            break  # Stop at an empty list
    return dimensions