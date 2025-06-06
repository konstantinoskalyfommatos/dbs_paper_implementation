import random
from pprint import pprint

def _serialize_query(table: str) -> dict[str, str]:
    """Returns the table as a dictionary.
    
    Example: 'name[The Hollow Bell Café], eatType[restaurant], priceRange[more than £30], familyFriendly[no]'
    """
    results = []
    table = table.strip()
    
    # Split by comma to get individual attribute-value pairs
    pairs = table.split(', ')
    
    dict_table = {}
    for pair in pairs:
        key = pair.split("[")[0].strip()
        value = pair.split("[")[1].replace("]", "").strip()
        dict_table[key] = value

    dict_table[random.choice([key for key in dict_table.keys() if key != 'name'])] = ''
    del dict_table[random.choice([key for key in dict_table.keys() if key != 'name'])]

    pprint(dict_table)

    for key, value in dict_table.items():
        results.append(f"<R>{key}<R>{value}<R>")

    return '<C>'.join(results)

query = "name[The Roasted Retreat], food[Indian], priceRange[cheap], customer rating[5 out of 5], area[city centre]"
print(_serialize_query(query))