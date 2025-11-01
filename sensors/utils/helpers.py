def normalize(data, min_val=0.0, max_val=1.0):
    """
    Normalizes a list of numbers to a given range.
    """
    if not data:
        return []
    min_data = min(data)
    max_data = max(data)
    if max_data == min_data:
        return [min_val for _ in data]
    return [
        ((x - min_data) / (max_data - min_data)) * (max_val - min_val) + min_val
        for x in data
    ]
