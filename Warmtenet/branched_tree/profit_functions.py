

def calculate_revenue(index, points, buildings):
    panden = points.loc[index, 'panden']
    revenue = 0
    for pand in panden:
        price = buildings.loc[pand, 'threshold']
        amount = buildings.loc[pand, 'warmtevraag']
        revenue += price * amount

    return revenue


