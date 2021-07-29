

def calculate_revenue(index, points, buildings):
    panden = points.loc[index, 'panden']
    revenue = 0
    for pand in panden:
        price = buildings.loc[buildings.pandidentificatie == pand, 'threshold'].values[0]
        amount = buildings.loc[buildings.pandidentificatie == pand, 'warmtevraag'].values[0]
        revenue += price * amount

    return revenue


