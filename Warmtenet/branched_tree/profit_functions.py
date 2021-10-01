

def calculate_revenue(index, panden, buildings):
    panden = points.loc[index, 'panden']
    revenue = 0
    for pand in panden:
        amount = buildings.loc[buildings.pandidentificatie == pand, 'warmtevraag'].values[0]
        revenue += price * amount

    return revenue


