from shapely.geometry import Point, LineString


def cut(line, distance):
    """Cuts a line in two at a distance from its starting point"""
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

def split_line_with_points(line, points):
    """Splits a line string in several segments considering a list of points.

    The points used to cut the line are assumed to be in the line string
    and given in the order of appearance they have in the line string.
    ['LINESTRING (1 2, 8 7, 4 5, 2 4)', 'LINESTRING (2 4, 4 7, 8 5, 9 18)', 'LINESTRING (9 18, 1 2, 12 7, 4 5, 6 5)', 'LINESTRING (6 5, 4 9)']

    """
    segments = []
    current_line = line
    for p in points:
        d = current_line.project(p)
        list_of_segments = cut(current_line, d)
        if len(list_of_segments) == 2:
            seg, current_line = list_of_segments[0], list_of_segments[1]
            segments.append(seg)
        else:
            current_line = list_of_segments[0]
    segments.append(current_line)
    return segments