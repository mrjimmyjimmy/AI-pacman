def nearestPoint( pos ):
    """
    Finds the nearest grid point to a position (discretizes).
    """
    ( current_row, current_col ) = pos

    grid_row = int( current_row + 0.5 )
    grid_col = int( current_col + 0.5 )
    return ( grid_row, grid_col )
if -9999999 > - float('inf'):
    print float('inf')