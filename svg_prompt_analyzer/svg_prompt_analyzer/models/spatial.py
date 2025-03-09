"""
Spatial Model Module
==================
This module defines spatial relationship enumerations and utilities.
"""

from enum import Enum


class SpatialRelation(Enum):
    """Enumeration of spatial relationships between objects."""
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    INSIDE = "inside"
    OUTSIDE = "outside"
    AROUND = "around"
    BETWEEN = "between"
    ON = "on"
    UNDER = "under"
    OVER = "over"
    NEAR = "near"
    FAR = "far"
    CENTER = "center"
    EDGE = "edge"
    LAYERED = "layered"
    # Additional relations for enhanced spatial handling
    TOWARD = "toward"           # For "facing" relationships
    CONNECTED = "connected"     # For "connected by" relationships 
    INTERWOVEN = "interwoven"   # For "interwoven with" relationships
    LANDSCAPE = "landscape"     # For landscape-specific relationships
    WIDE = "wide"               # For "expanse" relationships
    UNKNOWN = "unknown"


# Dictionary mapping spatial keywords to relations
SPATIAL_KEYWORDS = {
    SpatialRelation.ABOVE: ["above", "over", "atop", "overhead"],
    SpatialRelation.BELOW: ["below", "under", "underneath", "beneath"],
    SpatialRelation.LEFT: ["left", "leftward", "west"],
    SpatialRelation.RIGHT: ["right", "rightward", "east"],
    SpatialRelation.INSIDE: ["inside", "within", "in", "into"],
    SpatialRelation.OUTSIDE: ["outside", "out", "outer"],
    SpatialRelation.AROUND: ["around", "surrounding", "encircling", "enclosing"],
    SpatialRelation.BETWEEN: ["between", "amid", "amidst", "among"],
    SpatialRelation.ON: ["on", "upon", "atop", "onto"],
    SpatialRelation.UNDER: ["under", "beneath", "below", "underneath"],
    SpatialRelation.OVER: ["over", "above", "atop", "overhead"],
    SpatialRelation.NEAR: ["near", "close", "beside", "by", "next to", "adjacent"],
    SpatialRelation.FAR: ["far", "distant", "away", "remote"],
    SpatialRelation.CENTER: ["center", "middle", "central", "centered"],
    SpatialRelation.EDGE: ["edge", "border", "perimeter", "margin"],
    SpatialRelation.LAYERED: ["layered", "stacked", "overlapping", "overlaid"],
    SpatialRelation.TOWARD: ["toward", "towards", "facing"],
    SpatialRelation.CONNECTED: ["connected", "linked", "joined", "attached"],
    SpatialRelation.INTERWOVEN: ["interwoven", "intertwined", "woven"],
    SpatialRelation.LANDSCAPE: ["vista", "vistas", "panorama"],
    SpatialRelation.WIDE: ["wide", "broad", "expanse", "expansive"]
}