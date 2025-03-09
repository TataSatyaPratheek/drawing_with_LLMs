"""
Visual Validator Module
===================
This module provides validation for visual scene layouts to ensure
spatial coherence, object relationships, and design principles.
"""

import math
import logging
from typing import Dict, Any, List, Tuple, Optional, Set

from svg_prompt_analyzer.models.scene import Scene
from svg_prompt_analyzer.models.visual_object import VisualObject, ObjectType
from svg_prompt_analyzer.models.shape import Shape
from svg_prompt_analyzer.models.spatial import SpatialRelation

logger = logging.getLogger(__name__)


class VisualValidator:
    """
    Validator for ensuring visual coherence of scenes.
    Checks for spatial inconsistencies, physical impossibilities, and design principles.
    """
    
    def __init__(self, 
                 enforce_physical_constraints: bool = True,
                 enforce_design_principles: bool = True,
                 auto_fix: bool = True):
        """
        Initialize the visual validator.
        
        Args:
            enforce_physical_constraints: Whether to enforce physical constraints
            enforce_design_principles: Whether to enforce design principles
            auto_fix: Whether to automatically fix issues
        """
        self.enforce_physical_constraints = enforce_physical_constraints
        self.enforce_design_principles = enforce_design_principles
        self.auto_fix = auto_fix
        
    def validate_scene(self, scene: Scene) -> Tuple[bool, List[Dict[str, Any]], Scene]:
        """
        Validate a scene for visual coherence.
        
        Args:
            scene: Scene to validate
            
        Returns:
            Tuple of (is_valid, issues, fixed_scene)
        """
        issues = []
        
        # Create a working copy of the scene
        working_scene = Scene(
            id=scene.id,
            prompt=scene.prompt,
            background_color=scene.background_color,
            objects=scene.objects.copy(),
            width=scene.width,
            height=scene.height,
            patterns=scene.patterns.copy() if scene.patterns else {},
            defs=scene.defs.copy() if scene.defs else [],
            special_elements=scene.special_elements.copy() if hasattr(scene, 'special_elements') else []
        )
        
        # 1. Validate spatial relationships
        spatial_issues = self._validate_spatial_relationships(working_scene)
        issues.extend(spatial_issues)
        
        # 2. Validate physical constraints
        if self.enforce_physical_constraints:
            physical_issues = self._validate_physical_constraints(working_scene)
            issues.extend(physical_issues)
        
        # 3. Validate design principles
        if self.enforce_design_principles:
            design_issues = self._validate_design_principles(working_scene)
            issues.extend(design_issues)
        
        # 4. Validate object completeness
        completeness_issues = self._validate_object_completeness(working_scene)
        issues.extend(completeness_issues)
        
        # 5. Validate z-index coherence
        z_index_issues = self._validate_z_index_coherence(working_scene)
        issues.extend(z_index_issues)
        
        # Auto-fix issues if enabled
        if self.auto_fix and issues:
            logger.info(f"Auto-fixing {len(issues)} visual validation issues")
            working_scene = self._auto_fix_issues(working_scene, issues)
        
        # Scene is valid if there are no remaining issues
        is_valid = len(issues) == 0
        
        return is_valid, issues, working_scene
    
    def _validate_spatial_relationships(self, scene: Scene) -> List[Dict[str, Any]]:
        """
        Validate spatial relationships between objects.
        
        Args:
            scene: Scene to validate
            
        Returns:
            List of spatial relationship issues
        """
        issues = []
        
        # Skip validation if there are not enough objects
        if len(scene.objects) <= 1:
            return issues
        
        # Check for overlapping objects (unless they have a nesting relationship)
        for i, obj1 in enumerate(scene.objects):
            for j, obj2 in enumerate(scene.objects[i+1:], i+1):
                # Skip validation if either object has no position or size
                if not hasattr(obj1, 'position') or not hasattr(obj2, 'position'):
                    continue
                    
                if not hasattr(obj1, 'size') or not hasattr(obj2, 'size'):
                    continue
                
                # Calculate distance between objects
                distance = math.hypot(
                    obj1.position[0] - obj2.position[0],
                    obj1.position[1] - obj2.position[1]
                )
                
                # Calculate overlap
                combined_radius = (obj1.size + obj2.size) / 2
                overlap = combined_radius - distance
                
                # Check if objects are overlapping
                if overlap > 0.01:  # Small threshold to allow minor overlaps
                    # Check if one object is meant to contain the other
                    contains_relationship = False
                    
                    # Object 1 contains Object 2
                    if obj1.size > obj2.size * 1.5 and distance < obj1.size / 2:
                        contains_relationship = True
                        
                        # Ensure proper z-ordering
                        if obj1.z_index >= obj2.z_index:
                            issues.append({
                                "type": "z_index_inconsistency",
                                "description": f"Object {obj1.name} contains {obj2.name} but has higher or equal z-index",
                                "severity": "medium",
                                "objects": [obj1.id, obj2.id],
                                "fix": "adjust_z_index"
                            })
                    
                    # Object 2 contains Object 1
                    elif obj2.size > obj1.size * 1.5 and distance < obj2.size / 2:
                        contains_relationship = True
                        
                        # Ensure proper z-ordering
                        if obj2.z_index >= obj1.z_index:
                            issues.append({
                                "type": "z_index_inconsistency",
                                "description": f"Object {obj2.name} contains {obj1.name} but has higher or equal z-index",
                                "severity": "medium",
                                "objects": [obj1.id, obj2.id],
                                "fix": "adjust_z_index"
                            })
                    
                    # If not a containment relationship, mark as inappropriate overlap
                    if not contains_relationship:
                        issues.append({
                            "type": "inappropriate_overlap",
                            "description": f"Objects {obj1.name} and {obj2.name} inappropriately overlap",
                            "severity": "medium",
                            "objects": [obj1.id, obj2.id],
                            "overlap_amount": overlap,
                            "fix": "separate_objects"
                        })
        
        # Check for objects outside the scene bounds
        for obj in scene.objects:
            # Skip validation if object has no position or size
            if not hasattr(obj, 'position') or not hasattr(obj, 'size'):
                continue
                
            # Calculate object bounds in normalized coordinates (0-1)
            left = obj.position[0] - obj.size / 2
            right = obj.position[0] + obj.size / 2
            top = obj.position[1] - obj.size / 2
            bottom = obj.position[1] + obj.size / 2
            
            # Check if object is outside scene bounds
            if left < 0 or right > 1 or top < 0 or bottom > 1:
                issues.append({
                    "type": "out_of_bounds",
                    "description": f"Object {obj.name} is outside scene bounds",
                    "severity": "high",
                    "object": obj.id,
                    "position": obj.position,
                    "size": obj.size,
                    "fix": "adjust_position"
                })
        
        return issues
    
    def _validate_physical_constraints(self, scene: Scene) -> List[Dict[str, Any]]:
        """
        Validate physical constraints of the scene.
        
        Args:
            scene: Scene to validate
            
        Returns:
            List of physical constraint issues
        """
        issues = []
        
        # Look for physically impossible object arrangements
        
        # Check for floating objects that should be supported
        ground_bound_types = {
            ObjectType.ARCHITECTURE, 
            ObjectType.NATURE
        }
        
        ground_level = 0.8  # Normalized y-coordinate for ground level
        
        # Get objects that are on the ground
        ground_objects = []
        for obj in scene.objects:
            # Skip validation if object has no position, size, or type
            if not hasattr(obj, 'position') or not hasattr(obj, 'size') or not hasattr(obj, 'object_type'):
                continue
                
            # Check if object is a ground object
            if obj.object_type in ground_bound_types:
                # Check if object bottom is near ground level
                obj_bottom = obj.position[1] + obj.size / 2
                
                if abs(obj_bottom - ground_level) < 0.1:
                    ground_objects.append(obj)
        
        # Check for objects that should be on the ground but aren't
        for obj in scene.objects:
            # Skip validation if object has no position, size, or type
            if not hasattr(obj, 'position') or not hasattr(obj, 'size') or not hasattr(obj, 'object_type'):
                continue
                
            # Skip objects that aren't ground-bound types
            if obj.object_type not in ground_bound_types:
                continue
            
            # Check if the object is floating
            obj_bottom = obj.position[1] + obj.size / 2
            
            if abs(obj_bottom - ground_level) > 0.1:
                # Check if it's supported by another object
                is_supported = False
                
                for ground_obj in ground_objects:
                    # Skip validation if ground object has no position or size
                    if not hasattr(ground_obj, 'position') or not hasattr(ground_obj, 'size'):
                        continue
                        
                    # Check if object is above and in contact with ground object
                    horizontal_distance = abs(obj.position[0] - ground_obj.position[0])
                    vertical_distance = abs(obj.position[1] - ground_obj.position[1])
                    
                    if (horizontal_distance < (obj.size + ground_obj.size) / 2 and
                        vertical_distance < (obj.size + ground_obj.size) / 2):
                        is_supported = True
                        break
                
                if not is_supported and "float" not in obj.name.lower():
                    issues.append({
                        "type": "floating_object",
                        "description": f"Object {obj.name} is floating without support",
                        "severity": "medium",
                        "object": obj.id,
                        "fix": "ground_object"
                    })
        
        # Check for size inconsistencies
        object_size_ranges = {
            ObjectType.ARCHITECTURE: (0.3, 0.7),  # Buildings are large
            ObjectType.NATURE: (0.1, 0.6),  # Trees, mountains vary in size
            ObjectType.GEOMETRIC: (0.05, 0.3),  # Geometric shapes are usually smaller
            ObjectType.CELESTIAL: (0.1, 0.5),  # Sun, moon, stars are medium-sized
            ObjectType.CLOTHING: (0.2, 0.5),  # Clothing items are medium-sized
        }
        
        for obj in scene.objects:
            # Skip validation if object has no size or type
            if not hasattr(obj, 'size') or not hasattr(obj, 'object_type'):
                continue
                
            # Check if object size is within reasonable range
            if obj.object_type in object_size_ranges:
                min_size, max_size = object_size_ranges[obj.object_type]
                
                if obj.size < min_size:
                    issues.append({
                        "type": "object_too_small",
                        "description": f"Object {obj.name} is unnaturally small for its type",
                        "severity": "low",
                        "object": obj.id,
                        "current_size": obj.size,
                        "recommended_size": min_size,
                        "fix": "adjust_size"
                    })
                elif obj.size > max_size:
                    issues.append({
                        "type": "object_too_large",
                        "description": f"Object {obj.name} is unnaturally large for its type",
                        "severity": "low",
                        "object": obj.id,
                        "current_size": obj.size,
                        "recommended_size": max_size,
                        "fix": "adjust_size"
                    })
        
        return issues
    
    def _validate_design_principles(self, scene: Scene) -> List[Dict[str, Any]]:
        """
        Validate design principles of the scene.
        
        Args:
            scene: Scene to validate
            
        Returns:
            List of design principle issues
        """
        issues = []
        
        # Calculate scene center
        scene_center = (0.5, 0.5)
        
        # Calculate visual weight distribution
        left_weight = 0
        right_weight = 0
        top_weight = 0
        bottom_weight = 0
        
        # Track object presence in quadrants
        quadrants = [0, 0, 0, 0]  # top-left, top-right, bottom-left, bottom-right
        
        for obj in scene.objects:
            # Skip validation if object has no position or size
            if not hasattr(obj, 'position') or not hasattr(obj, 'size'):
                continue
                
            # Calculate distance from center
            dx = obj.position[0] - scene_center[0]
            dy = obj.position[1] - scene_center[1]
            
            # Calculate visual weight (size squared × importance factor)
            importance_factor = 1.0
            if hasattr(obj, 'z_index'):
                importance_factor = 1.0 + obj.z_index * 0.1
                
            visual_weight = obj.size * obj.size * importance_factor
            
            # Distribute weight
            if dx < 0:
                left_weight += visual_weight
            else:
                right_weight += visual_weight
                
            if dy < 0:
                top_weight += visual_weight
            else:
                bottom_weight += visual_weight
                
            # Track quadrant presence
            quadrant_idx = (1 if dx >= 0 else 0) + (2 if dy >= 0 else 0)
            quadrants[quadrant_idx] += 1
        
        # Check for balance issues
        horizontal_imbalance = abs(left_weight - right_weight) / max(0.001, left_weight + right_weight)
        vertical_imbalance = abs(top_weight - bottom_weight) / max(0.001, top_weight + bottom_weight)
        
        # If imbalance exceeds threshold, add issue
        balance_threshold = 0.5  # Allow up to 50% imbalance
        if horizontal_imbalance > balance_threshold:
            issues.append({
                "type": "horizontal_imbalance",
                "description": "Scene is horizontally imbalanced",
                "severity": "low",
                "left_weight": left_weight,
                "right_weight": right_weight,
                "imbalance": horizontal_imbalance,
                "fix": "adjust_balance"
            })
            
        if vertical_imbalance > balance_threshold:
            issues.append({
                "type": "vertical_imbalance",
                "description": "Scene is vertically imbalanced",
                "severity": "low",
                "top_weight": top_weight,
                "bottom_weight": bottom_weight,
                "imbalance": vertical_imbalance,
                "fix": "adjust_balance"
            })
        
        # Check for empty quadrants
        empty_quadrants = [i for i, count in enumerate(quadrants) if count == 0]
        if empty_quadrants and len(scene.objects) > 2:  # Only care about empties if we have more than 2 objects
            issues.append({
                "type": "empty_quadrants",
                "description": "Scene has empty quadrants",
                "severity": "low",
                "empty_quadrants": empty_quadrants,
                "fix": "distribute_objects"
            })
        
        # Check for edge crowding (too many objects near edges)
        edge_threshold = 0.1  # Normalized distance from edge
        edge_objects = 0
        
        for obj in scene.objects:
            # Skip validation if object has no position or size
            if not hasattr(obj, 'position') or not hasattr(obj, 'size'):
                continue
                
            # Check distance from edges
            distance_to_edge = min(
                obj.position[0],               # Distance to left edge
                1 - obj.position[0],           # Distance to right edge
                obj.position[1],               # Distance to top edge
                1 - obj.position[1]            # Distance to bottom edge
            )
            
            if distance_to_edge < edge_threshold:
                edge_objects += 1
        
        # If too many objects are near edges, add issue
        edge_threshold_pct = 0.5  # Max 50% of objects should be at edges
        if edge_objects > len(scene.objects) * edge_threshold_pct and len(scene.objects) > 2:
            issues.append({
                "type": "edge_crowding",
                "description": "Too many objects are crowded near the edges",
                "severity": "low",
                "edge_objects": edge_objects,
                "total_objects": len(scene.objects),
                "fix": "adjust_edge_objects"
            })
        
        return issues
    
    def _validate_object_completeness(self, scene: Scene) -> List[Dict[str, Any]]:
        """
        Validate completeness of objects in the scene.
        
        Args:
            scene: Scene to validate
            
        Returns:
            List of object completeness issues
        """
        issues = []
        
        for obj in scene.objects:
            # Check for missing color
            if not obj.color:
                issues.append({
                    "type": "missing_color",
                    "description": f"Object {obj.name} has no color",
                    "severity": "medium",
                    "object": obj.id,
                    "fix": "add_color"
                })
            
            # Check for missing shapes
            if not obj.shapes or len(obj.shapes) == 0:
                issues.append({
                    "type": "missing_shape",
                    "description": f"Object {obj.name} has no shape",
                    "severity": "high",
                    "object": obj.id,
                    "fix": "add_shape"
                })
        
        return issues
    
    def _validate_z_index_coherence(self, scene: Scene) -> List[Dict[str, Any]]:
        """
        Validate z-index coherence of objects in the scene.
        
        Args:
            scene: Scene to validate
            
        Returns:
            List of z-index coherence issues
        """
        issues = []
        
        # Skip validation if there are not enough objects
        if len(scene.objects) <= 1:
            return issues
        
        # Track overlapping objects
        for i, obj1 in enumerate(scene.objects):
            for j, obj2 in enumerate(scene.objects[i+1:], i+1):
                # Skip validation if either object has no position or size
                if not hasattr(obj1, 'position') or not hasattr(obj2, 'position'):
                    continue
                    
                if not hasattr(obj1, 'size') or not hasattr(obj2, 'size'):
                    continue
                
                # Calculate distance between objects
                distance = math.hypot(
                    obj1.position[0] - obj2.position[0],
                    obj1.position[1] - obj2.position[1]
                )
                
                # Calculate overlap
                combined_radius = (obj1.size + obj2.size) / 2
                overlap = combined_radius - distance
                
                # If objects overlap, check z-index consistency
                if overlap > 0:
                    # Get spatial relationship
                    # Y-coordinates in SVG increase downward, so lower y is "above"
                    obj1_above = obj1.position[1] < obj2.position[1]
                    
                    # Check if z-index is consistent with vertical position
                    if obj1_above and obj1.z_index <= obj2.z_index:
                        issues.append({
                            "type": "z_index_inconsistency",
                            "description": f"Object {obj1.name} is visually above {obj2.name} but has lower or equal z-index",
                            "severity": "medium",
                            "objects": [obj1.id, obj2.id],
                            "fix": "adjust_z_index"
                        })
                    elif not obj1_above and obj1.z_index >= obj2.z_index:
                        issues.append({
                            "type": "z_index_inconsistency",
                            "description": f"Object {obj1.name} is visually below {obj2.name} but has higher or equal z-index",
                            "severity": "medium",
                            "objects": [obj1.id, obj2.id],
                            "fix": "adjust_z_index"
                        })
        
        return issues
    
    def _auto_fix_issues(self, scene: Scene, issues: List[Dict[str, Any]]) -> Scene:
        """
        Automatically fix issues in the scene.
        
        Args:
            scene: Scene to fix
            issues: List of issues to fix
            
        Returns:
            Fixed scene
        """
        # Create a working copy of the scene
        fixed_scene = Scene(
            id=scene.id,
            prompt=scene.prompt,
            background_color=scene.background_color,
            objects=scene.objects.copy(),
            width=scene.width,
            height=scene.height,
            patterns=scene.patterns.copy() if scene.patterns else {},
            defs=scene.defs.copy() if scene.defs else [],
            special_elements=scene.special_elements.copy() if hasattr(scene, 'special_elements') else []
        )
        
        # Create id-to-object mapping for quick lookup
        object_map = {obj.id: obj for obj in fixed_scene.objects}
        
        # Apply fixes for each issue type
        for issue in issues:
            issue_type = issue.get("type")
            fix_type = issue.get("fix")
            
            if issue_type == "inappropriate_overlap" and fix_type == "separate_objects":
                # Get objects involved
                obj1_id, obj2_id = issue.get("objects", [None, None])
                obj1 = object_map.get(obj1_id)
                obj2 = object_map.get(obj2_id)
                
                if obj1 and obj2:
                    # Calculate vector between objects
                    dx = obj2.position[0] - obj1.position[0]
                    dy = obj2.position[1] - obj1.position[1]
                    distance = math.hypot(dx, dy)
                    
                    # Normalize vector
                    if distance > 0:
                        dx /= distance
                        dy /= distance
                    else:
                        # If objects are exactly at same position, move arbitrarily
                        dx, dy = 1, 0
                    
                    # Calculate required separation
                    overlap = issue.get("overlap_amount", 0)
                    separation = overlap * 1.1  # Add a small buffer
                    
                    # Move objects apart
                    obj1.position = (
                        obj1.position[0] - dx * separation / 2,
                        obj1.position[1] - dy * separation / 2
                    )
                    
                    obj2.position = (
                        obj2.position[0] + dx * separation / 2,
                        obj2.position[1] + dy * separation / 2
                    )
            
            elif issue_type == "out_of_bounds" and fix_type == "adjust_position":
                # Get object involved
                obj_id = issue.get("object")
                obj = object_map.get(obj_id)
                
                if obj:
                    # Get current position and size
                    x, y = obj.position
                    size = obj.size
                    
                    # Calculate bounds
                    left = x - size / 2
                    right = x + size / 2
                    top = y - size / 2
                    bottom = y + size / 2
                    
                    # Adjust position to bring within bounds
                    if left < 0:
                        x += -left
                    elif right > 1:
                        x -= (right - 1)
                        
                    if top < 0:
                        y += -top
                    elif bottom > 1:
                        y -= (bottom - 1)
                        
                    # Update position
                    obj.position = (x, y)
            
            elif issue_type == "z_index_inconsistency" and fix_type == "adjust_z_index":
                # Get objects involved
                obj_ids = issue.get("objects", [])
                
                if len(obj_ids) == 2:
                    obj1_id, obj2_id = obj_ids
                    obj1 = object_map.get(obj1_id)
                    obj2 = object_map.get(obj2_id)
                    
                    if obj1 and obj2:
                        # Swap z-indexes
                        temp_z = obj1.z_index
                        obj1.z_index = obj2.z_index + 1  # Ensure no overlap
                        obj2.z_index = temp_z - 1
            
            elif issue_type == "floating_object" and fix_type == "ground_object":
                # Get object involved
                obj_id = issue.get("object")
                obj = object_map.get(obj_id)
                
                if obj:
                    # Move to ground level
                    ground_y = 0.8  # Normalized y-coordinate for ground
                    new_y = ground_y - obj.size / 2
                    
                    # Update position
                    obj.position = (obj.position[0], new_y)
            
            elif (issue_type == "object_too_small" or issue_type == "object_too_large") and fix_type == "adjust_size":
                # Get object involved
                obj_id = issue.get("object")
                obj = object_map.get(obj_id)
                
                if obj:
                    # Get recommended size
                    recommended_size = issue.get("recommended_size")
                    
                    if recommended_size:
                        # Update size
                        obj.size = recommended_size
            
            elif issue_type == "missing_color" and fix_type == "add_color":
                # Get object involved
                obj_id = issue.get("object")
                obj = object_map.get(obj_id)
                
                if obj:
                    # Add a default color based on object type
                    from svg_prompt_analyzer.models.color import Color
                    
                    # Choose color based on object type
                    color_map = {
                        ObjectType.NATURE: "green",
                        ObjectType.ARCHITECTURE: "gray",
                        ObjectType.GEOMETRIC: "blue",
                        ObjectType.CELESTIAL: "yellow",
                        ObjectType.CLOTHING: "red",
                        ObjectType.ABSTRACT: "purple"
                    }
                    
                    color_name = color_map.get(obj.object_type, "gray")
                    obj.color = Color(name=color_name)
            
            elif issue_type == "missing_shape" and fix_type == "add_shape":
                # Get object involved
                obj_id = issue.get("object")
                obj = object_map.get(obj_id)
                
                if obj:
                    # Add a default shape based on object type
                    from svg_prompt_analyzer.models.shape import Shape, Attribute
                    
                    # Choose shape based on object type
                    shape_map = {
                        ObjectType.NATURE: "triangle",  # Tree-like
                        ObjectType.ARCHITECTURE: "rectangle",  # Building-like
                        ObjectType.GEOMETRIC: "circle",  # Basic geometric
                        ObjectType.CELESTIAL: "circle",  # Sun, moon, stars
                        ObjectType.CLOTHING: "rectangle",  # Clothing item
                        ObjectType.ABSTRACT: "rectangle"  # Default
                    }
                    
                    shape_type = shape_map.get(obj.object_type, "rectangle")
                    
                    # Create shape
                    shape = Shape(
                        shape_type=shape_type,
                        attributes=[
                            Attribute("fill", obj.color.hex_code if obj.color else "#808080"),
                            Attribute("stroke", "#000000"),
                            Attribute("stroke-width", 1)
                        ]
                    )
                    
                    # Add shape to object
                    obj.shapes = [shape]
            
            elif issue_type == "horizontal_imbalance" and fix_type == "adjust_balance":
                # Get imbalance data
                left_weight = issue.get("left_weight", 0)
                right_weight = issue.get("right_weight", 0)
                
                # Determine which side is heavier
                left_heavier = left_weight > right_weight
                
                # Find movable objects on heavier side
                movable_objects = []
                for obj in fixed_scene.objects:
                    # Skip validation if object has no position
                    if not hasattr(obj, 'position'):
                        continue
                        
                    # Check if object is on heavier side
                    on_left = obj.position[0] < 0.5
                    
                    if on_left == left_heavier:
                        movable_objects.append(obj)
                
                # Sort by distance from center
                movable_objects.sort(key=lambda obj: abs(obj.position[0] - 0.5))
                
                # Move the most central object toward the lighter side
                if movable_objects:
                    obj = movable_objects[0]
                    # Move halfway to center
                    new_x = (obj.position[0] + 0.5) / 2
                    obj.position = (new_x, obj.position[1])
            
            elif issue_type == "vertical_imbalance" and fix_type == "adjust_balance":
                # Get imbalance data
                top_weight = issue.get("top_weight", 0)
                bottom_weight = issue.get("bottom_weight", 0)
                
                # Determine which side is heavier
                top_heavier = top_weight > bottom_weight
                
                # Find movable objects on heavier side
                movable_objects = []
                for obj in fixed_scene.objects:
                    # Skip validation if object has no position
                    if not hasattr(obj, 'position'):
                        continue
                        
                    # Check if object is on heavier side
                    on_top = obj.position[1] < 0.5
                    
                    if on_top == top_heavier:
                        movable_objects.append(obj)
                
                # Sort by distance from center
                movable_objects.sort(key=lambda obj: abs(obj.position[1] - 0.5))
                
                # Move the most central object toward the lighter side
                if movable_objects:
                    obj = movable_objects[0]
                    # Move halfway to center
                    new_y = (obj.position[1] + 0.5) / 2
                    obj.position = (obj.position[0], new_y)
            
            elif issue_type == "empty_quadrants" and fix_type == "distribute_objects":
                # Get empty quadrants
                empty_quadrants = issue.get("empty_quadrants", [])
                
                if empty_quadrants:
                    # Find smallest objects that could be moved
                    movable_objects = sorted(fixed_scene.objects, key=lambda obj: obj.size)
                    
                    # Move one object to each empty quadrant
                    for i, quadrant in enumerate(empty_quadrants):
                        if i < len(movable_objects):
                            obj = movable_objects[i]
                            
                            # Calculate quadrant center
                            quadrant_x = 0.25 if quadrant in [0, 2] else 0.75
                            quadrant_y = 0.25 if quadrant in [0, 1] else 0.75
                            
                            # Move object to quadrant
                            obj.position = (quadrant_x, quadrant_y)
            
            elif issue_type == "edge_crowding" and fix_type == "adjust_edge_objects":
                # Find objects near edges
                edge_threshold = 0.1  # Normalized distance from edge
                edge_objects = []
                
                for obj in fixed_scene.objects:
                    # Skip validation if object has no position
                    if not hasattr(obj, 'position'):
                        continue
                        
                    # Check distance from edges
                    distance_to_edge = min(
                        obj.position[0],               # Distance to left edge
                        1 - obj.position[0],           # Distance to right edge
                        obj.position[1],               # Distance to top edge
                        1 - obj.position[1]            # Distance to bottom edge
                    )
                    
                    if distance_to_edge < edge_threshold:
                        edge_objects.append(obj)
                
                # Sort by size (smaller objects are easier to move)
                edge_objects.sort(key=lambda obj: obj.size)
                
                # Move half of edge objects toward center
                for i, obj in enumerate(edge_objects[:len(edge_objects)//2]):
                    # Move toward center
                    new_x = (obj.position[0] + 0.5) / 2
                    new_y = (obj.position[1] + 0.5) / 2
                    
                    obj.position = (new_x, new_y)
        
        return fixed_scene