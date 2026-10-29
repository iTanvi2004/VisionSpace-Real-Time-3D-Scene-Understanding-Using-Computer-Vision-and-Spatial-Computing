"""
VisionSpace - Semantic Scene Understanding Module
Analyzes spatial relations (on top of, left of, behind, adjacent to),
classifies scene category (indoor/outdoor context), and synthesizes natural language spatial summaries.
"""

from typing import List, Dict, Any


class SemanticSceneUnderstander:
    """Provides high-level spatial reasoning and semantic context interpretation."""

    def __init__(self):
        # Context object clusters
        self.indoor_office_cues = {"laptop", "keyboard", "mouse", "chair", "tv", "book", "cell phone"}
        self.indoor_living_cues = {"couch", "sofa", "bed", "dining table", "tv", "potted plant", "clock"}
        self.indoor_kitchen_cues = {"bottle", "cup", "fork", "knife", "spoon", "bowl", "microwave", "sink"}
        self.outdoor_traffic_cues = {"car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "stop sign"}
        self.outdoor_nature_cues = {"tree", "bird", "dog", "cat", "horse"}

    def understand_scene(self, spatial_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Derives high-level semantic scene understanding from detected spatial objects.
        """
        if not spatial_objects:
            return {
                "scene_type": "Clear Environment",
                "summary": "No dominant objects detected in immediate visual field.",
                "relationships": [],
                "hazard_warnings": [],
                "nearest_object": None,
                "object_count": 0
            }

        # 1. Determine Context / Scene Category
        scene_type = self._classify_scene_type(spatial_objects)

        # 2. Extract Spatial Relationships between objects
        relationships = self._infer_relationships(spatial_objects)

        # 3. Detect Proximity Hazards
        hazard_warnings = self._detect_hazards(spatial_objects)

        # 4. Generate Natural Language Summary
        summary = self._generate_summary(scene_type, spatial_objects, relationships)

        nearest_obj = spatial_objects[0] if spatial_objects else None

        return {
            "scene_type": scene_type,
            "summary": summary,
            "relationships": relationships,
            "hazard_warnings": hazard_warnings,
            "nearest_object": nearest_obj,
            "object_count": len(spatial_objects)
        }

    def _classify_scene_type(self, spatial_objects: List[Dict[str, Any]]) -> str:
        """Classify scene context based on object composition."""
        classes = {obj["class_name"].lower() for obj in spatial_objects}

        office_score = len(classes.intersection(self.indoor_office_cues))
        living_score = len(classes.intersection(self.indoor_living_cues))
        kitchen_score = len(classes.intersection(self.indoor_kitchen_cues))
        traffic_score = len(classes.intersection(self.outdoor_traffic_cues))

        scores = [
            ("Indoor Workspace / Study", office_score),
            ("Indoor Living Space", living_score),
            ("Dining / Kitchen Environment", kitchen_score),
            ("Outdoor Traffic / Street", traffic_score)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        best_type, best_score = scores[0]
        if best_score > 0:
            return best_type
        if "person" in classes:
            return "Pedestrian / Interactive Area"
        return "General Indoor/Outdoor Environment"

    def _infer_relationships(self, objects: List[Dict[str, Any]]) -> List[str]:
        """
        Computes pairwise 3D spatial relations between nearby objects.
        """
        relations = []
        n = len(objects)

        for i in range(n):
            for j in range(i + 1, n):
                obj_a = objects[i]
                obj_b = objects[j]

                pos_a = obj_a["position_3d"]
                pos_b = obj_b["position_3d"]
                name_a = obj_a["class_name"]
                name_b = obj_b["class_name"]

                dx = pos_b["x"] - pos_a["x"]
                dy = pos_b["y"] - pos_a["y"]
                dz = pos_b["z"] - pos_a["z"]
                dist = (dx**2 + dy**2 + dz**2)**0.5

                # Check proximity for relational reasoning (< 2.0 meters apart)
                if dist < 2.0:
                    # Check horizontal relation
                    if abs(dx) > 0.3:
                        h_rel = f"{name_b} is {abs(dx):.1f}m to the {'right' if dx > 0 else 'left'} of {name_a}"
                        relations.append(h_rel)

                    # Check vertical / supporting relation (Y is downward in camera frame)
                    if dy < -0.2 and abs(dx) < 0.5:
                        relations.append(f"{name_b} is positioned above {name_a}")
                    elif dy > 0.2 and abs(dx) < 0.5:
                        relations.append(f"{name_a} is positioned above {name_b}")

                    # Check depth relation
                    if abs(dz) > 0.4:
                        d_rel = f"{name_a} is {abs(dz):.1f}m in front of {name_b}" if dz > 0 else f"{name_b} is {abs(dz):.1f}m in front of {name_a}"
                        relations.append(d_rel)

        return relations[:6]  # Return top most informative relations

    def _detect_hazards(self, objects: List[Dict[str, Any]]) -> List[str]:
        """Flags obstacles in critical warning proximity."""
        hazards = []
        for obj in objects:
            if obj["proximity_status"] == "CRITICAL":
                hazards.append(
                    f"PROXIMITY ALERT: {obj['class_name'].upper()} at {obj['distance_m']}m directly in path!"
                )
        return hazards

    def _generate_summary(self, scene_type: str, objects: List[Dict[str, Any]],
                          relations: List[str]) -> str:
        """Constructs an intelligent natural language narrative of the environment."""
        closest = objects[0]
        c_name = closest["class_name"]
        c_dist = closest["distance_m"]
        c_bear = closest["bearing_deg"]

        dir_str = "ahead"
        if c_bear < -10:
            dir_str = f"{abs(c_bear):.0f} deg to the left"
        elif c_bear > 10:
            dir_str = f"{c_bear:.0f} deg to the right"

        summary = f"{scene_type} observed with {len(objects)} active obstacle(s). "
        summary += f"Closest entity is a {c_name} at {c_dist}m ({dir_str})."

        if relations:
            summary += f" Key relation: {relations[0]}."

        return summary
