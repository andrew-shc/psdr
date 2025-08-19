from manim import *

class BasicScene(Scene):
    def construct(self):
        # Create left square
        left_square = Square(side_length=4, fill_opacity=0.05, stroke_opacity=0.3).shift(LEFT * 4)
        left_label = Text("Material Points", font_size=20).next_to(left_square, UP, buff=0.3)
        
        # Create right square
        right_square = Square(side_length=4, fill_opacity=0.05, stroke_opacity=0.3).shift(RIGHT * 4)
        right_label = Text("Spatial Points", font_size=20).next_to(right_square, UP, buff=0.3)
        
        # Add squares and labels immediately
        self.add(left_square, right_square, left_label, right_label)
        
        # Create labeled point for left square
        point_position = np.array([-4, -1, 0])  # Position relative to left square
        point = Dot(point_position, color=WHITE, radius=0.1)
        point_label = Text("C", font_size=24, color=WHITE).next_to(point, DOWN, buff=0.2)

        # Add point and label immediately
        self.add(point, point_label)


        # Create labeled point for right square
        right_point_position = np.array([4, -1, 0])  # Position relative to right square
        right_point = Dot(right_point_position, color=WHITE, radius=0.1)
        right_point_label = Text("C", font_size=24, color=WHITE).next_to(right_point, DOWN, buff=0.2)

        # Add right point and label immediately
        self.add(right_point, right_point_label)
        

        # Get evenly spaced points around the left square's perimeter
        side_length = 4  # Updated to match the actual square size
        half_side = side_length / 2
        square_center = LEFT * 4  # Center of left square
        circle_center = LEFT * 3 + UP * 1.0  # Center of circle (same as square)
        circle_radius = 0.8
        
        # Create a circle inside the left square
        circle = Circle(radius=circle_radius, color=GREEN, fill_opacity=0.05, stroke_opacity=0.4).shift(circle_center)
        self.add(circle)

        
        # Right square parameters
        right_square_center = RIGHT * 4  # Center of right square
        right_circle_center = RIGHT * 5 + UP * 1.0  # Center of circle for right square
        right_circle_radius = 0.8
        # Create a circle inside the right square
        right_circle = Circle(radius=right_circle_radius, color=GREEN, fill_opacity=0.05, stroke_opacity=0.4).shift(right_circle_center)
        self.add(right_circle)
        
        # Create N segments connecting to the left square with reflection off circle
        N = 40  # Number of segments
        segments = VGroup()
        intersection_points = VGroup()  # Store intersection points for visualization
        left_reflected_endpoints = {}  # Store reflected endpoints from left side
        
        # Camera ray parameters - restrict to a viewing cone
        camera_angle_range = PI / 2.5
        camera_center_angle = PI / 2
        start_angle = camera_center_angle - camera_angle_range / 2
        end_angle = camera_center_angle + camera_angle_range / 2
        
        for i in range(N):
            # Generate rays within the specified angular range
            angle = start_angle + i * (end_angle - start_angle) / (N - 1)
            
            # Calculate ray direction from point C
            ray_direction = np.array([np.cos(angle), np.sin(angle), 0])
            
            # Find intersection with square edges in this direction
            # Calculate intersection with each edge
            intersections = []
            
            # Square bounds relative to square center
            left_bound = square_center[0] - half_side
            right_bound = square_center[0] + half_side
            bottom_bound = square_center[1] - half_side
            top_bound = square_center[1] + half_side
            
            # Left edge (x = left_bound)
            if ray_direction[0] != 0:
                t = (left_bound - point_position[0]) / ray_direction[0]
                if t > 0:
                    y = point_position[1] + t * ray_direction[1]
                    if bottom_bound <= y <= top_bound:
                        intersections.append(point_position + t * ray_direction)
            
            # Right edge (x = right_bound)
            if ray_direction[0] != 0:
                t = (right_bound - point_position[0]) / ray_direction[0]
                if t > 0:
                    y = point_position[1] + t * ray_direction[1]
                    if bottom_bound <= y <= top_bound:
                        intersections.append(point_position + t * ray_direction)
            
            # Bottom edge (y = bottom_bound)
            if ray_direction[1] != 0:
                t = (bottom_bound - point_position[1]) / ray_direction[1]
                if t > 0:
                    x = point_position[0] + t * ray_direction[0]
                    if left_bound <= x <= right_bound:
                        intersections.append(point_position + t * ray_direction)
            
            # Top edge (y = top_bound)
            if ray_direction[1] != 0:
                t = (top_bound - point_position[1]) / ray_direction[1]
                if t > 0:
                    x = point_position[0] + t * ray_direction[0]
                    if left_bound <= x <= right_bound:
                        intersections.append(point_position + t * ray_direction)
            
            # Find the closest intersection (the edge point)
            if intersections:
                distances = [np.linalg.norm(pt - point_position) for pt in intersections]
                closest_idx = np.argmin(distances)
                edge_point = intersections[closest_idx]
            else:
                # Fallback: extend ray in direction
                edge_point = point_position + 5 * ray_direction
            
            # Check for intersection with circle
            # Vector from point to edge
            direction = edge_point - point_position
            direction_length = np.linalg.norm(direction)
            direction_unit = direction / direction_length
            
            # Vector from point to circle center
            to_circle = circle_center - point_position
            
            # Project to_circle onto direction to find closest approach
            projection_length = np.dot(to_circle, direction_unit)
            closest_point = point_position + projection_length * direction_unit
            
            # Distance from circle center to line
            distance_to_line = np.linalg.norm(circle_center - closest_point)
            
            # Check if line intersects circle
            if distance_to_line < circle_radius and 0 < projection_length < direction_length:
                # Calculate intersection point
                chord_half = np.sqrt(circle_radius**2 - distance_to_line**2)
                intersection_distance = projection_length - chord_half
                intersection_point = point_position + intersection_distance * direction_unit
                
                # Calculate reflection
                # Normal at intersection point
                normal = (intersection_point - circle_center) / circle_radius
                
                # Incident direction (from intersection to original target)
                incident = direction_unit
                
                # Reflected direction
                reflected = incident - 2 * np.dot(incident, normal) * normal
                
                # Find where reflected ray hits the square boundary
                # Start from intersection point and trace in reflected direction
                reflected_start = intersection_point
                
                # Find intersection with square edges
                # Square bounds relative to square center
                left_bound = square_center[0] - half_side
                right_bound = square_center[0] + half_side
                bottom_bound = square_center[1] - half_side
                top_bound = square_center[1] + half_side
                
                # Calculate intersection with each edge
                intersections = []
                
                # Left edge (x = left_bound)
                if reflected[0] != 0:
                    t = (left_bound - reflected_start[0]) / reflected[0]
                    if t > 0:
                        y = reflected_start[1] + t * reflected[1]
                        if bottom_bound <= y <= top_bound:
                            intersections.append(reflected_start + t * reflected)
                
                # Right edge (x = right_bound)
                if reflected[0] != 0:
                    t = (right_bound - reflected_start[0]) / reflected[0]
                    if t > 0:
                        y = reflected_start[1] + t * reflected[1]
                        if bottom_bound <= y <= top_bound:
                            intersections.append(reflected_start + t * reflected)
                
                # Bottom edge (y = bottom_bound)
                if reflected[1] != 0:
                    t = (bottom_bound - reflected_start[1]) / reflected[1]
                    if t > 0:
                        x = reflected_start[0] + t * reflected[0]
                        if left_bound <= x <= right_bound:
                            intersections.append(reflected_start + t * reflected)
                
                # Top edge (y = top_bound)
                if reflected[1] != 0:
                    t = (top_bound - reflected_start[1]) / reflected[1]
                    if t > 0:
                        x = reflected_start[0] + t * reflected[0]
                        if left_bound <= x <= right_bound:
                            intersections.append(reflected_start + t * reflected)
                
                # Find the closest intersection
                if intersections:
                    distances = [np.linalg.norm(pt - reflected_start) for pt in intersections]
                    closest_idx = np.argmin(distances)
                    reflected_end = intersections[closest_idx]
                else:
                    # Fallback: extend in reflected direction
                    reflected_end = intersection_point + 3 * reflected
                
                # Store the reflected endpoint for later use on right side
                left_reflected_endpoints[i] = reflected_end
                
                # Create two segments: point to intersection, intersection to square edge
                segment1 = Line(point_position, intersection_point, color=BLUE_B, stroke_width=2)
                segment2 = Line(intersection_point, reflected_end, color=BLUE_C, stroke_width=2)
                segments.add(segment1, segment2)

                # Add intersection point with circle (green)
                intersection_dot = Dot(intersection_point, color=GREEN, radius=0.05)
                intersection_points.add(intersection_dot)

                # Add intersection point where red segment hits square edge (white)
                red_end_dot = Dot(reflected_end, color=BLUE, radius=0.05)
                intersection_points.add(red_end_dot)
            else:
                # No intersection, create direct line
                segment = Line(point_position, edge_point, color=BLUE_B, stroke_width=2)
                segments.add(segment)
                
                # Add intersection point with square edge
                edge_dot = Dot(edge_point, color=BLUE, radius=0.05)
                intersection_points.add(edge_dot)
        
        # =============== RIGHT SQUARE SETUP ===============
        
        # Create N segments for right square with reflection off circle
        right_segments = VGroup()
        right_intersection_points = VGroup()  # Store intersection points for right square
        
        # Camera ray parameters for right square - restrict to a viewing cone
        right_camera_angle_range = PI / 2.5
        right_camera_center_angle = PI / 2
        right_start_angle = right_camera_center_angle - right_camera_angle_range / 2
        right_end_angle = right_camera_center_angle + right_camera_angle_range / 2
        
        for i in range(N):
            # Generate rays within the specified angular range
            angle = right_start_angle + i * (right_end_angle - right_start_angle) / (N - 1)
            
            # Calculate ray direction from right point D
            ray_direction = np.array([np.cos(angle), np.sin(angle), 0])
            
            # Find intersection with right square edges in this direction
            # Calculate intersection with each edge
            intersections = []
            
            # Right square bounds relative to square center
            left_bound = right_square_center[0] - half_side
            right_bound = right_square_center[0] + half_side
            bottom_bound = right_square_center[1] - half_side
            top_bound = right_square_center[1] + half_side
            
            # Left edge (x = left_bound)
            if ray_direction[0] != 0:
                t = (left_bound - right_point_position[0]) / ray_direction[0]
                if t > 0:
                    y = right_point_position[1] + t * ray_direction[1]
                    if bottom_bound <= y <= top_bound:
                        intersections.append(right_point_position + t * ray_direction)
            
            # Right edge (x = right_bound)
            if ray_direction[0] != 0:
                t = (right_bound - right_point_position[0]) / ray_direction[0]
                if t > 0:
                    y = right_point_position[1] + t * ray_direction[1]
                    if bottom_bound <= y <= top_bound:
                        intersections.append(right_point_position + t * ray_direction)
            
            # Bottom edge (y = bottom_bound)
            if ray_direction[1] != 0:
                t = (bottom_bound - right_point_position[1]) / ray_direction[1]
                if t > 0:
                    x = right_point_position[0] + t * ray_direction[0]
                    if left_bound <= x <= right_bound:
                        intersections.append(right_point_position + t * ray_direction)
            
            # Top edge (y = top_bound)
            if ray_direction[1] != 0:
                t = (top_bound - right_point_position[1]) / ray_direction[1]
                if t > 0:
                    x = right_point_position[0] + t * ray_direction[0]
                    if left_bound <= x <= right_bound:
                        intersections.append(right_point_position + t * ray_direction)
            
            # Find the closest intersection (the edge point)
            if intersections:
                distances = [np.linalg.norm(pt - right_point_position) for pt in intersections]
                closest_idx = np.argmin(distances)
                edge_point = intersections[closest_idx]
            else:
                # Fallback: extend ray in direction
                edge_point = right_point_position + 5 * ray_direction
            
            # Check for intersection with right circle
            # Vector from point to edge
            direction = edge_point - right_point_position
            direction_length = np.linalg.norm(direction)
            direction_unit = direction / direction_length
            
            # Vector from point to circle center
            to_circle = right_circle_center - right_point_position
            
            # Project to_circle onto direction to find closest approach
            projection_length = np.dot(to_circle, direction_unit)
            closest_point = right_point_position + projection_length * direction_unit
            
            # Distance from circle center to line
            distance_to_line = np.linalg.norm(right_circle_center - closest_point)
            
            # Check if line intersects circle
            if distance_to_line < right_circle_radius and 0 < projection_length < direction_length:
                # Calculate intersection point
                chord_half = np.sqrt(right_circle_radius**2 - distance_to_line**2)
                intersection_distance = projection_length - chord_half
                intersection_point = right_point_position + intersection_distance * direction_unit
                
                # Calculate reflection
                # Normal at intersection point
                normal = (intersection_point - right_circle_center) / right_circle_radius
                
                # Incident direction (from intersection to original target)
                incident = direction_unit
                
                # Reflected direction
                reflected = incident - 2 * np.dot(incident, normal) * normal
                
                # Find where reflected ray hits the right square boundary
                # Start from intersection point and trace in reflected direction
                reflected_start = intersection_point
                
                # Calculate intersection with each edge
                intersections = []
                
                # Left edge (x = left_bound)
                if reflected[0] != 0:
                    t = (left_bound - reflected_start[0]) / reflected[0]
                    if t > 0:
                        y = reflected_start[1] + t * reflected[1]
                        if bottom_bound <= y <= top_bound:
                            intersections.append(reflected_start + t * reflected)
                
                # Right edge (x = right_bound)
                if reflected[0] != 0:
                    t = (right_bound - reflected_start[0]) / reflected[0]
                    if t > 0:
                        y = reflected_start[1] + t * reflected[1]
                        if bottom_bound <= y <= top_bound:
                            intersections.append(reflected_start + t * reflected)
                
                # Bottom edge (y = bottom_bound)
                if reflected[1] != 0:
                    t = (bottom_bound - reflected_start[1]) / reflected[1]
                    if t > 0:
                        x = reflected_start[0] + t * reflected[0]
                        if left_bound <= x <= right_bound:
                            intersections.append(reflected_start + t * reflected)
                
                # Top edge (y = top_bound)
                if reflected[1] != 0:
                    t = (top_bound - reflected_start[1]) / reflected[1]
                    if t > 0:
                        x = reflected_start[0] + t * reflected[0]
                        if left_bound <= x <= right_bound:
                            intersections.append(reflected_start + t * reflected)
                
                # Find the closest intersection
                if intersections:
                    distances = [np.linalg.norm(pt - reflected_start) for pt in intersections]
                    closest_idx = np.argmin(distances)
                    reflected_end = intersections[closest_idx]
                else:
                    # Fallback: extend in reflected direction
                    reflected_end = intersection_point + 3 * reflected
                
                # Create two segments: point to intersection, intersection to square edge (consistent colors)
                segment1 = Line(right_point_position, intersection_point, color=BLUE_B, stroke_width=2)
                segment2 = Line(intersection_point, reflected_end, color=BLUE_C, stroke_width=2)
                right_segments.add(segment1, segment2)
                
                # Add intersection point with circle (green - same as left)
                intersection_dot = Dot(intersection_point, color=GREEN, radius=0.05)
                right_intersection_points.add(intersection_dot)

                # Add intersection point where reflected segment hits square edge (blue - same as left)
                red_end_dot = Dot(reflected_end, color=BLUE, radius=0.05)
                right_intersection_points.add(red_end_dot)
            else:
                # No intersection, create direct line (consistent color)
                segment = Line(right_point_position, edge_point, color=BLUE_B, stroke_width=2)
                right_segments.add(segment)
                
                # Add intersection point with square edge (blue - same as left)
                edge_dot = Dot(edge_point, color=BLUE, radius=0.05)
                right_intersection_points.add(edge_dot)
        
        # =============== ANIMATION SEQUENCE ===============
        # Show all intersection points from both squares
        all_intersection_points = VGroup(intersection_points, right_intersection_points)
        self.add(all_intersection_points)
        self.wait(1)
        
        # Show all segments from both squares
        all_segments = VGroup(segments, right_segments)
        self.play(Create(all_segments))
        
        # Ensure points and labels are always visible
        self.bring_to_front(point, point_label, right_point, right_point_label)
        self.wait(1)
        
        # =============== CONTINUOUSLY MOVE RIGHT CIRCLE ===============
        # Create updater functions for dynamic intersection points and segments
        circle_intersection_dots = VGroup()
        square_intersection_dots = VGroup()
        dynamic_segments = VGroup()
        
        # Keep track of which rays intersect the circle vs go directly to square
        rays_that_intersect_circle = []
        direct_rays_to_square = []
        original_square_intersections = {}  # Store original square intersection points
        
        # Identify which segments need to be dynamic vs static AND store original intersections
        for i in range(N):
            angle = right_start_angle + i * (right_end_angle - right_start_angle) / (N - 1)
            ray_direction = np.array([np.cos(angle), np.sin(angle), 0])
            
            # Find square intersection (same calculation as original)
            intersections = []
            left_bound = right_square_center[0] - half_side
            right_bound = right_square_center[0] + half_side
            bottom_bound = right_square_center[1] - half_side
            top_bound = right_square_center[1] + half_side
            
            if ray_direction[0] != 0:
                t = (left_bound - right_point_position[0]) / ray_direction[0]
                if t > 0:
                    y = right_point_position[1] + t * ray_direction[1]
                    if bottom_bound <= y <= top_bound:
                        intersections.append(right_point_position + t * ray_direction)
            
            if ray_direction[0] != 0:
                t = (right_bound - right_point_position[0]) / ray_direction[0]
                if t > 0:
                    y = right_point_position[1] + t * ray_direction[1]
                    if bottom_bound <= y <= top_bound:
                        intersections.append(right_point_position + t * ray_direction)
            
            if ray_direction[1] != 0:
                t = (bottom_bound - right_point_position[1]) / ray_direction[1]
                if t > 0:
                    x = right_point_position[0] + t * ray_direction[0]
                    if left_bound <= x <= right_bound:
                        intersections.append(right_point_position + t * ray_direction)
            
            if ray_direction[1] != 0:
                t = (top_bound - right_point_position[1]) / ray_direction[1]
                if t > 0:
                    x = right_point_position[0] + t * ray_direction[0]
                    if left_bound <= x <= right_bound:
                        intersections.append(right_point_position + t * ray_direction)
            
            if intersections:
                distances = [np.linalg.norm(pt - right_point_position) for pt in intersections]
                closest_idx = np.argmin(distances)
                edge_point = intersections[closest_idx]
            else:
                edge_point = right_point_position + 5 * ray_direction
            
            # Store the original square intersection point for this ray
            original_square_intersections[i] = edge_point
            
            # Check if this ray intersects the circle
            direction = edge_point - right_point_position
            direction_length = np.linalg.norm(direction)
            direction_unit = direction / direction_length
            to_circle = right_circle_center - right_point_position
            projection_length = np.dot(to_circle, direction_unit)
            closest_point = right_point_position + projection_length * direction_unit
            distance_to_line = np.linalg.norm(right_circle_center - closest_point)
            
            if distance_to_line < right_circle_radius and 0 < projection_length < direction_length:
                rays_that_intersect_circle.append(i)
            else:
                direct_rays_to_square.append(i)
        
        # Remove only the segments that intersect with the circle (keep direct segments)
        right_segments_to_keep = VGroup()
        right_intersections_to_keep = VGroup()
        
        segment_index = 0
        intersection_index = 0
        
        for i in range(N):
            if i in direct_rays_to_square:
                # Keep this segment and intersection (direct to square)
                if segment_index < len(right_segments.submobjects):
                    right_segments_to_keep.add(right_segments.submobjects[segment_index])
                    segment_index += 1
                if intersection_index < len(right_intersection_points.submobjects):
                    right_intersections_to_keep.add(right_intersection_points.submobjects[intersection_index])
                    intersection_index += 1
            else:
                # Skip segments that intersect circle (will be replaced by dynamic ones)
                if i in rays_that_intersect_circle:
                    segment_index += 2  # Skip both segments (to circle and reflected)
                    intersection_index += 2  # Skip both intersection points
        
        # Remove all right segments and intersections, then add back only the ones we want to keep
        self.remove(right_segments, right_intersection_points)
        
        # We'll need to handle direct segments dynamically too, so let's create dynamic versions for all
        dynamic_direct_segments = VGroup()
        dynamic_direct_intersection_dots = VGroup()
        new_rays_segments = VGroup()  # For newly generated rays
        new_rays_intersection_dots = VGroup()
        new_rays_circle_intersections = VGroup()  # For circle intersections of new rays
        
        # Store information about new rays for tracking
        new_rays_that_intersect_circle = []
        new_rays_direct_to_square = []
        new_rays_original_intersections = {}
        
        def update_right_circle_intersections(mob):
            # Clear previous dynamic elements from scene and groups
            self.remove(*circle_intersection_dots.submobjects)
            self.remove(*square_intersection_dots.submobjects)
            self.remove(*dynamic_segments.submobjects)
            self.remove(*dynamic_direct_segments.submobjects)
            self.remove(*dynamic_direct_intersection_dots.submobjects)
            self.remove(*new_rays_segments.submobjects)
            self.remove(*new_rays_intersection_dots.submobjects)
            self.remove(*new_rays_circle_intersections.submobjects)
            circle_intersection_dots.remove(*circle_intersection_dots.submobjects)
            square_intersection_dots.remove(*square_intersection_dots.submobjects)
            dynamic_segments.remove(*dynamic_segments.submobjects)
            dynamic_direct_segments.remove(*dynamic_direct_segments.submobjects)
            dynamic_direct_intersection_dots.remove(*dynamic_direct_intersection_dots.submobjects)
            new_rays_segments.remove(*new_rays_segments.submobjects)
            new_rays_intersection_dots.remove(*new_rays_intersection_dots.submobjects)
            new_rays_circle_intersections.remove(*new_rays_circle_intersections.submobjects)
            
            # Get current circle center position
            current_circle_center = right_circle.get_center()
            # Calculate how much the circle has moved from its original position
            circle_movement = current_circle_center - right_circle_center
            
            # Calculate how much the angular range should shift based on circle movement
            # Yellow rays should follow ALL segment Bs FROM THE RIGHT (outside the entire blue range)
            movement_distance = np.linalg.norm(circle_movement)
            angle_shift = min(movement_distance * 0.7, 0.5)  # Adjust this factor to control how fast new rays appear
            
            # Yellow rays should start AFTER the entire original viewing range has shifted
            # This ensures they're always to the right of ALL segment Bs
            original_angular_range = right_end_angle - right_start_angle
            angular_step = original_angular_range / (N - 1)  # Same angular step as original segments
            
            # Start yellow rays to the RIGHT (smaller angles) of the original range
            new_rays_end_angle = right_start_angle # - angular_step  # One step to the right of original range
            new_rays_start_angle = new_rays_end_angle + angle_shift  # Extend further right (smaller angles)
            
            # Calculate number of yellow rays to match original density
            new_rays_count = max(0, int(angle_shift / angular_step))  # Same angular spacing as originals
            
            # Generate yellow rays to fill the RIGHT side (smaller angles) as the circle moves left
            for j in range(new_rays_count):
                # Generate new ray angle
                angle = new_rays_start_angle + j * (new_rays_end_angle - new_rays_start_angle) / new_rays_count if new_rays_count >= 1 else new_rays_start_angle
                ray_direction = np.array([np.cos(angle), np.sin(angle), 0])
                
                # Find intersection with right square edges
                intersections = []
                left_bound = right_square_center[0] - half_side
                right_bound = right_square_center[0] + half_side
                bottom_bound = right_square_center[1] - half_side
                top_bound = right_square_center[1] + half_side
                
                # Calculate intersections with each edge
                if ray_direction[0] != 0:
                    t = (left_bound - right_point_position[0]) / ray_direction[0]
                    if t > 0:
                        y = right_point_position[1] + t * ray_direction[1]
                        if bottom_bound <= y <= top_bound:
                            intersections.append(right_point_position + t * ray_direction)
                
                if ray_direction[0] != 0:
                    t = (right_bound - right_point_position[0]) / ray_direction[0]
                    if t > 0:
                        y = right_point_position[1] + t * ray_direction[1]
                        if bottom_bound <= y <= top_bound:
                            intersections.append(right_point_position + t * ray_direction)
                
                if ray_direction[1] != 0:
                    t = (bottom_bound - right_point_position[1]) / ray_direction[1]
                    if t > 0:
                        x = right_point_position[0] + t * ray_direction[0]
                        if left_bound <= x <= right_bound:
                            intersections.append(right_point_position + t * ray_direction)
                
                if ray_direction[1] != 0:
                    t = (top_bound - right_point_position[1]) / ray_direction[1]
                    if t > 0:
                        x = right_point_position[0] + t * ray_direction[0]
                        if left_bound <= x <= right_bound:
                            intersections.append(right_point_position + t * ray_direction)
                
                if intersections:
                    distances = [np.linalg.norm(pt - right_point_position) for pt in intersections]
                    closest_idx = np.argmin(distances)
                    edge_point = intersections[closest_idx]
                    
                    # Check if new ray intersects with current circle position
                    direction = edge_point - right_point_position
                    direction_length = np.linalg.norm(direction)
                    direction_unit = direction / direction_length
                    
                    to_current_circle = current_circle_center - right_point_position
                    projection_length = np.dot(to_current_circle, direction_unit)
                    closest_point_on_ray = right_point_position + projection_length * direction_unit
                    distance_to_ray = np.linalg.norm(current_circle_center - closest_point_on_ray)
                    
                    # Check if circle intersects the new ray
                    if distance_to_ray < right_circle_radius and 0 < projection_length < direction_length:
                        # Ray intersects circle - create intersection and reflection
                        chord_half = np.sqrt(right_circle_radius**2 - distance_to_ray**2)
                        intersection_distance = projection_length - chord_half
                        intersection_point = right_point_position + intersection_distance * direction_unit
                        
                        # Calculate reflection with proper square intersection
                        normal = (intersection_point - current_circle_center) / right_circle_radius
                        incident = direction_unit
                        reflected = incident - 2 * np.dot(incident, normal) * normal
                        
                        # Find where reflected ray hits the square boundary
                        reflected_intersections = []
                        
                        # Check intersection with each edge
                        if reflected[0] != 0:
                            t = (left_bound - intersection_point[0]) / reflected[0]
                            if t > 0:
                                y = intersection_point[1] + t * reflected[1]
                                if bottom_bound <= y <= top_bound:
                                    reflected_intersections.append(intersection_point + t * reflected)
                        
                        if reflected[0] != 0:
                            t = (right_bound - intersection_point[0]) / reflected[0]
                            if t > 0:
                                y = intersection_point[1] + t * reflected[1]
                                if bottom_bound <= y <= top_bound:
                                    reflected_intersections.append(intersection_point + t * reflected)
                        
                        if reflected[1] != 0:
                            t = (bottom_bound - intersection_point[1]) / reflected[1]
                            if t > 0:
                                x = intersection_point[0] + t * reflected[0]
                                if left_bound <= x <= right_bound:
                                    reflected_intersections.append(intersection_point + t * reflected)
                        
                        if reflected[1] != 0:
                            t = (top_bound - intersection_point[1]) / reflected[1]
                            if t > 0:
                                x = intersection_point[0] + t * reflected[0]
                                if left_bound <= x <= right_bound:
                                    reflected_intersections.append(intersection_point + t * reflected)
                        
                        # Find the closest intersection
                        if reflected_intersections:
                            distances = [np.linalg.norm(pt - intersection_point) for pt in reflected_intersections]
                            closest_idx = np.argmin(distances)
                            reflected_end = reflected_intersections[closest_idx]
                        else:
                            # Fallback: extend in reflected direction
                            reflected_end = intersection_point + 3 * reflected
                        
                        # Create yellow segments for new rays
                        segment1 = Line(right_point_position, intersection_point, color=YELLOW, stroke_width=2)
                        segment2 = Line(intersection_point, reflected_end, color=YELLOW, stroke_width=2)
                        new_rays_segments.add(segment1, segment2)
                        
                        # Add intersection points
                        circle_dot = Dot(intersection_point, color=GREEN, radius=0.05)
                        square_dot = Dot(reflected_end, color=YELLOW, radius=0.05)
                        new_rays_intersection_dots.add(circle_dot, square_dot)
                    else:
                        # Direct ray to square
                        segment = Line(right_point_position, edge_point, color=YELLOW, stroke_width=2)
                        new_rays_segments.add(segment)
                        
                        edge_dot = Dot(edge_point, color=YELLOW, radius=0.05)
                        new_rays_intersection_dots.add(edge_dot)
            
            # Handle rays that originally intersected the circle
            for i in rays_that_intersect_circle:
                # Use the ORIGINAL square intersection point (stored earlier)
                edge_point = original_square_intersections[i]
                
                # Calculate original intersection with circle (before movement)
                direction = edge_point - right_point_position
                direction_length = np.linalg.norm(direction)
                direction_unit = direction / direction_length
                
                to_original_circle = right_circle_center - right_point_position
                projection_length = np.dot(to_original_circle, direction_unit)
                closest_point = right_point_position + projection_length * direction_unit
                distance_to_line = np.linalg.norm(right_circle_center - closest_point)
                
                # Calculate intersection point with circle at original position
                if distance_to_line < right_circle_radius and 0 < projection_length < direction_length:
                    chord_half = np.sqrt(right_circle_radius**2 - distance_to_line**2)
                    intersection_distance = projection_length - chord_half
                    original_intersection_point = right_point_position + intersection_distance * direction_unit
                    
                    # Move the intersection point WITH the circle (maintain relative position)
                    current_intersection_point = original_intersection_point + circle_movement
                    
                    # Use the CORRESPONDING left side's reflected endpoint, translated to right square
                    left_reflected_endpoint = left_reflected_endpoints[i]
                    # Translate from left square coordinate system to right square coordinate system
                    translation = RIGHT * 4 - LEFT * 4  # Translation vector from left to right
                    reflected_end_point = left_reflected_endpoint + translation
                    
                    # Check if reflected segment intersects with the circle
                    # Vector from circle intersection to reflected endpoint
                    reflected_direction = reflected_end_point - current_intersection_point
                    reflected_length = np.linalg.norm(reflected_direction)
                    
                    if reflected_length > 0:
                        reflected_unit = reflected_direction / reflected_length
                        
                        # Vector from reflected ray start to current circle center
                        to_current_circle = current_circle_center - current_intersection_point
                        
                        # Project circle center onto reflected ray
                        projection_on_reflected = np.dot(to_current_circle, reflected_unit)
                        closest_point_on_reflected = current_intersection_point + projection_on_reflected * reflected_unit
                        
                        # Distance from circle center to reflected ray
                        distance_to_reflected_ray = np.linalg.norm(current_circle_center - closest_point_on_reflected)
                        
                        # Check if reflected ray intersects circle and the intersection is within the segment
                        reflects_through_circle = (distance_to_reflected_ray < right_circle_radius and 
                                                 0 < projection_on_reflected < reflected_length)
                    else:
                        reflects_through_circle = False
                    
                    # Choose colors based on whether reflected segment goes through circle
                    if reflects_through_circle:
                        segment1_color = RED
                        segment2_color = RED
                        dot_colors = RED
                    else:
                        segment1_color = BLUE_B
                        segment2_color = BLUE_C
                        dot_colors = BLUE
                    
                    # Create segments with appropriate colors
                    segment1 = Line(right_point_position, current_intersection_point, color=segment1_color, stroke_width=2)
                    segment2 = Line(current_intersection_point, reflected_end_point, color=segment2_color, stroke_width=2)
                    dynamic_segments.add(segment1, segment2)
                    
                    # Add intersection points with appropriate colors
                    circle_dot = Dot(current_intersection_point, color=GREEN, radius=0.05)  # Circle intersection always green
                    square_dot = Dot(reflected_end_point, color=dot_colors, radius=0.05)  # Square intersection red if bisects
                    circle_intersection_dots.add(circle_dot)
                    square_intersection_dots.add(square_dot)
            
            # Handle rays that originally went direct to square (check if moving circle now intersects them)
            for i in direct_rays_to_square:
                # Get the original square intersection point for this direct ray
                edge_point = original_square_intersections[i]
                
                # Check if the moving circle now intersects this direct segment
                direction = edge_point - right_point_position
                direction_length = np.linalg.norm(direction)
                direction_unit = direction / direction_length
                
                # Vector from ray start to current circle center
                to_current_circle = current_circle_center - right_point_position
                
                # Project circle center onto ray direction
                projection_length = np.dot(to_current_circle, direction_unit)
                closest_point_on_ray = right_point_position + projection_length * direction_unit
                
                # Distance from circle center to ray line
                distance_to_ray = np.linalg.norm(current_circle_center - closest_point_on_ray)
                
                # Check if circle intersects the direct segment
                circle_intersects_direct = (distance_to_ray < right_circle_radius and 
                                          0 < projection_length < direction_length)
                
                # Choose color based on intersection
                if circle_intersects_direct:
                    segment_color = RED
                    dot_color = RED
                else:
                    segment_color = BLUE_B
                    dot_color = BLUE
                
                # Create direct segment with appropriate color
                direct_segment = Line(right_point_position, edge_point, color=segment_color, stroke_width=2)
                dynamic_direct_segments.add(direct_segment)
                
                # Add intersection dot with appropriate color
                edge_dot = Dot(edge_point, color=dot_color, radius=0.05)
                dynamic_direct_intersection_dots.add(edge_dot)
            
            # Add updated elements to scene
            self.add(circle_intersection_dots, square_intersection_dots, dynamic_segments, 
                    dynamic_direct_segments, dynamic_direct_intersection_dots, 
                    new_rays_segments, new_rays_intersection_dots, new_rays_circle_intersections)
        
        # Add updater to continuously update intersections
        right_circle.add_updater(update_right_circle_intersections)
        
        # Move the circle continuously to the left by 1.5 units over 3 seconds
        self.play(
            right_circle.animate.shift(LEFT * 2.0),
            run_time=3,
            rate_func=linear
        )
        
        # Remove updater and clean up dynamic elements
        right_circle.remove_updater(update_right_circle_intersections)
        
        # Final cleanup - ensure all dynamic elements are properly in scene for fadeout
        self.add(circle_intersection_dots, square_intersection_dots, dynamic_segments, 
                dynamic_direct_segments, dynamic_direct_intersection_dots,
                new_rays_segments, new_rays_intersection_dots, new_rays_circle_intersections)
        self.bring_to_front(point, point_label, right_point, right_point_label)
        self.wait(1)
        
        # Fade out everything
        # self.play(FadeOut(left_square), FadeOut(right_square), 
        #          FadeOut(left_label), FadeOut(right_label),
        #          FadeOut(point), FadeOut(point_label), 
        #          FadeOut(circle), FadeOut(right_circle),
        #          FadeOut(right_point), FadeOut(right_point_label),
        #          FadeOut(segments), FadeOut(intersection_points),
        #          FadeOut(circle_intersection_dots), FadeOut(square_intersection_dots), FadeOut(dynamic_segments),
        #          FadeOut(dynamic_direct_segments), FadeOut(dynamic_direct_intersection_dots),
        #          FadeOut(new_rays_segments), FadeOut(new_rays_intersection_dots), FadeOut(new_rays_circle_intersections))
