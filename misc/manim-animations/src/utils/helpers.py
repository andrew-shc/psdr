def generate_circle(radius, color):
    from manim import Circle, FillColor

    circle = Circle(radius=radius, color=color)
    return circle

def generate_square(side_length, color):
    from manim import Square, FillColor

    square = Square(side_length=side_length, color=color)
    return square

def manage_color(color_name):
    from manim import color

    return getattr(color, color_name, None)