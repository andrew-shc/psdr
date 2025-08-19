# Manim Animations Project

This project contains a collection of animations created using the Manim library. It serves as a demonstration of how to structure a Manim project with separate files for scenes and utility functions.

## Project Structure

```
manim-animations
├── src
│   ├── main.py          # Entry point for the Manim animations
│   ├── scenes
│   │   └── basic_scene.py  # Contains the BasicScene class for animations
│   └── utils
│       └── helpers.py   # Utility functions for creating animations
├── requirements.txt      # Lists the dependencies for the project
├── manim.cfg             # Configuration settings for Manim
└── README.md             # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd manim-animations
   ```

2. **Install dependencies:**
   Make sure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Configure Manim:**
   Adjust the settings in `manim.cfg` as needed for your environment.

## Usage

To run the animations, execute the following command in your terminal:
```
manim src/scenes/basic_scene.py BasicScene
```

This will render the animations defined in the `BasicScene` class located in `basic_scene.py`.

## Contributing

Feel free to contribute by adding new scenes or utility functions. Make sure to follow the project structure for consistency.